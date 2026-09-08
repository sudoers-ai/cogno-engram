"""Writes this library did NOT perform, counted and announced where they happen.

Two of engram's turn writes are deliberately conflict-tolerant, and both used to
absorb a collision in complete silence:

* ``save_turn`` is ``ON CONFLICT (scope, session_id, turn_n) DO NOTHING`` — a turn
  landing on an occupied coordinate is dropped, with no exception and no row.
* ``save_turn_trace`` is an upsert — a trace landing on an occupied coordinate
  overwrote whatever was there.

Silence is the defect. A production incident ran 48 h before anyone noticed that
turns were being discarded, because nothing in the process could say so: the
caller gets ``None`` either way, the log stays clean, and the only evidence is a
row that is missing from a table nobody counts.

**This module never raises, and callers must never make it raise.** These writes
run AFTER the reply has been delivered to the contact; an exception here would
take down a turn that already succeeded, which is a worse failure than the one
being reported. The fix is to make the loss VISIBLE, not to stop absorbing it.

Two channels, because they answer different questions:

* the **counter** (``counts()``) answers "is this happening, and how much" — it is
  process-local, monotonic, and cheap enough to read on every scrape;
* the **log event** answers "to whom, and when" — one named line per loss.

A host that wants the number in its own metrics pipeline registers an observer
with :func:`subscribe`; engram itself depends on no metrics library.

Nothing here records a scope, a session id or any turn content: the fields are a
tenant-free coordinate (``session``/``turn``) plus a digest of the scope, because
a scope's second segment is a contact's phone number and logs have no retention
deadline. The digest is de-identification of the flow, not encryption — it is
enough to correlate two losses as "the same conversation" and not enough to name
the conversation.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Callable, Dict, Optional

logger = logging.getLogger("cogno_engram.write_loss")

# ── the closed event vocabulary ───────────────────────────────────────────────
# A closed alphabet, for the same reason the rest of this codebase keeps them:
# a counter whose key set is open cannot be alerted on, because no dashboard can
# enumerate the keys it should be watching.

#: A turn was handed to ``save_turn`` and no row was written: its
#: ``(scope, session_id, turn_n)`` coordinate was already taken. The conversation
#: turn is GONE — this is the loud version of the ``DO NOTHING``.
TURN_DISCARDED = "turn_discarded_conflict"

#: ``save_turn`` was asked to allocate a coordinate and lost the race for it more
#: times than it is willing to retry. Distinct from ``TURN_DISCARDED``: nothing is
#: structurally wrong, the session is simply hotter than the retry budget.
TURN_ALLOCATION_EXHAUSTED = "turn_allocation_exhausted"

#: A trace was handed to ``save_turn_trace`` whose coordinate already held a
#: STRICTLY OLDER trace. The stored trace was kept and the incoming one dropped —
#: see ``PostgresStore.save_turn_trace`` for why that is the safe direction.
TRACE_OVERWRITE_REFUSED = "trace_overwrite_refused"

EVENTS = (TURN_DISCARDED, TURN_ALLOCATION_EXHAUSTED, TRACE_OVERWRITE_REFUSED)

#: Callback signature: ``(event, fields) -> None``. Fields are already redacted.
Observer = Callable[[str, Dict[str, object]], None]

_lock = threading.Lock()
_counts: Dict[str, int] = {e: 0 for e in EVENTS}
_observers: list[Observer] = []


def scope_digest(scope: str) -> str:
    """A short, stable, non-reversing handle for a scope.

    Twelve hex characters of SHA-256. Enough to say "these two losses are the same
    conversation"; not enough to say which conversation. An unsalted digest over a
    phone number is de-identification, not encryption — treat a stored one as
    pseudonymous personal data, the same ceiling the host's ``scope_sha`` carries.
    """
    return hashlib.sha256((scope or "").encode("utf-8")).hexdigest()[:12]


def record(event: str, *, scope: str = "", session: str = "", turn: Optional[int] = None,
           **extra: object) -> None:
    """Count and announce one write loss. Never raises.

    An unknown ``event`` is still logged and still counted (under its own key), on
    the principle that a miscounted loss beats a swallowed one — but ``EVENTS`` is
    the set a dashboard should be built from.
    """
    fields: Dict[str, object] = {}
    try:
        if scope:
            fields["scope_sha"] = scope_digest(scope)
        if session:
            fields["session"] = str(session)
        if turn is not None:
            fields["turn"] = turn
        fields.update(extra)
        with _lock:
            _counts[event] = _counts.get(event, 0) + 1
            observers = list(_observers)
        logger.error("event=%s %s", event,
                     " ".join(f"{k}={v}" for k, v in fields.items()))
        for obs in observers:
            try:
                obs(event, dict(fields))
            except Exception:  # noqa: BLE001 — an observer must never break a write path
                logger.warning("event=write_loss_observer_failed observer=%r", obs,
                               exc_info=True)
    except Exception:  # noqa: BLE001 — reporting a loss must never become a second failure
        logger.warning("event=write_loss_record_failed name=%s", event, exc_info=True)


def counts() -> Dict[str, int]:
    """Process-local, monotonic totals per event. Always contains every key in
    ``EVENTS`` (zeros included) — a metric that appears only once it is non-zero
    reads as "no data" exactly when it matters most."""
    with _lock:
        return dict(_counts)


def subscribe(observer: Observer) -> Callable[[], None]:
    """Register a callback for every loss; returns an unsubscribe callable."""
    with _lock:
        _observers.append(observer)

    def _off() -> None:
        with _lock:
            if observer in _observers:
                _observers.remove(observer)
    return _off


def reset() -> None:
    """Zero the counters and drop observers. For tests; production never calls it."""
    with _lock:
        # Back to exactly the closed alphabet at zero: an off-vocabulary key that
        # ``record`` tolerated is DROPPED, not carried into the next test as a
        # zero that looks like a real metric.
        _counts.clear()
        _counts.update({e: 0 for e in EVENTS})
        _observers.clear()
