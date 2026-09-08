"""How long a persisted turn trace stays revisable.

One definition, read by BOTH adapters. The Postgres and in-memory stores are
hand-written copies of one rule, and the copies are the ones that drift — so the
number that decides whether a write is a revision or an overwrite lives here,
once, rather than beside each of them.
"""

from __future__ import annotations
# How long a stored turn trace stays REVISABLE. Past it, an incoming trace at the
# same ``(scope, session_id, turn_n)`` is refused rather than allowed to overwrite.
#
# This is a bound on "still the same turn", not a tuning knob. Everything that
# legitimately rewrites a trace does so inside the turn that produced it — the
# host's EGO/SUPEREGO correction loop, a re-voice, a backfill batch re-running its
# own rows — and those close in seconds to minutes. The overwrites that motivated
# the guard were NINE TO TWENTY-FOUR DAYS late: three orders of magnitude outside
# any window that could be called generous.
#
# The direction of the trade is deliberate. Refusing every overwrite outright is
# simpler, and it is what a strict reading of "never replace an older trace" asks
# for — but it would convert every legitimate same-turn revision into a NEW silent
# loss, and proving that set empty across an ecosystem means proving a negative.
# A window cannot introduce a loss that did not already exist: anything it admits,
# the old unconditional upsert admitted too. Pass 0 for strict refusal.
TRACE_REVISION_WINDOW_S = 3600.0
