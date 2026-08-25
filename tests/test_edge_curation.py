"""Who ASSERTED an edge decides whether the agent may say it.

A graph edge becomes a sentence the agent states about a person as if it knew. "Your son Pedro"
is either a kindness or an invention, and nothing downstream can tell which — so the difference
has to be carried by the edge itself. A host (or a staff member typing into an admin page)
**asserts**; an LLM extraction **proposes**, and a proposal is not spoken until a human says so.

The invariant every test here circles: **a `proposed` edge never reaches a prompt.** It is
enforced at the source (`walk`), repeated at the last step before text (`format_graph_context`),
and — the part that is easy to get wrong — it also does not decide what the walk can REACH.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.graph_context import format_graph_context
from cogno_engram.types import (
    EDGE_ACCEPTED,
    EDGE_PROPOSED,
    EDGE_REJECTED,
    VALID_EDGE_STATUS,
    VALID_PROXIMITY_RELATIONS,
    GraphEdge,
    sanitize_edge_status,
)

SCOPE = "tenant:acme|identity:jose"


def edge(source: str, target: str, relation: str, **kw) -> GraphEdge:
    return GraphEdge(SCOPE, source, target, relation, **kw)


@pytest.fixture
def kg() -> InMemoryGraph:
    return InMemoryGraph()


# ── the invariant ─────────────────────────────────────────────────────────

async def test_a_proposal_is_not_returned_by_a_walk(kg):
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF"))
    await kg.upsert_edge(edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED))

    walked = await kg.walk(SCOPE, "José")
    assert [(e.source, e.target) for e in walked] == [("José", "Pedro")]


async def test_a_proposal_does_not_decide_what_the_walk_can_REACH(kg):
    """The half that is easy to miss. Filtering the RESULT while letting a proposal route the
    traversal leaks the same unverified claim one hop further away: `Rex` stays out, but
    `Pastor Alemão` — reachable only through `Rex` — would walk straight into the prompt with
    nothing marking it as unreviewed."""
    await kg.upsert_edge(edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED))
    await kg.upsert_edge(edge("Rex", "Pastor Alemão", "BREED"))          # accepted!

    assert await kg.walk(SCOPE, "José", max_depth=3) == []


async def test_the_prompt_formatter_repeats_the_filter(kg):
    """Defence in depth, at the LAST step before the text becomes a prompt. The cost of the two
    layers disagreeing is asymmetric: a dropped edge is a missed kindness, a leaked one is the
    agent stating an unreviewed claim about a person as fact. A caller that builds an edge list
    by hand — a test, an admin view, a future store — gets the guarantee without knowing it."""
    handed_directly = [edge("José", "Pedro", "PARENT_OF"),
                       edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED)]
    rendered = format_graph_context(handed_directly)
    assert "Pedro" in rendered and "Rex" not in rendered


async def test_a_block_of_nothing_but_proposals_renders_NOTHING(kg):
    """Not a header with no lines — an empty string. An empty `[Knowledge Graph]` block spends
    prompt budget to say the agent knows nothing, and reads to a model as an invitation."""
    assert format_graph_context([edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED)]) == ""


# ── the verdict ───────────────────────────────────────────────────────────

async def test_accepting_puts_it_in_the_prompt_AND_makes_it_traversable(kg):
    await kg.upsert_edge(edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED))
    await kg.upsert_edge(edge("Rex", "Pastor Alemão", "BREED"))
    assert await kg.walk(SCOPE, "José", max_depth=3) == []

    assert await kg.set_edge_status(SCOPE, "José", "Rex", "OWNS_PET", EDGE_ACCEPTED)
    reached = {e.target for e in await kg.walk(SCOPE, "José", max_depth=3)}
    assert reached == {"Rex", "Pastor Alemão"}      # the hop opens the path behind it


async def test_rejecting_keeps_it_out_and_off_the_queue(kg):
    await kg.upsert_edge(edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED))
    assert await kg.set_edge_status(SCOPE, "José", "Rex", "OWNS_PET", EDGE_REJECTED)
    assert await kg.walk(SCOPE, "José") == []
    assert await kg.pending_edges(SCOPE) == []      # answered — no longer waiting on a human


async def test_a_verdict_on_an_edge_that_is_not_there_is_reported_not_invented(kg):
    assert await kg.set_edge_status(SCOPE, "José", "Ninguém", "OWNS_PET", EDGE_ACCEPTED) is False


async def test_the_queue_holds_only_what_is_waiting(kg):
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF"))                      # accepted
    await kg.upsert_edge(edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED))
    await kg.upsert_edge(edge("José", "Flamengo", "SUPPORTS", status=EDGE_REJECTED))

    assert [e.target for e in await kg.pending_edges(SCOPE)] == ["Rex"]


# ── re-assertion ──────────────────────────────────────────────────────────

async def test_a_re_extraction_does_not_wipe_what_a_human_typed(kg):
    """The LLM runs again every N turns; the note a person wrote runs once."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF",
                              attributes={"age": 8, "note": "joga futebol no sábado"}))
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", attributes={"age": 9}))

    kept = (await kg.walk(SCOPE, "José"))[0]
    assert kept.attributes == {"age": 9, "note": "joga futebol no sábado"}


async def test_a_deliberate_re_assertion_PROMOTES_a_proposal(kg):
    """A host writing the same edge on purpose is itself a verdict."""
    await kg.upsert_edge(edge("José", "Rex", "OWNS_PET", status=EDGE_PROPOSED))
    await kg.upsert_edge(edge("José", "Rex", "OWNS_PET"))            # default: accepted
    assert [e.target for e in await kg.walk(SCOPE, "José")] == ["Rex"]


async def test_a_re_extraction_can_NEVER_demote_a_verdict(kg):
    """The asymmetry that makes curation worth doing: if the next LLM pass could push an
    accepted fact back into the queue, every review would expire on its own and the person
    approving them would learn to stop."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF"))
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", status=EDGE_PROPOSED))

    assert [e.target for e in await kg.walk(SCOPE, "José")] == ["Pedro"]
    assert await kg.pending_edges(SCOPE) == []


# ── back-compat + vocabulary ──────────────────────────────────────────────

async def test_an_edge_that_says_nothing_behaves_exactly_as_before(kg):
    """Every caller written before this field existed keeps working, and `accepted` is why: the
    feature is inert until something explicitly proposes."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF"))
    walked = await kg.walk(SCOPE, "José")
    assert walked and walked[0].status == EDGE_ACCEPTED and walked[0].attributes == {}


@pytest.mark.parametrize("raw,expected", [
    # said, and readable
    ("proposed", EDGE_PROPOSED), ("  REJECTED ", EDGE_REJECTED), ("Accepted", EDGE_ACCEPTED),
    # NOT said → the old behaviour, which is what keeps every pre-curation caller working
    ("", EDGE_ACCEPTED), ("   ", EDGE_ACCEPTED), (None, EDGE_ACCEPTED),
    # said, and NOT readable → held for review, never asserted
    ("propsed", EDGE_PROPOSED), ("garbage", EDGE_PROPOSED), (3, EDGE_PROPOSED),
    (["proposed"], EDGE_PROPOSED),
])
def test_absent_and_garbled_are_not_the_same_thing(raw, expected):
    """The half a review had to point out. Folding a typo into `accepted` **inverts the
    caller's intent in the one direction this feature exists to prevent**: `status="propsed"`
    means someone meant to hold the edge, and it walked into the prompt as an assertion.

    An unreadable intent is held; the cost of guessing wrong that way is a queue entry."""
    assert sanitize_edge_status(raw) == expected


def test_a_garbled_status_cannot_reach_the_prompt_through_the_TYPE_either():
    """Normalised in `GraphEdge.__post_init__`, so neither store can be the one that gets it
    right. It used to run on the Postgres write path only: `status="Accepted"` walked fine out
    of Postgres and, in memory, was invisible to `walk()` AND absent from `pending_edges()` —
    an edge in a black hole, un-renderable and un-reviewable, with no error anywhere."""
    assert edge("José", "Rex", "OWNS_PET", status="Accepted").status == EDGE_ACCEPTED
    assert edge("José", "Rex", "OWNS_PET", status="propsed").status == EDGE_PROPOSED
    assert edge("José", "Rex", "OWNS_PET", attributes=None).attributes == {}


def test_the_vocabularies_are_closed_and_disjoint_from_each_other():
    assert VALID_EDGE_STATUS == {EDGE_ACCEPTED, EDGE_PROPOSED, EDGE_REJECTED}
    # proximity relations describe a PERSON's close world; they are not statuses and the two
    # sets must never be confused by a caller reading one field into the other
    assert not VALID_PROXIMITY_RELATIONS & VALID_EDGE_STATUS
    assert {"PARENT_OF", "OWNS_PET", "NICKNAME_OF"} <= VALID_PROXIMITY_RELATIONS


# ── the invariant, through the OTHER doors ────────────────────────────────

async def test_a_proposal_does_not_leak_through_neighbors(kg):
    """`walk` was filtered and `neighbors` was not — in BOTH adapters. The relation label is
    gone, but "this person is connected to José" is exactly the unverified claim the feature
    holds back, and `NodeContext` hands edges and neighbors to the same caller."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", status=EDGE_PROPOSED))
    assert await kg.neighbors(SCOPE, "José") == []

    await kg.set_edge_status(SCOPE, "José", "Pedro", "PARENT_OF", EDGE_ACCEPTED)
    assert [n.label for n in await kg.neighbors(SCOPE, "José")] == ["Pedro"]


async def test_a_proposal_does_not_leak_through_node_context(kg):
    """The adapter-parity hole: Postgres routed this through `walk` and was filtered; the
    in-memory twin read `self._edges` directly and was not. Same call, same edge, two answers —
    so a host rendering `node_context.edges` spoke the unreviewed claim in dev and not in prod."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", status=EDGE_PROPOSED))
    ctx = await kg.get_node_context(SCOPE, "José")
    assert ctx is not None and ctx.edges == [] and ctx.neighbors == []


async def test_the_curation_queue_drains_oldest_first(kg):
    """With a `limit` and no cursor, newest-first makes the OLDEST proposals — the ones a
    curator most needs to clear — permanently unreachable, and the queue never empties. The two
    adapters disagreed on this and the disagreement was invisible."""
    for i in range(5):
        await kg.upsert_edge(edge("José", f"Contato {i}", "FRIEND_OF", status=EDGE_PROPOSED))
    assert [e.target for e in await kg.pending_edges(SCOPE, limit=2)] == ["Contato 0", "Contato 1"]


# ── the detail actually reaches the prompt ────────────────────────────────

def test_the_attributes_are_RENDERED_not_just_stored():
    """The whole point of the field, and it was missing: stored, merged, migrated and
    round-tripped through both stores while the formatter emitted only `a --[R]--> b`. The
    type's docstring claimed the opposite and the README advertised it."""
    rendered = format_graph_context([
        edge("José", "Pedro", "PARENT_OF", attributes={"note": "joga futebol", "age": 8})])
    assert rendered.endswith("- José --[PARENT_OF]--> Pedro (age: 8; note: joga futebol)")


def test_an_edge_with_no_detail_renders_exactly_as_before():
    assert format_graph_context([edge("José", "Rex", "OWNS_PET")]).endswith(
        "- José --[OWNS_PET]--> Rex")


def test_a_typed_note_cannot_break_the_bullet_or_eat_the_block():
    """The value comes from a person typing into an admin field. A newline inside a bullet turns
    one fact into what reads as two, and one verbose note must not spend the whole budget."""
    noisy = format_graph_context([edge("José", "Pedro", "PARENT_OF",
                                       attributes={"note": "linha um\nlinha dois"})])
    assert "\n" not in noisy.split("\n", 1)[1]                 # one header, then ONE bullet
    long = format_graph_context([edge("José", "X", "PREFERS", attributes={"x": "y" * 500})])
    assert len(long.split("\n")[1]) < 200 and long.endswith("…)")


# ── the verdict is sticky, and that is a decision ─────────────────────────

async def test_a_rejected_edge_is_NOT_resurrected_by_the_next_extraction(kg):
    """`upsert_edge` cannot tell a deliberate correction from the LLM re-emitting the same edge,
    and it DEFAULTS to `accepted` — so promoting from `rejected` here would bring back every
    rejected edge on the next Tier-2 run. A rejection is a person saying the claim is wrong
    about this contact; undoing it takes a person, through `set_edge_status`."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF"))
    await kg.set_edge_status(SCOPE, "José", "Pedro", "PARENT_OF", EDGE_REJECTED)

    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF"))        # the extraction runs again
    assert await kg.walk(SCOPE, "José") == []
    assert await kg.pending_edges(SCOPE) == []

    assert await kg.set_edge_status(SCOPE, "José", "Pedro", "PARENT_OF", EDGE_ACCEPTED)
    assert [e.target for e in await kg.walk(SCOPE, "José")] == ["Pedro"]


# ── the two ways a verification found to break the invariant ──────────────

@pytest.mark.parametrize("not_a_verdict", [None, "", "   ", "acepted", "sim", 3, ["accepted"]])
async def test_a_MISSING_verdict_does_not_publish_the_edge(kg, not_a_verdict):
    """The inversion a review measured: a TYPO'd verdict was safely held as `proposed`, while a
    MISSING one published the unreviewed edge — and returned `True`, so the curation UI showed
    success.

    `sanitize_edge_status` answers a STORAGE question (an edge with no status predates the
    field, so `accepted` is back-compat). `set_edge_status` answers a VERDICT question, where
    there is no legacy caller and "absent" means nobody decided. Reusing one for the other is
    what put an unreviewed claim in the prompt."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", status=EDGE_PROPOSED))
    with pytest.raises(ValueError, match="not a verdict"):
        await kg.set_edge_status(SCOPE, "José", "Pedro", "PARENT_OF", not_a_verdict)
    assert await kg.walk(SCOPE, "José") == []
    assert [e.target for e in await kg.pending_edges(SCOPE)] == ["Pedro"]


async def test_a_PROPOSAL_cannot_modify_a_VERDICT_in_any_field(kg):
    """The gate covered `status` and not `attributes` — and `_detail` renders attributes into
    the prompt. Measured: a human asserts the relation, the next pass proposes the same edge
    carrying `{"note": "expelled from school for cheating"}`; the status correctly stays
    accepted and the queue stays empty, and the prompt says the note.

    Reviewed means reviewed AS IT STOOD. A proposal with something to add needs its own turn
    through the queue."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", attributes={"age": 8}))
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", status=EDGE_PROPOSED,
                              attributes={"note": "expulso da escola por cola"}))

    spoken = format_graph_context(await kg.walk(SCOPE, "José"))
    assert "expulso" not in spoken and "age: 8" in spoken
    assert await kg.pending_edges(SCOPE) == []      # ...and it did not sneak into the queue


async def test_the_queue_hands_out_COPIES_not_the_stored_edge(kg):
    """Postgres builds fresh rows; this returned live references, so a curation UI that edited a
    queue item published the edge in one store and did nothing in the other — the two
    disagreeing on exactly the invariant this feature is."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", status=EDGE_PROPOSED))
    queued = (await kg.pending_edges(SCOPE))[0]
    queued.status = EDGE_ACCEPTED               # a UI editing what it was handed
    assert await kg.walk(SCOPE, "José") == []   # ...changes nothing until it says so


async def test_a_non_positive_limit_means_EMPTY_in_both_stores(kg):
    """Postgres raises on a negative LIMIT and this truncated. A curator paginating with
    `limit = cap - shown` hits zero eventually."""
    await kg.upsert_edge(edge("José", "Pedro", "PARENT_OF", status=EDGE_PROPOSED))
    assert await kg.pending_edges(SCOPE, limit=0) == []
    assert await kg.pending_edges(SCOPE, limit=-1) == []


def test_the_curation_vocabulary_is_reachable_from_the_PACKAGE_ROOT():
    """The CHANGELOG advertises these as additions and every test above imports them from
    `cogno_engram.types` — so the package-root export could be missing and the suite would stay
    green over a symbol no host can import. Advertised-but-unreachable, with nothing to catch it."""
    import cogno_engram

    for name in ("EDGE_ACCEPTED", "EDGE_PROPOSED", "EDGE_REJECTED", "VALID_EDGE_STATUS",
                 "VALID_PROXIMITY_RELATIONS", "sanitize_edge_status", "require_edge_status"):
        assert hasattr(cogno_engram, name), f"{name} não é importável da raiz do pacote"
        assert name in cogno_engram.__all__, f"{name} fora do __all__"


async def test_a_proposal_still_creates_its_ENDPOINT_NODES_and_that_is_a_known_bound(kg):
    """Stated rather than fixed, because the bound is narrower than it looks.

    A proposed edge auto-creates both endpoints (parity with Postgres, so a sloppy extraction
    never dangles) and NODES carry no curation state — so `list_nodes`/`find_node` disclose the
    label of a person named only by an unreviewed edge. The PROMPT is not affected: every path
    that reaches a prompt goes through `walk`, which is filtered at the source, at the traversal
    and again at the formatter. What is exposed is the ADMIN surface, to a curator who is about
    to read the proposal anyway.

    Pinned so the next reader finds a decision instead of a surprise; closing it properly means
    giving nodes a status of their own, which is a bigger change than this one."""
    await kg.upsert_edge(edge("José", "Amante Secreta", "FRIEND_OF", status=EDGE_PROPOSED))
    assert {n.label for n in await kg.list_nodes(SCOPE)} == {"José", "Amante Secreta"}
    assert format_graph_context(await kg.walk(SCOPE, "José")) == ""      # never spoken
