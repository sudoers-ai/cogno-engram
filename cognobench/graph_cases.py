"""Synthetic knowledge-graph cases — build a small graph, walk it, check reach."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphCase:
    id: str
    nodes: list[tuple[str, str]]            # (label, node_type)
    edges: list[tuple[str, str, str]]       # (source, target, relation)
    start: str
    max_depth: int
    expect_relations: set[str]              # relations reachable within max_depth
    description: str = ""


CASES: list[GraphCase] = [
    GraphCase(
        id="vet_owner_pet_breed",
        nodes=[("José", "PERSON"), ("Rex", "ANIMAL"), ("Pastor Alemão", "CONCEPT")],
        edges=[("José", "Rex", "OWNS"), ("Rex", "Pastor Alemão", "BREED")],
        start="José", max_depth=2,
        expect_relations={"OWNS", "BREED"},
        description="2-hop reaches the breed via the pet",
    ),
    GraphCase(
        id="depth_limit_one_hop",
        nodes=[("José", "PERSON"), ("Rex", "ANIMAL"), ("Pastor Alemão", "CONCEPT")],
        edges=[("José", "Rex", "OWNS"), ("Rex", "Pastor Alemão", "BREED")],
        start="José", max_depth=1,
        expect_relations={"OWNS"},
        description="depth 1 stops at the pet",
    ),
    GraphCase(
        id="finance_employer_chain",
        nodes=[("Maria", "PERSON"), ("Acme", "ORG"), ("São Paulo", "PLACE")],
        edges=[("Maria", "Acme", "WORKS_AT"), ("Acme", "São Paulo", "LOCATED_IN")],
        start="Maria", max_depth=2,
        expect_relations={"WORKS_AT", "LOCATED_IN"},
    ),
    GraphCase(
        id="bidirectional_reach",
        nodes=[("Ana", "PERSON"), ("Bob", "PERSON")],
        edges=[("Ana", "Bob", "KNOWS")],
        start="Bob", max_depth=1,
        expect_relations={"KNOWS"},
        description="incoming edge reached from the target side",
    ),
    GraphCase(
        id="cycle_terminates",
        nodes=[("A", "CONCEPT"), ("B", "CONCEPT")],
        edges=[("A", "B", "R1"), ("B", "A", "R2")],
        start="A", max_depth=4,
        expect_relations={"R1", "R2"},
        description="a cycle must terminate, not hang",
    ),
    GraphCase(
        id="restaurant_dish_pref",
        nodes=[("Carlos", "PERSON"), ("Pizza", "OBJECT"), ("Calabresa", "CONCEPT")],
        edges=[("Carlos", "Pizza", "PREFERS"), ("Pizza", "Calabresa", "FLAVOUR")],
        start="Carlos", max_depth=2,
        expect_relations={"PREFERS", "FLAVOUR"},
    ),
]
