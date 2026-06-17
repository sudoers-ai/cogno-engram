"""EngramBench — a self-contained quality harness for the cogno-engram substrate.

Decoupled from the library (not shipped in the wheel — `packages.find` only
includes `cogno_engram*`). Measures the substrate's three jobs: hybrid memory
retrieval (hit@1 against distractors), Tier-1 consolidation accuracy, and
knowledge-graph reachability. Runs against the zero-dependency in-memory adapter
with a deterministic hashing embedder, so it needs no model and no database.
"""
