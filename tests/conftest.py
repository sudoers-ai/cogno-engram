import pytest

from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def buffer() -> InMemoryBuffer:
    return InMemoryBuffer()


@pytest.fixture
def graph() -> InMemoryGraph:
    return InMemoryGraph()
