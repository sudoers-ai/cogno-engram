import pytest

from cogno_engram.types import TurnRecord


def _turn(n):
    return TurnRecord("sess", "s", n, f"msg {n}")


async def test_window_slides(buffer):
    for n in range(1, 6):
        await buffer.push("s", "sess", _turn(n))
    window = await buffer.window("s", "sess", size=3)
    assert [t.turn_n for t in window] == [3, 4, 5]


async def test_clear(buffer):
    await buffer.push("s", "sess", _turn(1))
    await buffer.clear("s", "sess")
    assert await buffer.window("s", "sess") == []


async def test_scope_isolation(buffer):
    await buffer.push("a", "sess", _turn(1))
    await buffer.push("b", "sess", _turn(2))
    assert [t.turn_n for t in await buffer.window("a", "sess")] == [1]


async def test_blank_scope_rejected(buffer):
    with pytest.raises(ValueError):
        await buffer.push("", "sess", _turn(1))
