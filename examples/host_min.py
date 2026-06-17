"""
examples/host_min.py — a minimal host wiring cogno-anima + cogno-engram end to end.

The runnable companion to ``docs/HOST_INTEGRATION.md``: a tiny host that lets a
conversation **perceive → route → remember**. cogno-anima provides the cognition
(NOUMENO/NER/ID/EGO/SUPEREGO); cogno-engram provides the memory substrate
(buffer + store + graph + hypnos consolidation). The host owns the glue: the
opaque ``scope``, recall-before-cognition, persist-after, and the consolidation.

Uses the zero-dependency in-memory engram adapters, so it runs with just Ollama —
swap three constructors for Postgres/Redis in production (see the docs).

Prereqs:  pip install -e ../cogno-core        # cogno-anima (sibling), editable
          a local Ollama at http://localhost:11434  (ollama pull mistral:latest)
Run:      python3 examples/host_min.py
"""

from __future__ import annotations

import asyncio

import httpx

from cogno_anima.llm import CachingEmbedder, OllamaBackend, OllamaEmbedder
from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.id import IDStage
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import PipelineContext, ToolResult

from cogno_engram import hypnos, rerank
from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore
from cogno_engram.types import RetrievalQuery, TurnRecord

MODEL = "mistral:latest"
BASE_URL = "http://localhost:11434"

VOICE_PROMPT = ("You are a warm, concise personal assistant. Use any known facts "
                "about the user when relevant. Reply in the user's language.")
EGO_PROMPT = "You are an assistant. If the user only chats, do not call a tool."
SCOPE_PROMPT = "A helpful personal assistant. In scope: chat, personal facts, preferences."


class NoToolsDispatcher:
    """The EGO needs a dispatcher; this host exposes no tools (chat-only demo)."""

    def tools_schema(self) -> list[dict]:
        return []

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(output="", ok=False, error=f"no tools ({name})")


class MemoryHost:
    def __init__(self, gen, text_backend, embedder) -> None:
        self._gen = gen
        self._text = text_backend
        self._embedder = embedder
        self._noumeno = Noumeno(embedder=embedder)
        self._ner = IntentAnalyzer()
        self._id = IDStage()
        self._ego = EgoStage()
        self._superego = SuperegoStage()
        # cogno-engram substrate (swap for Postgres/Redis in prod)
        self._store = InMemoryStore()
        self._buffer = InMemoryBuffer()
        self._kg = InMemoryGraph()
        self._sessions: dict[str, str] = {}   # scope → engram session id

    async def _session_id(self, scope: str) -> str:
        if scope not in self._sessions:
            self._sessions[scope] = (await self._store.create_session(scope)).id
        return self._sessions[scope]

    async def turn(self, scope: str, text: str) -> str:
        session_id = await self._session_id(scope)

        # ① RECALL — long-term memories + short-term window (engram)
        q = RetrievalQuery(text=text, embedding=await self._embedder.embed(text))
        recalled = rerank(await self._store.load_memories(scope, query=q, limit=20),
                          query_text=text, top_k=5)
        if recalled:
            print("   🧠 recalled:", [m.content for m in recalled])

        # ② COGNITION — anima pipeline (memories injected into the voice prompt)
        ctx = PipelineContext(user_input=text)
        ctx = await self._noumeno.process(ctx, self._gen)
        ctx = await self._ner.process(ctx, self._gen)
        ctx = await self._id.process(ctx, self._embedder)

        if ctx.id_result and ctx.id_result.triad_route == "EGO":
            ctx = await self._ego.process(ctx, self._text, NoToolsDispatcher(), system_prompt=EGO_PROMPT)
        voice_prompt = VOICE_PROMPT
        if recalled:
            voice_prompt += "\n[KNOWN ABOUT USER]\n" + "\n".join(f"- {m.content}" for m in recalled)
        ctx.superego_result = await self._superego.voice(ctx, self._text, voice_prompt=voice_prompt)
        reply = ctx.superego_result.response

        # ③ PERSIST + ④ Tier-1 micro (engram)
        turn = TurnRecord(
            session_id, scope, ctx.id_result.turn_number if ctx.id_result else 1, text, response=reply,
            goal=getattr(ctx.id_result, "active_goal", "") or "",
            goal_status=getattr(ctx.id_result, "goal_status", "") or "",
            sentiment=getattr(ctx.intent, "sentiment", "") or "",
            domains=list(getattr(ctx.intent, "domains", []) or []),
            pii_types=list(getattr(ctx.intent, "pii", []) or []))
        await self._store.save_turn(turn)
        await self._buffer.push(scope, session_id, turn)
        for m in hypnos.micro_consolidate(turn):
            m.embedding = await self._embedder.embed(m.content)
            await self._store.save_memory(m)
        return reply

    async def consolidate(self, scope: str) -> None:
        """Sleep-time consolidation (host would schedule this; here we call it directly)."""
        session_id = self._sessions.get(scope)
        if not session_id:
            return
        mems = await hypnos.periodic_consolidate(
            self._store, self._gen, scope=scope, session_id=session_id,
            embedder=self._embedder, kg=self._kg)
        print("   💤 consolidated:", [m.content for m in mems] or "(none)")


async def _ollama_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            return (await c.get(f"{BASE_URL}/")).status_code == 200
    except Exception:
        return False


async def main() -> None:
    if not await _ollama_up():
        print(f"Ollama not reachable at {BASE_URL} — start it (and `ollama pull {MODEL}`).")
        return

    gen = OllamaBackend(model=MODEL, base_url=BASE_URL, temperature=0.0, format="json")
    text = OllamaBackend(model=MODEL, base_url=BASE_URL, temperature=0.0)
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text", base_url=BASE_URL))
    host = MemoryHost(gen, text, embedder)
    scope = "demo-tenant/user-1"

    # Turn 1 states a preference; we consolidate; Turn 2 should recall it.
    for user_text in ["Oi! Meu nome é João e eu prefiro pagar tudo no pix.",
                      "Pode me lembrar como eu gosto de pagar?"]:
        print(f"\n👤 {user_text}")
        if "lembrar" in user_text:
            await host.consolidate(scope)        # sleep-time pass before the recall turn
        reply = await host.turn(scope, user_text)
        print(f"🤖 {reply}")


if __name__ == "__main__":
    asyncio.run(main())
