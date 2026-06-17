"""A tiny Ollama client for the gated LLM-consolidation dimension.

Self-contained (lazy ``httpx``), so the bench doesn't depend on cogno-anima. It
implements just the duck-typed ``generate`` that ``hypnos`` calls.
"""

from __future__ import annotations

from typing import Optional


class OllamaBackend:
    def __init__(self, model: str = "mistral:latest",
                 base_url: str = "http://localhost:11434", temperature: float = 0.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        import httpx
        payload = {
            "model": self.model, "system": system, "prompt": prompt,
            "stream": False, "format": "json", "think": False,
            "options": {"temperature": self.temperature},
        }
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        text = data.get("response") or data.get("thinking") or ""
        return text, int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))


async def is_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as c:
            return (await c.get(f"{base_url}/api/tags")).status_code == 200
    except Exception:
        return False


def has_model(models_json: dict, model: str) -> Optional[bool]:
    names = [m.get("name", "") for m in models_json.get("models", [])]
    return model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)
