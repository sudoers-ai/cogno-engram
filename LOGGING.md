# Logging — convenção desta lib

Esta biblioteca **emite** logs; o **host configura** (handlers, formato, nível,
contexto de tenant). Regras:

1. Use `logging.getLogger(__name__)` no topo do módulo. Nada de handlers,
   formatters, `basicConfig` ou um `get_logger` próprio.
2. Mensagem = só o fato de domínio, em `key=value`, sempre lazy:
   `logger.info("tier=2 event=consolidation_done extracted=%d", n)`.
   NÃO coloque tenant_id / timestamp / channel na mensagem — o host injeta
   via contextvars + Filter no root logger (carimbado em todo LogRecord).
3. Níveis:
   - **ERROR**  → nunca aqui; erro fatal vira exceção e propaga (host loga ERROR).
   - **WARNING**→ condição recuperada/tratada (fallback, parse coercion, verify falho).
   - **INFO**   → marco caro e raro; NÃO happy-path por request.
   - **DEBUG**  → trace de fidelidade total (prompt/raw/scores). DEV-ONLY,
                  jamais ligado em produção multi-tenant. Redija secrets (apikey).
4. Controle de nível é por pacote: `logging.getLogger("cogno_engram").setLevel(...)`.

O host anexa o handler (TenantFilter + JsonFormatter) ao root logger real;
veja `cogno/core/logging.py` no host como referência.

## Nota específica do cogno-engram

- **WARNING** em JSON inválido da LLM de consolidação (`_parse_memories`) e da
  extração de relações do Knowledge Graph (degradação tratada, não fatal).
- **INFO** nos marcos caros e raros: consolidação Tier-2/Tier-3 concluída,
  pruning/re-embedding/poda de órfãos (contagens).
- **DEBUG** em ingestão de entidades NER no grafo e nos scores do reranker
  (Pass-1 sim/recency/category + cross-encoder Pass-2). A transcrição enviada ao
  modelo é conteúdo de usuário → DEBUG apenas (dev-only).
