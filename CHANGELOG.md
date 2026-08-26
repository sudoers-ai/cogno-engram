# Changelog

## Unreleased

### Added

- **`audience` na ARESTA: o tenant vê tudo, um identity só a sua vida.** Decisão de produto de
  2026-08-25. A coluna vai na aresta e isso é forçado, não preferido: `knowledge_nodes` é único
  em `(scope, lower(label), node_type)`, logo o nó "Maria" é UMA linha para o tenant inteiro —
  dois contactos que mencionem uma Maria partilham-na, e não há "a Maria do José" para marcar.
  A ARESTA é que é dele. Visibilidade de nó é DERIVADA: um nó é visível quando alguma aresta
  que o leitor pode ver lhe toca; um nó órfão é só de staff.

  Valores: `''` **não classificada** (staff sim, contacto não), `tenant` (facto de negócio, todos),
  `identity:<id>` (a vida de um contacto). Produzidos só por `audience_for`/`sanitize_audience`.
  **O default é fail-CLOSED para o contacto**: um escritor que se esqueça custa um bloco em
  falta — visível, chato, seguro — e nunca uma fuga. Dois discriminadores neste código nasceram
  permissivos (`status` a `accepted`, `source_session` vazio) e ambos tiveram de ser desfeitos
  depois de já terem falado.

  **`audience` é keyword OBRIGATÓRIO** nas nove leituras que podem devolver dado de contacto.
  Com um opcional, esquecer devolve TUDO e a falha é silenciosa; obrigatório, esquecer é
  `TypeError` na chamada. Medido: ao pôr o keyword, **72 chamadas** na suíte deste repo
  falharam, as 72 por falta do argumento — nenhuma mudança silenciosa de comportamento.

- **`KnowledgeGraph.has_edges(scope, label)`** — o predicado de órfão, sem audiência e sem
  status, de propósito. O chamador é o `prune_orphan_nodes`, que APAGA: ali uma leitura
  filtrada não estreita o que se vê, alarga o que se destrói — um nó cujas arestas são todas de
  outro contacto pareceria solto e seria removido. A pergunta "aponta alguma coisa para este
  nó" não tem audiência. Achado em revisão, antes de entrar.

- **`maintenance.classify_edge_audience`** — a migração das arestas antigas. Carimbo vazio →
  `tenant` (só staff/admin/KB escreve sem sessão); carimbo cheio → a vida desse contacto, com o
  mapa sessão→identity injectado pelo host. Sessão irrecuperável fica `''`: staff continua a
  ver, nenhum contacto vê, e é um "não sei" honesto em vez de um dono errado. `dry_run=True`
  primeiro, idempotente, e testada verbatim.

- **`propose_relations` aceita um PREDICADO** (`(source, target, relation) -> bool`), não só um
  booleano. "Rever tudo ou não rever nada" é a granularidade errada para o que a opção protege:
  as arestas que viram uma frase sobre uma PESSOA ("sua esposa Maria") são uma classe pequena e
  nomeável; as outras ("a clínica aceita Unimed") são factos de domínio que um walk deve
  continuar a afirmar. O tudo-ou-nada obriga um host a escolher entre falar alegações não
  revistas sobre a família de alguém e perder o bloco de conhecimento inteiro — e o primeiro
  host a encontrar essa escolha tomou a primeira opção sem reparar, durante meses, em produção.
  Mesma forma e mesma costura do `edge_filter`. Um predicado que LEVANTA devolve `proposed`,
  nunca `accepted`: uma aresta que ninguém conseguiu classificar espera por um humano.

- **Edge curation** — `GraphEdge` gains `attributes` (free-form detail: `{"age": 8, "note": …}`)
  and `status` (`accepted` | `proposed` | `rejected`), plus `VALID_PROXIMITY_RELATIONS`, a closed
  vocabulary for the relations that describe a person's close world.

  Who **asserted** an edge decides whether it may be spoken. An edge becomes a sentence the agent
  states about a person as if it knew — "your son Pedro" is either a kindness or an invention —
  so a host asserts and an LLM extraction proposes.

- `KnowledgeGraph.pending_edges(scope)` / `set_edge_status(...)` — the curation queue and the
  verdict. **`walk()` returns accepted edges only and has no flag to say otherwise**: a walk
  feeds the prompt, and a keyword that could turn the filter off is a keyword someone eventually
  passes. A proposal is also skipped by the TRAVERSAL, not merely filtered from the result —
  otherwise it decides what the walk can reach and leaks the same unverified claim one hop away.
  `format_graph_context` repeats the filter at the last step before text, as defence in depth.

- `hypnos.periodic_consolidate(propose_relations=True)` — Tier 2 writes its extracted edges as
  `proposed`. **Opt-in**: flipping the default would silently empty the graph block of every host
  already running, with nothing in the logs saying why.

- **`KnowledgeGraph.count_nodes(scope, *, label=None)`** — how many nodes a scope holds, or how
  many carry a label, as a **query** instead of a page.
  `list_nodes` is `ORDER BY id LIMIT n`: no label filter, no offset. A caller asking *"is this
  label unique in this scope?"* over it gets the right answer only while the tenant stays smaller
  than the page — and a homonym created past the cut is invisible. That was a live defect: the
  host had to refuse to answer whenever the page came back full, because the alternative was
  speaking a stranger's facts about a contact.
  `lower(label)` on both sides, matching `find_node` and the `walk` seed — a case-sensitive count
  would answer a different question from the one the caller is about to act on.
  **Contract change:** `KnowledgeGraph` is `@runtime_checkable`, so a host with its own adapter
  must add the method to keep satisfying it.

### Changed

- **O comentário do índice `idx_turn_traces_scope_time` passa a dizer o que foi medido**, em três
  pontos onde afirmava de mais: contradizia-se ("não tinha índice nenhum que a servisse" vs "o
  ramo `scope = %s` já era servido pela UNIQUE" — o segundo é o correcto); citava
  `idx_turns_scope_time` como o irmão que "já tinha" o índice, quando esse irmão é btree COMUM e
  pelo mesmo argumento não serve o ramo do LIKE dele próprio (medido a 200k, `admin_turns` e
  `admin_scopes` dão ambos `Parallel Seq Scan on turns` — os outros dois consumidores do padrão
  ficam NOMEADOS lá); e o `created_at` no segundo lugar, onde uma corrida única dizia "empate" e
  sete dizem 10,97 ms contra 4,84 ms com as distribuições sem sobreposição. O `DESC`, esse, nunca
  é lido (o `Sort` explícito por cima do `BitmapOr`) e fica por consistência de forma.

- `PostgresStore.save_turn_trace` now honours `TurnTrace.created_at` when set (the
  in-memory adapter always did); absent, the column default stamps the row as before.
  A backfilled or imported trace no longer reads as "now", so a `since` window over it
  means what it says.

- Postgres: `knowledge_edges` gains the two columns, with an **additive `ALTER TABLE` migration**
  (`CREATE TABLE IF NOT EXISTS` is a no-op against a live table) and an index on `(scope, status)`.
  The backfill DEFAULT is `accepted` — nothing a host already asserted becomes unreviewed
  overnight. Re-asserting an edge **merges** attributes and may PROMOTE a proposal, but can never
  demote a verdict: a review that the next LLM pass could expire is a review nobody would do.

- `format_graph_context` renders `attributes` as a bounded parenthetical
  (`- José --[PARENT_OF]--> Pedro (age: 8; note: …)`), newline-flattened.

- `sanitize_edge_status` distinguishes **absent** (`None`/`""` → `accepted`, back-compat) from
  **present-but-unreadable** (a typo → `proposed`). Folding a typo into `accepted` would invert
  the caller's intent in the one direction the feature exists to prevent. Normalisation runs in
  `GraphEdge.__post_init__`, so the two stores cannot disagree.

- `neighbors()` and `get_node_context()` obey the same rule as `walk()` in both adapters: an
  unreviewed edge still DISCLOSES its endpoint, and `NodeContext` hands both fields to one caller.

- `pending_edges` returns **oldest first** in both adapters, so a bounded queue drains.

- `rejected` is sticky — `set_edge_status` is the only way back (`upsert_edge` cannot tell a
  deliberate correction from a re-extraction).

Callers that never set `status` are unaffected **in data**: the default is `accepted` and every
existing walk returns what it returned before. The **contract** does change — `KnowledgeGraph` is
`@runtime_checkable` and gained `pending_edges`/`set_edge_status`, so a host with its own adapter
stops satisfying it under mypy/`isinstance` until it implements both.

### Fixed

- **`José` e `Jose` passam a ser a mesma pessoa — e `find_node` deixa de perder o nó sob collation
  `C`.** Decisão de produto do dono: num CRM que recebe WhatsApp, o contacto escreve o nome sem
  acento metade das vezes, e um grafo que trate os dois como nós distintos parte a vida da pessoa
  em duas.

  O defeito por baixo era outro e mais estreito: a identidade de nó era `lower(label)` no Postgres
  e `label.lower()` no Python, e **as duas discordavam**. Num cluster `LC_COLLATE 'C'` o `lower()`
  do Postgres nem sequer dobra maiúsculas acentuadas — `lower('JOSÉ')` dá `'josÉ'` — logo
  `find_node(scope, "JOSÉ")` devolvia `None` para um nó gravado como `josé`, enquanto o adaptador
  in-memory acertava. Medido: 7 de 14 rótulos falhavam sob `C`, 0 sob `en_US.utf8`.

  Agora há **uma** definição, `cogno_engram/folding.py::fold_label`, e as duas metades correm-na:
  o Python directamente, o Postgres pela função `engram_fold` que o `ensure_schema` cria. Três
  camadas, todas necessárias e todas medidas: `casefold()` (trata `ß`→`ss`, que `lower` não),
  NFD + remoção de marcas combinantes (tira o acento), e uma tabela de transliteração de 15
  entradas **derivada do `unaccent` do Postgres** (`æ`→`ae`, `ø`→`o`, `ł`→`l` — caracteres sem
  decomposição combinante, que o passo 2 deixaria intactos).

  `tests/test_folding_parity.py` re-deriva a tabela contra um Postgres a sério e falha se deixar
  de bater: uma cópia de um dicionário que vive noutro processo apodrece em silêncio, e apodrecer
  aqui significa os dois adaptadores darem respostas diferentes à mesma pergunta. O alfabeto
  coberto está DECLARADO — Latin-1 Supplement + Latin Extended-A, onde vivem os nomes
  pt/es/en/de/fr/it. Fora dele os dois lados podem divergir (medido: sigma final grego).

  **A migração pode FALHAR, e isso é o comportamento certo.** Numa base que já tenha `José` e
  `Jose` como nós separados, o índice novo recusa-se a nascer — e o `ensure_schema` levanta com os
  rótulos em conflito NOMEADOS, em vez da chave dobrada que o Postgres reporta. Fundir
  automaticamente escolheria um dos rótulos e mudaria as arestas do outro de dono, em silêncio,
  num grafo cujo propósito é dizer factos sobre pessoas.

- **O teste de plano do índice da subárvore ficava VERMELHO em código correcto.** Ele afirmava o
  NOME do índice; num cluster com collation `C` (`initdb --locale=C`, `postgres:alpine`, qualquer
  base criada `LC_COLLATE 'C'`) a UNIQUE pré-existente já serve ambos os ramos do OR, não há Seq
  Scan nenhum — e o teste falhava com uma mensagem a dizer o contrário do que o plano mostrava.
  Passa a afirmar a ausência de `Seq Scan`, que é a propriedade do próprio título e é verdadeira
  sob `C`, `en_US.utf8` e ICU. Reproduzido: versão antiga sob `C` vermelha, nova verde.

  Segundo defeito no mesmo teste: ele copiava à mão uma aproximação do SQL em vez de exercitar o
  `admin_traces`. Uma garantia de desempenho sobre uma consulta que ninguém emite não é garantia
  — medido, trocar o predicado por um curinga à cabeça (que nenhum índice pode servir) deixava-o
  VERDE. Agora o SQL é capturado do método e é esse que vai ao EXPLAIN; essa mutação passou a
  matar. O que ele continua a não apanhar, dito na docstring, é deriva do ORDER BY — um
  `BitmapOr` nunca preserva ordem, o plano acaba sempre num `Sort`, e a propriedade sob teste é
  insensível a ela.

- **O índice da subárvore passa a ser verificado no modo PARTICIONADO**, que é o que a produção
  corre (`cogno_host/migrate.py::init_db` usa `partition_by_scope=True`). Aí o índice do pai é
  propagado aos filhos sob nome auto-gerado, o nome do pai nunca aparece num plano, e uma
  asserção de plano exigiria linhas suficientes para cada partição passar o limiar — minutos de
  teste. É por isso, deliberadamente, uma asserção de DDL: prova que o índice EXISTE em cada
  partição, não que o planeador o escolhe lá (tecto dito na docstring).

- **O irmão `idx_turns_scope_time` tinha o MESMO defeito que o índice dos traços corrigiu.** Ele
  era btree COMUM, e num collation que não seja `C` um btree comum não serve `LIKE 'prefixo/%'` —
  portanto `admin_turns` e `admin_scopes` varriam a tabela inteira, COM o índice presente. Este
  ficheiro chegou a citá-lo como o irmão que "já tinha" o índice; estava ao contrário.

  **Substitui em vez de acrescentar, e a escolha é medida** — 200k linhas, tenant a ~10% da
  tabela, medianas de 7–9 corridas por célula:

  | índice | tamanho | escrita 20k | subárvore |
  |---|---|---|---|
  | btree comum (o antigo) | 24 MB | 121 ms (117–142) | **Seq Scan**, 18–24 ms |
  | os DOIS | 34 MB | 152 ms (133–165) | Bitmap Heap, 10 ms |
  | só `text_pattern_ops` | 24 MB | 115 ms (103–163) | Bitmap Heap, 8–12 ms |

  Manter os dois custaria +10 MB e ~26% de escrita na tabela mais quente do schema, para nada: o
  `text_pattern_ops` serve TAMBÉM o ramo `=` e mantém o Index Scan ordenado da consulta de
  igualdade+ordenação (0,087 vs 0,094 ms). A ordem de saída do `admin_scopes` é idêntica — o
  `ORDER BY scope` usa o collation da coluna, não a opclass do índice.

  **O nome muda de propósito:** `CREATE INDEX IF NOT EXISTS` com o nome antigo e definição nova é
  um no-op SILENCIOSO, e o conserto subiria inerte em toda a instalação existente. Nome novo
  (`idx_turns_scope_pattern`) + `DROP` do antigo, nesta ordem — se o processo morrer entre os
  dois, fica-se com dois índices (lento a escrever, correcto a ler) e não com nenhum.

- **Os irmãos do `admin_traces` não tinham guarda nenhuma a fixar que CHAMAM o escaping.**
  `_subtree_like`/`_SUBTREE` são partilhados por `admin_turns`, `admin_scopes` e `admin_traces`,
  portanto qualquer mutação DENTRO do helper morria pelos casos do `admin_traces` — o que dava a
  impressão de que o padrão estava coberto. Medido: trocar `like = self._subtree_like(prefix)`
  por `like = prefix + "/%"` dentro do `admin_turns` **sobrevivia a 236 verdes** e devolvia
  `['tXa/u1', 'tZa/u9', 't_a', 't_a/u1']` — com o `user_input` de outros tenants. Idem
  `admin_scopes`. Ambas as mutações morrem agora.

- **Ao nível do BANCO só um dos três metacaracteres do LIKE era exercitado.** O caso adversarial
  de integração usava um único prefixo, contendo `_`; medido, uma mutação que escapasse `\` e `_`
  mas não `%` sobrevivia ao ficheiro de integração inteiro (40 verdes). Passa a ser parametrizado
  sobre `_`, `%` e `\` — e um censo contra o banco confirma que são exactamente esses três, de 31
  caracteres de pontuação/espaço. As três mutações por metacaractere morrem.

- **`admin_traces` ganha o índice que o irmão já tinha.** A leitura de subárvore filtra por escopo
  + `created_at` e ordenava sem índice nenhum — enquanto o `admin_turns`, igualmente uma leitura de
  manutenção, tem o seu (`idx_turns_scope_time`). A assimetria era o achado; o custo absoluto ainda
  não doía.

  **`text_pattern_ops` não é decoração**, e é onde a correcção "óbvia" falha: a base corre em
  `en_US.utf8`, e num collation que não seja C um btree COMUM **não serve** `LIKE 'prefixo/%'`.
  Medido em 200k linhas, com um tenant a 0,025% da tabela:

  | índice | plano | tempo |
  |---|---|---|
  | nenhum | Parallel Seq Scan | 12,8 ms |
  | btree comum | Parallel Seq Scan | 13,0 ms — nem é considerado |
  | `text_pattern_ops` | Bitmap Heap Scan | **0,21 ms** |

  O ramo `scope = %s` já era servido pela UNIQUE `(scope, session_id, turn_n)`; faltava o ramo do
  LIKE, e é por isso que o `BitmapOr` do plano usa os dois. O teste afirma o **plano**, não a DDL —
  uma asserção de DDL passaria com um índice que o planeador nunca escolhe, que é exactamente o
  estado que a correcção óbvia produz.

- **[HIGH] `delete_edges_by_session(scope, "")` apagaria a classe protegida.** Um id de sessão
  em branco não é um wildcard: casa com todas as arestas de `source_session` vazio, que é
  exactamente o que nada automatizado escreve — as notas que um HUMANO ou a API de admin lá
  puseram. Um turno com dislike a chegar com id vazio apagava-as todas, e um
  `DELETE ... WHERE source_session = ''` lê-se como inteiramente normal num log. Os dois
  adaptadores passam a recusar (`ValueError`) em vez de devolver 0 — um id vazio ali é bug do
  chamador, e engolir esconde o bug fingindo que a poda correu.

- **A read hands back a COPY, at all four doors.** `walk()`, `get_node_context().edges` and
  `upsert_edge` (which stored the CALLER's object) returned live references into the in-memory
  store, so a caller that touched what it was given changed what the prompt says — while
  Postgres, which builds fresh rows, did not. Measured side by side:
  `walk(...)[0].attributes["note"] = "LEAKED"` rendered into the in-memory block and not into
  the Postgres one. Same code, two prompts, on the invariant the curation feature is.

  The copy is also deep enough to matter: `dataclasses.replace` alone shares the `attributes`
  dict with the store, and that dict is what `format_graph_context` renders.

## 0.1.1 — 2026-08-02

Maintenance ops for an embedding-model switch.

- `reembed_knowledge_nodes`: only `memories` had a re-embed op, so a graph node
  left in the old vector space became silently unreachable by
  `find_nodes_by_embedding` — re-embedding one store and not the other left the
  system half-migrated in a way nothing reported.
- Both re-embed ops now prefer the embedder's `embed_batch` when it offers one,
  falling back to sequential `embed`. Re-embedding is the bulk operation by
  definition, and against a metered provider the difference is latency and
  rate-limit headroom.

## 0.1.0 — 2026-07-25

First public release on PyPI.

Persistence substrate for the Cogno cognitive pipeline — memory store, knowledge graph, conversation buffer, and sleep-time consolidation (hypnos)
