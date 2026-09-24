# Changelog

## Unreleased — ÂNCORAS: quem a pergunta NOMEIA, e a decisão `partial` (2026-09-24)

### Added

- **`cogno_engram.lexical`: âncoras e a camada `partial`.** Uma ÂNCORA é um rótulo de quem ou
  do que a pergunta fala — uma entidade que ela nomeia, ou quem pergunta quando fala na 1.ª
  pessoa; QUEM é cada uma decide-o o chamador. Uma aresta é «sobre» uma âncora quando o SEU
  EXTREMO (origem ou alvo) contém as palavras dela, inteiras e por ordem, com a dobra do
  tokenizador (`anchor`, `about`). Quando NADA passa o piso, `decide(..., anchors=[…])` responde
  **`partial`** (`DECISION_PARTIAL`) com essas arestas e as que estão UM salto adiante delas
  (`partial`, top `TOP_K`), em vez de `nothing_relevant`; `render(..., decision=)` diz que a
  resposta é parcial («Nothing recorded answers … directly. Recorded about who or what it names
  (it may not answer the question …)»). `FIRST_PERSON` + `speaks_of_self` lêem a 1.ª pessoa do
  SINGULAR (pt/en, alfabeto do tokenizador; o plural fica de fora, porque na boca do staff é a
  empresa).
- **Porquê — medido no replay de um consumidor, sobre chamadas reais:** em todas as chamadas em
  que o passeio por proximidade tinha uma aresta rotulada relevante e esta decisão disse «nada
  relevante», essa aresta JÁ ERA candidata e pontuou abaixo do piso. Não era recolha: uma pergunta
  na 1.ª pessoa não partilha palavra com `<quem pergunta> --[TEACHES]--> <turma>`, e uma pergunta
  longa sobre uma pessoa dilui a única palavra comum. Juntar mais arestas da pessoa não mexe
  nisso; dizer o que significa uma aresta QUE TOCA a entidade nomeada mexe.
- **O que NÃO muda:** tudo o que passa o piso (a camada nunca compete com um resultado relevante,
  não mexe em pontuação nem no piso); uma fonte partida continua a dar `error`; sem âncoras, a
  decisão é exactamente a de antes. `graph_candidates` documenta a convenção do passeio de uma
  âncora (variante ≠ 0, id de nó ordinal `a<k>`): nunca `old`, e uma aresta partilhada com o
  passeio da pergunta é UMA candidata, com o id que ganhou primeiro.
- **O custo, dito:** uma pergunta sobre um atributo que ninguém registou, que nomeia uma entidade
  COM arestas, passa a receber as arestas dessa entidade como resposta parcial onde antes recebia
  «nada relevante». É o preço de nunca dizer «nada» sobre uma pessoa que o grafo conhece; o render
  é o que o mantém honesto.
- **Testes** (`tests/test_lexical_anchors.py`, 14, conteúdo inventado): as três formas medidas
  (quem pergunta; um salto adiante do que a pergunta nomeia; uma pessoa diluída numa pergunta
  longa), cada uma nos DOIS mundos (sem âncoras: `nothing_relevant`; com: `partial`); o controlo
  de que a camada nunca compete com um resultado relevante (com uma âncora que TEM arestas); a
  fonte partida que ganha à camada, e o seu par; palavras inteiras e por ordem, nos dois extremos,
  só no grafo; o alfabeto da 1.ª pessoa. **Mutações, cada uma morta:** camada desligada (5
  vermelhos), sem o salto (2), só a origem como extremo (2), a camada antes do `error` (1), a
  camada a competir com um resultado relevante (1).

## Unreleased — o custo do ranking prova-se por RAZÃO intercalada e o tecto por MECANISMO; os «< 50 ms» passam a medição (2026-09-24)

### Changed

- **`tests/test_lexical_cost.py`** deixa de afirmar um tempo absoluto. O gémeo antigo prendia o
  ranking limitado abaixo de 50 ms, e isso media a MÁQUINA e o INSTRUMENTO, não o código: o
  `process_time` conta só o CPU do processo, mas um processo que partilha núcleos e caches com
  vizinhos ocupados gasta mais do seu próprio CPU no mesmo trabalho (no consumidor que trouxe este
  gémeo, o portão leu 93,6 e 160,9 ms com a caixa a load ~13), e o tracer de cobertura desta CI
  quadruplicava a leitura. Agora:
  - **o tecto, sem relógio** — `test_over_3MB_the_builder_stops_at_MAX_CANDIDATES_mechanically`:
    sobre ~3 MB, o construtor pedido com `limit=MAX_CANDIDATES` devolve exactamente isso, de um
    conjunto mais de dez vezes maior, e o conjunto limitado ainda responde;
  - **o que o tecto compra, por razão** —
    `test_TWIN_the_capped_ranking_is_an_order_of_magnitude_cheaper_timed_intercalated`: limitado e
    sem limites medidos INTERCALADOS (c, u, c, u, c, u), no mesmo processo e à mesma carga, mínimo
    de cada lado, `sem limites ≥ 10 × limitado`. O conjunto é ~16 vezes o tecto; a razão fica perto
    de 16 com a máquina leve e sob carga, porque a carga incha as duas pernas por igual.
  - Os números ficam no docstring do módulo como MEDIÇÃO (24/09/2026, 20 threads, load ~9: ~22 ms
    limitado, ~325–350 ms sem limites).
  O README e o docstring de `MAX_CANDIDATES` dizem o mesmo (só texto).
  Saem `test_TWIN_3MB_of_sections_capped_at_MAX_CANDIDATES_ranks_under_50ms` e
  `test_PAIR_the_same_3MB_uncapped_is_far_over_the_bound`; entram os dois acima. Só testes; nenhum
  código da lib muda.

## Unreleased — a ingestão em DOIS passos: preparar (e mostrar o custo) antes de confirmar (F2.4, 2026-09-24)

### Added

- **`cogno_engram.ingest.prepare()`** — extrai, parte em trechos e ESTIMA os tokens sobre os
  trechos EXACTOS que o commit vai embeber, e estaciona a versão em **`awaiting_confirmation`**
  com os trechos SEM vector (tabela nova `kb_drafts`). Não recebe embedder nem `gate` (nem os
  pode receber: o teste lê a assinatura), portanto não gasta nada — um erro do ficheiro sai
  AQUI, antes de se pedir confirmação. `estimated_tokens` e `expires_at` (o `now` injectado + 24 h,
  `documents.DRAFT_TTL`) ficam PERSISTIDOS na versão (`KbVersion`), para o GET do host.
- **`commit()`** — reclama o rascunho atomicamente (`awaiting_confirmation` → `processing`, o
  rascunho sai), passa ao `gate` a MESMA estimativa, embebe com o `pace`, troca. Devolve o uso
  como sempre. Resultados novos: `not_prepared`, `expired`; um segundo commit é `unchanged` com
  zero chamadas; três confirmações em simultâneo embebem o rascunho uma vez.
- **`expire_drafts(store, now=…) -> int`** — o varrimento do tick: cada rascunho com
  `expires_at <= now` passa a `error`/`expired`, perde o rascunho e o bytea do original, e deixa
  lápide `expired`. A versão fica (o dono vê porquê); uma versão servida ao lado continua servida.
- **Store:** `get_version`, `save_draft`, `claim_draft`, `pending_drafts`, `expire_drafts`;
  estado `awaiting_confirmation`, motivo `expired`, lápide `expired` nos alfabetos fechados; e
  nenhum adaptador escreve um estado à mão (teste). `kb_versions` ganha `estimated_tokens` e
  `expires_at` por `ADD COLUMN` verificado no catálogo, e um índice PARCIAL sobre os rascunhos.
- **`ingest()`** mantém a assinatura e é `prepare()` + `commit()` seguidos — um teste compara os
  dois caminhos campo a campo.
- **`discard_draft(owner_key, document_id, version, *, actor)`** — faz AGORA o que a expiração faz
  às 24 h, a pedido de quem subiu o ficheiro: a versão passa a `error`/`discarded`, o rascunho e o
  original saem, lápide `discarded` com o `actor`. Só a um rascunho: noutro estado responde
  `not_a_draft` (uma versão servida sai por `delete_document`), de outro dono `missing`; um 2.º
  descarte responde `discarded` sem segunda lápide. **Porquê (privacidade):** um upload errado ao
  lado de uma versão servida guardava o original 24 h sem o cliente o poder tirar — a condição (e)
  do Director diz «apagar tira na hora».
- **`interrupt_stale(*, older_than, limit)`** — o segundo varrimento do tick, cross-owner: toda a
  versão em `processing` cujo `claimed_at` é anterior a `older_than` passa a
  `error`/`interrupted`, com lápide (um `prepare` a meio de um crash; um `commit` que morreu depois
  de reclamar — o claim já consumiu o rascunho, não há retoma). **`claimed_at`** é novo em
  `kb_versions` (aditivo): o momento em que a versão ENTROU em `processing` pela última vez (o
  `begin_version`, com o relógio do `prepare`, e o `claim_draft`); o `created_at` não distingue um
  rascunho confirmado há um minuto de um abandonado há uma hora. Motivos `discarded`/`interrupted`
  e as lápides correspondentes nos alfabetos fechados — antes, `fail_version(reason="interrupted")`
  saía `internal`.
- **As duas guardas de «commit sem prepare», cada uma com a SUA pergunta e o SEU teste.** O
  `claim_draft` pergunta só «há rascunho?» (nos dois adaptadores:
  `test_the_claim_refuses_a_version_whose_draft_is_gone`, versão ainda à espera mas sem
  rascunho); o `commit` pergunta «a versão está à espera de confirmação?»
  (`test_a_commit_refuses_a_version_out_of_awaiting_even_with_a_draft_present`, montado no
  duplo: rascunho presente, versão fora de `awaiting_confirmation`). **Vermelho-antes medido:**
  o `claim` também verificava o estado, e a mutação da guarda do `commit` SOBREVIVIA ao seu
  teste; separadas as perguntas, cada mutação morre sozinha pela asserção.
- Mutações do descarte e do varrimento de interrompidos (9, cada uma sozinha, âncora contada):
  todas morrem pela asserção — incluindo a que deixa o original no descarte (`97 == 0`) e a que
  julga pelo `created_at` em vez do `claimed_at` (`1 == 0`).

## Unreleased — documentos: um quarto port, versionado, pesquisado a pedido (F2.4, 2026-09-24)

### Added

- **`DocumentStore`** (`cogno_engram.ports`) com as regras num módulo só
  (`cogno_engram.documents`), o duplo `InMemoryDocumentStore` e o adaptador
  `PostgresDocumentStore`. Texto que alguém ESCREVEU para ser lido — Markdown ou PDF —, sem
  tabela nem leitura em comum com as memórias ou o grafo.
  - **Dono opaco** (`owner_key`), com a regra de subárvore da casa: `purge_owner_subtree("t1")`
    apaga `t1` e `t1/…`, nunca `t10`. A engram não sabe o que é tenant nem persona.
  - **Leitores opacos** (`profiles`) e `profile` OBRIGATÓRIO em toda leitura do caminho do
    leitor (`search`, `readable_documents`), sem wildcard; branco → `ValueError`. Só a versão
    SERVIDA em estado `ready` é lida.
  - **Um modelo por versão** (`embed_model`, rótulo `<spec>@<largura>` com a largura verificada
    contra a coluna). A busca recebe UM vector e o rótulo; se algum trecho legível for de outro
    modelo, ou não houver vector, a busca INTEIRA é léxica e marcada
    `kb_embed_space_unavailable` — nunca um cosseno entre modelos.
  - **Pontuações cruas em [0, 1], sem piso**: `vector_score` = 1 − distância cosseno cortada a
    [0, 1] (ou `None` quando não medido), `lexical_score` (Postgres: `ts_rank_cd` com
    normalização 32), `score` renormalizado — igual ao `lexical_score` numa busca léxica.
    Dentro de um resultado, ou todos os hits têm `vector_score` ou nenhum. Empates por
    (documento, versão, ordem).
  - **Troca de versão atómica** e idempotente por (documento, sha256, modelo); uma falha deixa a
    versão servida a responder; um trecho nunca se grava sem o seu vector.
  - **Apagar tira da busca na hora**, leva os originais de TODAS as versões e deixa uma LÁPIDE
    (ids, versões, quando, quem — sem título nem texto). Um job que acaba depois não escreve nada.
- **`cogno_engram.chunking`** — em CARACTERES (~2000, 15% de sobreposição), por cabeçalho
  (Markdown, blocos de código respeitados) ou por página (PDF, com o trilho dos marcadores), cada
  trecho com o caminho de títulos à cabeça.
- **`cogno_engram.ingest.ingest()`** — tecto de bytes ANTES do extractor → tentativa registada →
  extrair → partir → `gate` (antes de embeber: recusa = zero chamadas) → embeber (`pace`, com
  `TokensPerMinute` pronto) → trechos → troca. DEVOLVE o uso que o embedder reportou
  (`embedding_tokens`, `embedding_calls`, `usage_reported`); quem cobra é o host. `reindex()` a
  partir do original guardado; `stale_documents()` lista o trabalho de uma troca global de modelo.
- **`TextExtractor`** — Protocol ESTRUTURAL (a engram não importa a vox, nem a vox a engram):
  bytes + tectos por keyword, páginas de texto de volta, ou uma excepção com `reason` de
  `no_text`/`over_limit`/`encrypted`/`invalid`/`timeout`. O de PDF vive na `cogno-vox`.
- **`documents_probe()`** — todas as leituras contra um dono vazio, para o `/health` do host.
  `test_the_probe_passes_on_a_fresh_schema_and_fails_without_any_column` deita abaixo cada
  coluna de cada tabela `kb_*` (lista lida do catálogo) e exige que a sonda falhe. **Vermelho
  medido antes do conserto:** com o `_records` a saltar a consulta das versões quando não havia
  linhas, sete colunas de `kb_versions` caíam sem a sonda dar por isso.
- **`ensure_schema`** cria `kb_documents`, `kb_versions`, `kb_chunks`, `kb_originals` e
  `kb_tombstones` (aditivo, `IF NOT EXISTS`; nenhum `ALTER` ao que já existia). Sem índice
  vectorial em `kb_chunks` de propósito — a busca é limitada aos trechos servidos de um dono;
  GIN no `tsv`. O original fica numa tabela PRÓPRIA que nenhuma leitura do caminho do leitor lê:
  `test_the_reader_path_never_reads_an_original` corre o caminho do leitor com um papel sem
  `SELECT` na coluna dos bytes, e o controlo (`get_original`) é recusado.

- **`test_the_lexical_order_is_the_same_in_both_adapters`** (integração) — a paridade de ORDEM
  do léxico entre os dois adaptadores, sob `ts_config='simple'`, num corpus DECLARADO no teste
  (ASCII, cada palavra da pergunta no máximo uma vez por trecho): a mesma ordem de hits e o 0 no
  mesmo sítio (trecho sem a palavra → 0 nos dois, e ausente numa busca léxica). **Não é o mesmo
  número** — o teste afirma que os valores DIFEREM, e o docstring de `in_memory._doc_lexical`
  diz por extenso onde os dois concordam e onde não. Vermelho produzido nos dois lados:
  achatar a ordem no in-memory e invertê-la no Postgres fazem-no falhar.

- **O stand-in léxico do in-memory usa a dobra GERAL** (`textfold.fold`, sem passos, `\w+`, SEM
  stopwords) — neutro, como o `simple` do Postgres mais a dobra de acentos. Não a `fold_label`
  (a regra de IDENTIDADE de rótulos do grafo) nem o `lexical.tokens` (a régua do motor de
  relevância, com stopwords e prefixo). `tests/test_documents_use_the_general_fold.py` conta a
  palavra `fold_label` em `documents`, `ingest`, `chunking` e na secção de documentos do
  `in_memory` e exige ZERO, com controlo (a mesma contagem na metade do grafo NÃO é 0) e com a
  metade positiva (`ø` separa as duas dobras). Nasceu `xfail(strict=True)` enquanto a #64 não
  tinha aterrado; aterrou primeiro, a troca fez-se aqui e a marca saiu.

- **Acentos na busca de documentos: `ensure_schema(..., ts_config, unaccent=True)`** cria, SE
  NÃO EXISTIR, a configuração derivada `cogno_<base>_unaccent` — `COPY` da base, e os tokens
  não-ASCII (`word`/`hword`/`hword_part`) passam pelo `unaccent` antes dos dicionários da PRÓPRIA
  base (o stem continua) — e gera o `kb_chunks.tsv` com ela; `PostgresDocumentStore(ts_config,
  unaccent=True)` interpreta a pergunta com A MESMA (`documents_ts_config`, uma função para os
  dois lados). Só `kb_chunks`: o `tsv` de `memories` não se mexe. Testes de integração: «sabado»
  encontra «Sábado» e vice-versa, «Sábados» encontra pelo stem, um termo sem relação não encontra
  nada; **vermelho produzido** — com `unaccent=False` o mesmo corpus e a mesma pergunta falham.
  Mudar a configuração numa base EXISTENTE não recalcula a coluna gerada: o `ensure_schema`
  regista `event=kb_ts_config_mismatch` e `rebuild_documents_tsv(...)` é a migração (reescreve a
  tabela com lock exclusivo — passo de operador). O defeito que fecha: a componente léxica pesa
  0,4 na busca NORMAL, e «José e Jose são a mesma pessoa» vale para as palavras também.

### Notes

- **Os backups da base de dados guardam um original até à retenção deles** — a purga não os
  alcança. A retenção é uma decisão pendente do dono; fica só registada.
- Os valores léxicos dos dois adaptadores para o mesmo trecho NÃO são iguais (o in-memory mede a
  fracção de palavras); o contrato é o INTERVALO. O `vector_score` é o mesmo número nos dois
  (teste de paridade).

## Unreleased — a DOBRA de texto e a RELEVÂNCIA léxica passam a viver aqui: uma dobra, um tokenizador, um score, um piso (2026-09-24)

### Added

- **`cogno_engram.textfold`** — `fold(text, *, punctuation=False, apostrophes=False,
  collapse_whitespace=False, strip=False)`: a ÚNICA dobra de acento e caixa para todo léxico que
  um consumidor compara, com as diferenças entre consumidores como argumentos que o chamador tem
  de DIZER. Base: NFKD → marcas combinantes fora → `casefold`, por esta ordem.

  **Veio do host de referência, sem mudar um byte de código** (comparado por AST, docstrings e
  comentários à parte: as quatro instruções de topo do módulo são idênticas às do host). O host
  passa a RE-EXPORTÁ-LA (`from cogno_engram.textfold import fold`) e apaga o algoritmo: uma
  definição, dois repositórios, e o teste de identidade (`is`) do lado do host. Os porquês vão
  na FORMA: NFKD e não NFD (compatibilidade: ligaduras, numerais romanos, algarismos em círculo,
  os alfabetos matemáticos em que se escrevem nomes de exibição); a dobra de caixa POR ÚLTIMO
  (com ela primeiro, os caracteres cuja decomposição de compatibilidade é maiúscula saem
  maiúsculos e uma segunda dobra muda-os outra vez — com ela por último a dobra é idempotente em
  todo o Unicode); `casefold` e não `lower` («Straße» é «Strasse»; o sigma final em contexto).

  **`folding.fold_label` fica DISTINTA, e declarada** — é a dobra de CHAVE dos nós, tem de
  concordar com o `unaccent` do Postgres (translitera, NFD) e não segue esta. Nenhuma linha dela
  mudou neste PR. As duas diferem em 3 757 code points, e um teste diz isso nos dois sentidos.

- **`cogno_engram.lexical`** — o motor que diz se um candidato é SOBRE a pergunta, e que, quando
  nada é, responde *nothing relevant* em vez de entregar o vizinho mais próximo. Uma recuperação
  por PROXIMIDADE tem sempre um mais próximo, portanto devolve sempre alguma coisa; o que a
  transforma num «não há nada» honesto é um PISO, e um piso precisa de um score que signifique o
  mesmo onde quer que seja tirado.

  **Veio do host de referência, e veio INTEIRO — não é um motor novo.** Vivia lá (a pesquisa
  híbrida do `knowledge_search`, em sombra, e o tokenizador do material que o `consult_material`
  lê). A regra do ecossistema é que o host fica com o negócio e a engenharia sai para as libs: o
  que aqui chega é o algoritmo; quem pode ler o quê, o tecto de TEMPO do turno, o corte do
  material e a calibração ficam no host. **A equivalência foi MEDIDA contra as funções que
  substitui, não afirmada**: 0 de 1 112 064 code points divergem na dobra, 0 de 100 000 cadeias no
  tokenizador e nos `terms` (prefixos 0/5/6/7), 0 de 400 mundos aleatórios em candidatos,
  ranking, decisão, `render` e `variants`. O comparador foi provado a ver: contra `fold_label` dá
  3 757 code points divergentes.

  - **um tokenizador** — `tokens` + `STOPWORDS` (a `textfold.fold`, o plural português, palavras
    funcionais fora), a ÚNICA definição de palavra que um ranking e todo piso sobre ele partilham
    (é um objecto: um teste pode afirmar identidade em vez de concordância);
  - **um score** — `terms`, `relevance` (a fracção das palavras da PERGUNTA que o candidato
    carrega, a melhor das variantes — a reescrita canónica e as palavras do próprio contacto),
    `rank` com herança de UM salto ao longo de um walk do grafo (`HOP_DECAY`) e desempate
    determinístico, `chosen`, `variants`;
  - **uma decisão** — `decide` sobre um alfabeto fechado (`DECISION_RELEVANT`/`_NOTHING`/`_ERROR`):
    uma fonte que PARTIU nunca se lê como uma fonte que não tem nada;
  - **candidatos a partir dos tipos DESTA lib** — `graph_candidates` (os `(variant, rank, node_id,
    edges)` de um walk), `memory_candidates` (registos de memória), `edge_text`, `edge_end`, cada
    um com um id SEM CONTEÚDO (`edge:<nó>.<n>`, `mem:<id>`) que uma resposta cita e um traço pode
    guardar; `render` é o payload com esses ids;
  - **as constantes** — `RELEVANCE_FLOOR` (0,3), `STEM_PREFIX` (0), `HOP_DECAY` (0,5), `TOP_K` (5)
    e **`MAX_CANDIDATES`** (2 000), o tecto de TRABALHO por contagem.

  **O piso é um ponto desta escala, e a curva que o escolheu é do CONSUMIDOR.** Não há aqui teste
  de calibração, e isso é de propósito: o 0,3 foi escolhido sobre o conjunto rotulado de quem o
  consome (~40 perguntas inventadas, três fontes, quatro leitores), e é esse teste que fixa esta
  constante — uma mudança aqui que mova o óptimo fica vermelha lá, no bump do pino.
  `test_the_default_floor_keeps_a_third_and_drops_a_quarter` diz o que 0,3 FAZ nesta escala, não
  porque é 0,3.

  **O tecto é por CONTAGEM, e o tamanho de cada candidato é do chamador.** Um ranking é CPU e,
  num event loop, CPU não se interrompe: ~3 MB de candidatos do tamanho de uma secção prenderam o
  loop ~325–400 ms no `rank` (medido aqui e, antes, no host). Com `MAX_CANDIDATES` ficam em
  ~22 ms nesta máquina. 2 000 candidatos de 1,5 KB são os mesmos 3 MB outra vez: quem constrói
  candidatos a partir de um documento grande corta-o antes; os construtores param EM `limit`
  (pede-se um a mais e sabe-se que o tecto bateu sem construir a cauda).

### Changed (em relação às cópias que substituem)

- **`graph_candidates(..., baseline_nodes=0)`** — a marca `old` («esta aresta também é do caminho
  ANTIGO») dependia de uma constante do host (quantos nós o caminho antigo anda). É um facto sobre
  o OUTRO caminho, portanto passa a ser um argumento; `0` não marca nenhuma. O host passa o seu.
- **`edge_end`** — era `_end`, privada, e o bench do host importava-a; um nome privado importado de
  fora de uma lib é uma API que ninguém declarou.
- **O docstring de `edge_text` deixa de dizer «a MESMA forma que o `format_graph_context`»** sem
  qualificação: é a mesma FORMA, não os mesmos bytes — o detalhe aqui corta a 160 sem reticências,
  lá a 120 com `…`. Mantido como estava porque este é o texto que é PONTUADO, e um corte que mude
  move scores.
- **`docs/HOST_INTEGRATION.md`** ganha as duas costuras: como um host compõe as suas leituras com o
  `lexical` (e o que continua a ser dele — leituras, tamanho, relógio, calibração) e como re-exporta
  a `textfold` em vez de manter uma cópia.
- **Os docstrings de `textfold` perdem os nomes dos módulos do host** que cada passo servia e o
  caso que ilustrava a ordem; ficam as razões.

### Testes

- `tests/test_textfold.py` (11) — os sete primeiros MUDARAM-SE com a função (os três passos
  explícitos, o que a base deixa em paz, a idempotência em todo o Unicode e em 100 000 cadeias, os
  nomes em letras matemáticas — gerados, não colados —, o conjunto onde a ORDEM mudou e o conjunto
  onde o `casefold` mudou, sempre como CONJUNTOS e nunca como contagem: a contagem é da base
  Unicode do Python, e cresce entre 3.10 e 3.12); os restantes são a paridade da FUNÇÃO nesses dois
  conjuntos (responde à base nova, e as bases antigas divergem lá — o controlo), o `None`, e a
  `fold_label` declarada distinta.
- `tests/test_lexical.py` (23) — os sete primeiros MUDARAM-SE com o motor (termos, relevância nas
  duas línguas, o zero que nunca passa, a variante que só entra se acrescenta palavras, o salto
  só para a frente, o desempate, as stopwords no alfabeto do tokenizador), com os fixtures
  RE-INVENTADOS; os restantes são desta lib (o plural e o que ele não faz, o piso nesta escala,
  ids sem conteúdo, dedupe, `baseline_nodes`, `edge_text`, memórias sem id ou vazias, erro vs
  nada, top-k, desempate completo, `render`).
- `tests/test_lexical_cost.py` (3) — o gémeo de tempo e o seu par, e o tecto que pára a
  CONSTRUÇÃO e não só o resultado. O relógio é o CPU do processo, com o GC em pausa e **sem
  tracer**: esta CI corre a suíte sob `--cov`, e o tracer de linha quadruplicou a leitura (medido
  na perna 3.10: 86,6 ms sob cobertura, ~22 ms sem ela). É o custo do instrumento, não do código;
  o par é medido na mesma condição, para que o tecto não possa passar por o tracer estar ligado
  numa metade e desligado na outra.

## Unreleased — o vocabulário de status deixa de ser escrito à mão dentro do SQL (2026-08-27)

### Changed

- **`adapters/postgres.py` passa a ler `EDGE_ACCEPTED`/`EDGE_PROPOSED` de `types.py` nos sete
  sítios em que os escrevia à mão dentro do SQL** — DDL da tabela, DDL da migração aditiva, os
  três ramos do `upsert_edge`, o filtro do `walk` e o do `neighbors`.

  **O ficheiro já importava as constantes e já as interpolava** — `_EDGE_VISIBLE` faz isso com
  o vocabulário de audiência, e a linha 1269 fá-lo com o próprio `EDGE_ACCEPTED`, a quarenta
  linhas de uma que escrevia `'accepted'` à mão. As duas metades do mesmo ficheiro discordavam
  sobre onde mora o vocabulário; agora não.

  **Porque isto não é cosmética: uma deriva aqui não levanta erro, deixa de casar.**
  `AND e.status = 'acepted'` é SQL válido que devolve zero linhas — o `walk` não alcança nada e
  o `neighbors` não revela ninguém, e o grafo parece **vazio** em vez de partido. Pelo outro
  lado, um DEFAULT fora do vocabulário é mapeado para `proposed` pelo `sanitize_edge_status` na
  leitura, e toda aresta nova sairia da travessia sem que ninguém lhe tivesse tocado.

  **O SQL renderizado é byte a byte o mesmo** (provado linha a linha contra `origin/main`: nove
  linhas tocadas, zero acrescentadas ou removidas, cada uma idêntica depois de desfazer a
  substituição). As duas formas de errar o escape de chaves na DDL são ruidosas — um `{}` por
  escapar é `SyntaxError` no import, um `{{}}` a mais é jsonb inválido que o Postgres recusa.

### Added

- **`tests/test_edge_status_vocabulary.py`** — nenhum adaptador escreve um valor de status à
  mão (prosa pode citá-lo, código não), o SQL só interpola constantes do vocabulário, e a
  **LISTA** fica presa: um quarto status torna o teste vermelho e obriga a rever cada consulta
  que filtra por status, em vez de derivar em silêncio. Um teste irmão prende o pressuposto do
  próprio verificador — a divisão no `#` só é segura enquanto nenhum fragmento de SQL contiver
  um `#`, senão o infractor estaria dentro da metade descartada.

- **Dois testes de integração que a mutação provou em falta**:
  `test_pg_a_proposal_re_proposed_still_MERGES_what_it_learned` (duas extracções da mesma
  aresta não revista são pares: a segunda **funde** o que aprendeu, e nada afirmava isso — o
  portão podia passar a reter em qualquer status guardado com a suíte inteira verde) e
  `test_pg_an_edge_written_without_a_status_column_is_ACCEPTED` (o DEFAULT da DDL, que as
  escritas deste pacote nunca exercitam porque `upsert_edge` manda sempre um status, mas que
  decide o que um backfill do host ou um restauro produzem).

## Unreleased — a retenção passa a ver os scopes que nenhum tenant possui (2026-08-27)

### Added

- **`MemoryStore.memory_scopes()` / `maintenance.memory_scopes(store)`** — todos os scopes com
  memória, **incluindo os que já não têm dono**.

  **Não é uma excepção ao guarda de scope: é uma pergunta que o guarda nunca cobriu.**
  O `_require_scope` existe para impedir uma **leitura ATRAVÉS de scopes** — que uma consulta
  devolva conteúdo de vários contactos porque alguém passou vazio. **Enumerar CHAVES é outra
  pergunta**, e a fronteira que as mantém separadas é absoluta:

  > **identificadores à saída, NUNCA conteúdo.** Nem uma memória, nem um rótulo, nem uma
  > contagem por categoria. Uma lista de chaves e nada mais.

  `test_devolve_CHAVES_e_nunca_conteudo` prende-a, com controlo positivo (o segredo **está** na
  loja, logo a asserção mede a fronteira e não uma loja vazia) e com mutação: fazer o método
  devolver o conteúdo mata seis testes.

  **A razão de existir, inteira:** a retenção **não pode ser guiada por tenant**, porque o scope
  cujo tenant desapareceu é precisamente **o que mais precisa de ser podado** — ninguém o possui,
  ninguém vai pedi-lo, e mais nada o visita. Medido na caixa viva a 27/08: **14 scopes têm
  memórias, 13 têm tenant vivo**, e o que falta é `default/guest` com **29 memórias de
  VISITANTES** (pessoas que nunca se registaram), **com categorias que incluem `pii`**.
  **A regra salta exactamente quem tem menos base para ser retido.**

  Vive em `maintenance`, com nome próprio, e **não pede scope** — se algum dia passar a pedir,
  deixa de poder ver o órfão, que é a única razão de existir. Um teste prende também isso.

## Unreleased — a poda pode CONTAR antes de apagar (2026-08-27)

### Added

- **`prune_memories(..., dry_run=True)` e `MemoryStore.delete_memories(..., dry_run=True)`**:
  devolvem quantas memórias **iriam** sair, sem tocar em nenhuma.

  **Porquê agora:** o `prune_memories` existe e está testado **desde sempre, e ninguém o chama** —
  varrido no host com controlo positivo (a mesma varredura encontra
  `from cogno_engram import maintenance` no `reembed.py`, portanto sabe achar). **A memória só
  cresce; nada sai por idade.** Ligá-lo é barato — a função difícil estava feita. O que faltava
  era poder **aprovar** a regra: retenção é irreversível, e ninguém deve descobrir o que uma
  regra de 120 dias remove **vendo-a remover**.

  **UM predicado, dois verbos.** O filtro é construído uma vez e só a cláusula da frente muda
  (`SELECT count(*)` em vez de `DELETE`). Contar com uma consulta à parte seria **re-derivar a
  regra que decide um apagamento**, e as duas divergiriam no dia em que um filtro fosse
  acrescentado — a forma de defeito que este repositório passa a vida a encontrar. Aqui *"o que
  iria"* e *"o que foi"* não podem discordar, e `test_the_dry_run_number_is_EXACTLY_what_the_real_run_removes`
  prende-o.

  **`dry_run` é opt-in nos DOIS níveis**, e o segundo foi apanhado por uma mutação sobrevivente:
  virar o default do ADAPTADOR passava os dez primeiros testes, porque o ajudante passa sempre o
  valor explicitamente. Mas `delete_memories` é porta pública — um chamador directo veria a
  faxina **parar em silêncio**, a devolver números certos e a não limpar nada.
  `test_the_PORTS_default_is_also_to_delete` fecha-o.

  **E o tecto de confiança ganhou a frase que faltava:** sem `max_confidence`, isto apaga um
  facto CONFIRMADO por ser velho — perda de dados vestida com a palavra "limpeza". O teste tem
  o seu próprio controlo (`test_WITHOUT_the_ceiling_the_confirmed_fact_would_go`), que prova que
  mede o tecto e não a idade.

## Unreleased — o tipo do nó é normalizado na FRONTEIRA (2026-08-27)

### Fixed

- **`GraphNode.__post_init__` dobra a CAIXA do `node_type`.** O índice único do Postgres é
  `(scope, engram_fold(label), node_type)`: a metade do RÓTULO é dobrada, a do TIPO não era.
  `Rex/PERSON` e `Rex/person` seriam **duas linhas para uma coisa** — a mesma forma do
  `José`/`Jose`, com metade do trabalho já feito. **Uma identidade meio-dobrada é pior que uma
  crua, porque parece resolvida.**

  **Na FRONTEIRA e não num terceiro ajudante:** o `graph_context.ingest_entities` e o `hypnos` já
  normalizavam, e os dois estão certos — mas são caminhos de conveniência, não a porta. A forma
  DOCUMENTADA de entrar é construir um `GraphNode` e chamar `upsert_node`, que recebia o valor
  cru; e esta é uma lib **pública**, cujos consumidores não são só o nosso host.

  **PREVENÇÃO, não reparação, e a distinção é medida** — na caixa viva, às **03:26 de 27/08**:
  394 nós, **zero** fora de maiúsculas, **zero** pares que difiram só na caixa do tipo, com
  **controlo positivo** (a mesma forma de consulta encontra 10 grupos de mesmo-rótulo/tipos
  diferentes, logo sabe encontrar). Não há nada para migrar, e é isso que a torna barata hoje:
  **com uma única linha em minúsculas a resposta inverteria** — normalizar só na ESCRITA e migrar
  primeiro —, porque o `__post_init__` corre também quando os adaptadores constroem um nó A PARTIR
  DE UMA LINHA, e um objecto que discorda da sua linha faz um ler-modificar-gravar criar uma
  SEGUNDA linha em vez de actualizar a primeira.

  **Só a caixa.** Um tipo desconhecido é dobrado mas **não coagido** a `CONCEPT`: coagir aqui
  reescreveria um valor à SAÍDA da base, que é uma decisão diferente e com perda. Os ajudantes de
  escrita já coagem contra `VALID_NODE_TYPES` — é o trabalho deles.

  **Não resolve os 10 grupos** de `Anselmo/CONCEPT` vs `Anselmo/PERSON`: isso é desacordo semântico
  sobre o que a coisa É, outro eixo, e continua parqueado.

## Unreleased

### Added

- **`graph_stats`: o desempate segue a ordem do store (`id`), não o alfabeto.** O chamador
  anterior ordenava sobre o que o `list_nodes` devolvia — `ORDER BY id`. O primeiro corte
  desempatava por rótulo, igualmente determinístico e silenciosamente diferente: dois nós de
  grau 1 trocavam de lugar, e foi um teste do HOST que não foi escrito para esta mudança que o
  apanhou. **Um desempate é comportamento, e esta mudança é de custo.**

- **`KnowledgeGraph.graph_stats(scope, *, audience, top=5)` — o resumo do grafo em DUAS leituras
  agregadas, em vez de `1 + 3N`.** O chamador (a rota `knowledge_stats` do host) precisava de
  quatro números — total de nós, total de arestas, histograma por tipo, e os mais ligados — e
  **nada no porto sabia responder "qual é o grau de cada nó" em bloco**. Então listava todos os
  nós e pedia `get_node_context` para cada um; esse ajudante é ele próprio
  `find_node` + `walk` + `neighbors`, logo o custo real era `1 + 3N`.

  Medido contra Postgres real, ligações por chamada:

  | nós | antes | depois |
  |----:|------:|-------:|
  |  10 |    31 |      1 |
  |  50 |   151 |      1 |
  | 100 |   301 |      1 |

  Na caixa viva são **388 nós → 1165 ligações por cada abertura da página**, e crescia com o
  grafo. O `test_the_cost_stops_growing_with_the_graph` mede **duas** dimensões e não uma: um
  custo constante que por acaso igualasse o de um grafo não provava nada — é a INCLINAÇÃO que
  interessa.

  **É mudança de CUSTO, não de SIGNIFICADO**, e essa é a parte difícil de provar: um PR de
  desempenho que mexe num número em silêncio é pior que a versão lenta. Por isso o teste
  principal **não afirma os números** — recalcula-os pelo caminho antigo, nó a nó, e exige que
  os dois concordem. As regras que ficam intactas: nós contados pela regra de audiência
  (DERIVADA para leitor não-staff, logo um órfão é só-staff), arestas DISTINTAS por
  `(source, target, relation)` e **só ACEITES** — porque o grau antigo vinha do `walk`, e o
  `walk` não atravessa outro estado.

  A sonda de fuga de audiência (`test_audience_leak.py`) passou a cobri-lo, e **foi ela que
  apanhou a omissão**: um agregado não devolve linhas próprias e por isso não parece divulgação
  — mas `top_connected` carrega nós inteiros, e um total que conta os nós de outro contacto
  divulga que ele existe.

- **`GraphEdge.created_at` — a aresta passa a lembrar-se de QUANDO.** A coluna existe em
  `knowledge_edges` desde que a tabela existe (`created_at timestamptz NOT NULL DEFAULT now()`),
  é escrita em todas as arestas, e a dataclass **deitava-a fora**: a porta perdia-a entre a base
  e quem lê.

  Não é decoração. Uma vista de grafo por contacto serve para **verificar e corrigir**, e **um
  facto errado sem data não é corrigível** — quem olha não sabe se é de ontem ou de Março, logo
  não sabe se ainda vale.

  `None` significa "não veio de um store": quem constrói uma aresta para ESCREVER não pode saber
  a data, e inventá-la aqui tornaria "quando aprendemos isto?" respondível com o instante em que
  alguém construiu um objecto.

  Nos **dois** adaptadores, e a paridade é afirmada num teste: o Postgres lê a coluna (pelo
  construtor único `_edge_from_row`, cujo docstring já dizia que é ali que uma coluna nova deixa
  de ser carregada em silêncio); o in-memory carimba na PRIMEIRA inserção, como o `DEFAULT now()`
  faz — e preserva uma data que o chamador traga, senão um replay deixa de poder reproduzir o
  passado.

### Changed

- **A base descartável passou a ser o DESTINO por omissão das suítes que fazem `DROP TABLE`.**
  Dono, 2026-08-26: *"Já temos um test só para os testes de integração, isso deveria ser
  padrão."* A guarda de 2026-08-04 transformou o engano numa recusa, mas continuava a deixar a
  pessoa **escrever** um DSN — e a forma que causou o estrago é justamente a que a shell já
  tem à mão (`COGNO_PG_DSN` exportado, um `Ctrl-C`/`Ctrl-V` de distância de `ENGRAM_TEST_DSN`).

  **Antes:** `DSN = os.getenv("ENGRAM_TEST_DSN", "")` em cada módulo; variável por pôr →
  `pytest.skip`. **Agora:** `resolve_test_dsn()` — `ENGRAM_TEST_DSN` explícito ganha; sem ela,
  `engram_test` no servidor LOCAL que `COGNO_PG_DSN` já nomeia (ou nos defaults do libpq, que
  são o que o serviço do CI serve); nada à escuta → `""` → salta exactamente como saltava.

  O que torna isto seguro não é uma verificação, é uma **construção**: o nome da base nunca é
  trazido de lado nenhum, é **escrito** (`_for_test_database`). Dar a esta função o DSN exacto
  que causou a perda devolve o descartável — e `test_db_guard.py` pina isso nos dois sentidos,
  incluindo que TODO default nomeia uma base de teste, seja qual for o ambiente. Um
  `COGNO_PG_DSN` REMOTO não é adoptado: `engram_test` na instância gerida de alguém não é nossa
  para criar, quanto mais para largar.

- **A guarda passou a inspeccionar o DSN RESOLVIDO, não a variável crua.** Tem de olhar para a
  mesma string que os fixtures vão abrir, ou as duas divergem e só uma é verificada. É também a
  segunda rede sob o parágrafo acima: um erro em `_for_test_database` não destrói nada, porque
  o `pytest_collection_modifyitems` volta a recusar o nome.

- **`test_it_refuses_during_COLLECTION_and_not_once_a_test_is_running`** — `--collect-only` não
  abre ligação nenhuma, portanto se o aborto na mesma dispara, disparou primeiro. A distinção é
  a guarda inteira (uma verificação dentro de um fixture já deixou o `pytest` chegar ao ponto em
  que a instrução seguinte é `DROP TABLE`) e era invisível a qualquer asserção que só olhasse o
  código de saída de uma corrida completa.

- **O teste de convenção deixou de poder passar em vazio.** Ele varre os módulos que leem o DSN;
  como esses deixaram de nomear a variável directamente, o termo de busca é o que pode
  envelhecer em silêncio — agora afirma também que a varredura ainda os encontra (≥5).

### Fixed

- **`README.md` ensinava `ENGRAM_TEST_DSN=…@localhost:55432/postgres`** — uma base que existe em
  TODOS os servidores, produção incluída, e que a própria guarda recusa. O comando documentado
  abortava. Agora não há DSN para escrever: `-e POSTGRES_DB=engram_test` no contentor e `pytest`.

- **Um módulo diferente abortava o schema tal como uma tabela plana.** O conserto anterior
  perguntava se a tabela estava particionada; `relkind` diz PARTICIONADA, **não com quê**. Medido
  na caixa demo a 2026-08-26, logo a seguir a esse merge: `turn_traces` com quatro filhos, o host
  a pedir oito, `partition "turn_traces_p4" would overlap partition "turn_traces_p0"` — e o grafo
  de conhecimento vem DEPOIS do laço. Mesma classe, mesma consequência, gatilho diferente.

  Agora há **duas defesas, e cada uma faz o que a outra não faz**: a sonda conta os filhos
  (`pg_inherits`) e, divergindo do pedido, salta com os DOIS números no evento
  (`reason=exists_with_4_partitions requested=8` — accionável, ao contrário de "would overlap");
  e o próprio DDL corre em transacção aninhada, rebaixando `InvalidObjectDefinition` e
  `InvalidTableDefinition` ao mesmo evento. A segunda existe porque perguntar nunca cobre todas
  as formas: uma tabela em LIST/RANGE de um produto pai, ou com outra CHAVE de partição, passa
  nas duas sondas e só a instrução a descobre. Qualquer outro erro (permissões, disco, um bug a
  sério) continua a levantar — esses não são "esta tabela tem história".

  O laço saiu para `_partition_existing_table`, com a razão inteira num docstring em vez de
  trinta linhas de comentário dentro do `ensure_schema`.


- **`ensure_schema` dizia-se idempotente e não era, contra uma base criada PLANA.**
  `CREATE TABLE IF NOT EXISTS turns (...) PARTITION BY HASH (scope)` é um NO-OP quando a tabela
  já existe — o Postgres não verifica que a definição bate — portanto uma base nascida sem
  partições (host antigo, ou `partition_by_scope=False`) chegava ao laço de partições com uma
  tabela plana, e o `PARTITION OF` levantava `InvalidObjectDefinition: "turns" is not
  partitioned`.

  Isso abortava a chamada INTEIRA, e o grafo de conhecimento é criado **onze instruções depois**.
  Medido numa caixa real a 2026-08-25: `sessions`/`turns`/`memories`/`turn_traces` existiam e
  `knowledge_edges` **não**, portanto o host corria sem grafo, o `/health` dizia `stale`, e a
  única pista no log era um erro de particionamento. O remédio documentado
  (`python -m cogno_host.migrate`, anunciado como idempotente) nunca podia consertá-lo, porque
  era exactamente a chamada que morria.

  **Particionamento é DÉBITO; as tabelas e colunas depois dele são CORRECÇÃO.** Uma optimização
  não pode ser fatal a um passo de correcção atrás dela. Agora a tabela é perguntada antes: se
  existir não-particionada, sai um `ERROR` que nomeia a tabela e o remédio
  (`event=partitioning_skipped`), salta as partições DESSA tabela e continua. Converter plana →
  particionada mexe dados e é decisão do operador, nunca efeito colateral de pedir um schema.

  Pinado nos dois sentidos: uma base plana recebe o resto do schema (e a tabela fica plana — o
  salto não converte), e uma base particionada continua a receber as suas partições.

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

- **Duas regressões que a própria dobragem introduziu, achadas em revisão adversarial:**
  - **O rótulo ACENTUADO perdia-se.** `ON CONFLICT` nunca escrevia `label`, portanto a primeira
    grafia a chegar ficava para sempre — e como o argumento desta funcionalidade é que "o contacto
    escreve o nome sem acento metade das vezes", a grafia SEM acento é a que chega primeiro com
    mais frequência. O contrário exacto do que o módulo promete. Agora a grafia com diacríticos
    SOBE e nunca desce (`folding.has_diacritics`, uma definição, os dois adaptadores).
  - **`delete_node` apagava DOIS nós.** `José` como PERSON e `Jose` como CONCEPT coexistem
    legalmente depois da migração, mas o comando recebe só um RÓTULO — e apagava ambos, com as
    arestas de ambos atrás por `ON DELETE CASCADE`. O `cogno-ui` chama-o com um id que o host
    converte em rótulo: o operador clicava num nó e perdia outro. `set_edge_status` e
    `set_edge_audience` alargavam igual, e a segunda é um controlo de PRIVACIDADE. Os três passam
    por `_one_node_id`: rótulo exacto ganha, ambíguo RECUSA. E `_resolve_node_id` (que cria a
    ligação, não destrói) passa a preferir o exacto de forma determinística em vez de `LIMIT 1`
    sem ordem.

- **O diagnóstico de colisão funciona também sem autocommit.** `ensure_schema` é API pública e
  recebe as duas espécies de conexão; sem autocommit o CREATE falhado envenenava a transacção, o
  diagnóstico degradava para lista vazia, e o operador recebia a chave dobrada que este módulo diz
  que ele não precisa. E deixou de truncar em silêncio aos 20 grupos.

- **REQUISITOS NOVOS do adaptador Postgres, e são duros.** `ensure_schema` passa a exigir:
  - a extensão **`unaccent`** disponível e instalável no schema `public` (o `ensure_schema`
    corre o `CREATE EXTENSION`, mas o pacote `postgresql-contrib` tem de estar presente);
  - suporte a **ICU** — a função de dobragem usa `COLLATE "und-x-icu"`, e sem ele o Postgres
    responde `collation "und-x-icu" for encoding "UTF8" does not exist`. Falha **cedo e alto**,
    na criação da função e portanto no primeiro `ensure_schema`, não numa consulta meses depois.

  Nenhum dos dois é exótico em PG ≥ 15 (a imagem `pgvector/pgvector:pg16` tem ambos), mas um
  requisito que só existe no código é um requisito que alguém descobre em produção.

- **Ferramenta de fusão para as colisões que a identidade nova cria**
  (`cogno_engram/fold_migration.py`): `fold_collisions()` devolve o relatório — que nós, que
  rótulos, quantas arestas cada um — e `merge_fold_collisions()` aplica, **com `dry_run=True` por
  omissão**. O `ensure_schema` continua a recusar-se a fundir sozinho, e isso está certo; mas
  parar aí deixava o operador com um traceback e um `psql`, e o SQL que ele escreveria à pressa é
  exactamente o perigoso: em Postgres as arestas referenciam `source_id`/`target_id` com
  `ON DELETE CASCADE`, portanto **apagar o nó duplicado leva as arestas dele consigo, sem aviso**.
  A ferramenta reponta primeiro e apaga depois, remove as que passariam a duplicar ou a apontar
  para si próprias, e guarda o rótulo perdido em `attributes.aliases` — perder a grafia é perder
  informação, e é o alias que permite desfazer a fusão à mão.

  Sem esta ferramenta o estado depois de uma migração recusada é pior do que parece, medido: o
  índice antigo de pé, a função criada, o índice novo ausente — e o código novo a ler isso faz
  **todo `upsert_node` levantar** até alguém fundir à mão. A ferramenta é o que separa "migração
  recusada com instrução" de "grafo morto para escrita".

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
