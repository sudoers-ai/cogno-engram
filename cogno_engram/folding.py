"""How a label is FOLDED before two of them are compared — one algorithm, both adapters.

`José` e `Jose` são a mesma pessoa. Essa frase é do dono do produto, e é uma decisão de
PRODUTO, não um detalhe de implementação: num CRM que recebe mensagens de WhatsApp, o
contacto escreve o nome sem acento metade das vezes, e um grafo que trate os dois como nós
distintos parte a vida da pessoa em duas.

O que este módulo NÃO é: um normalizador Unicode de uso geral. É a definição, num sítio só, de
"estes dois rótulos são o mesmo rótulo" — e a razão de existir é que essa definição estava
ESCRITA DUAS VEZES e as duas cópias discordavam.

## O defeito que o originou

O adaptador Postgres comparava com `lower(label)`; o in-memory com `label.lower()` do Python. Sob
um cluster criado `LC_COLLATE 'C'` — `initdb --locale=C`, `postgres:alpine`, escolha comum por
desempenho — o `lower()` do Postgres **não dobra sequer as maiúsculas acentuadas**:

    lower('JOSÉ') = 'josÉ'      -- o É fica intacto
    'JOSÉ'.lower() = 'josé'     -- o Python dobra

Logo `find_node(scope, "JOSÉ")` devolvia `None` para um nó gravado como "josé", e os dois
adaptadores — que esta lib promete serem intermutáveis — davam respostas diferentes para a mesma
pergunta. Medido: 7 de 14 rótulos falhavam sob `C`, 0 sob `en_US.utf8`; era um defeito que só
aparecia no cluster de outra pessoa.

## As três camadas, e porque são três

1. **`casefold()`, não `lower()`.** `casefold` é a dobragem para COMPARAÇÃO: trata `ß` → `ss`, que
   `lower` deixa quieto. Medido contra o Postgres, `lower` divergia em `ß` e `casefold` não.
2. **NFD + remover marcas combinantes.** É o que tira o acento: `josé` → `jose`.
3. **`_TRANSLIT`, e é a camada que ninguém adivinha.** O `unaccent` do Postgres não se limita a
   tirar marcas — carrega um dicionário de TRANSLITERAÇÃO: `æ`→`ae`, `ø`→`o`, `ł`→`l`. Esses
   caracteres não têm decomposição combinante, portanto o passo 2 deixa-os intactos e as duas
   metades divergem. A tabela abaixo foi **DERIVADA do Postgres**, não escrita à mão — e derivada
   sobre a forma **pós-`casefold`+NFD**, que é onde ela se aplica. A primeira derivação foi feita
   sobre os caracteres ORIGINAIS e produziu uma entrada `"ʼn"` de DUAS letras: como a tabela é
   consultada caractere a caractere, essa linha nunca disparava — entrada morta que dava resultado
   errado sem parecer errada. O teste de paridade apanhou-a. A entrada certa é `"ʼ": "'"`, um
   caractere. São 15 mapeamentos sobre os 34 caracteres que sobrevivem ao passo 2. `tests/test_folding_parity.py` re-deriva-a contra um Postgres
   a sério e falha se alguma vez deixar de bater, porque uma tabela copiada à mão de um dicionário
   que vive noutro processo é uma cópia que apodrece em silêncio.

## O tecto, dito em voz alta

O acordo é garantido sobre **Latin-1 Supplement + Latin Extended-A** — o alfabeto onde vivem os
nomes pt/es/en/de/fr/it, que é o domínio deste produto. Fora dele os dois lados PODEM divergir
(medido: sigma final grego dá `σ` no Python e `ς` no Postgres). Não é esquecimento: fazer os dois
concordarem em todo o Unicode exigiria portar o dicionário inteiro do `unaccent`, e um domínio
maior do que o produto tem não paga a cópia. O teste de paridade declara o alfabeto que cobre.
"""

from __future__ import annotations

import unicodedata

# DERIVADA do `unaccent` do Postgres sobre Latin-1 Sup + Latin Extended-A, não escrita à mão.
# Só as 15 de 190 onde `casefold` + NFD NÃO chega ao mesmo resultado — os caracteres sem
# decomposição combinante, que o Postgres translitera por dicionário.
_TRANSLIT = {
    "æ": "ae", "ð": "d", "ø": "o", "þ": "th",
    "đ": "d", "ħ": "h", "ı": "i", "ĳ": "ij",
    "ĸ": "q", "ŀ": "l", "ł": "l", "ŋ": "n",
    "œ": "oe", "ŧ": "t", "ʼ": "'",
}


def fold_label(label: str) -> str:
    """A forma canónica de um rótulo para COMPARAÇÃO — nunca para exibição.

    O que sai daqui é uma chave, não um nome: `José` e `Jose` saem ambos `jose`, e é o rótulo
    ORIGINAL que continua a ser guardado e mostrado. Confundir os dois far-se-ia perder o acento
    no nome de uma pessoa, que é precisamente o oposto do que esta função existe para conseguir.
    """
    dobrado = unicodedata.normalize("NFD", label.casefold())
    sem_marcas = "".join(c for c in dobrado if not unicodedata.combining(c))
    return "".join(_TRANSLIT.get(c, c) for c in sem_marcas)


def has_diacritics(label: str) -> bool:
    """O rótulo carrega informação que a dobragem apaga? (acento, cedilha, ligadura, `ł`…)

    É o que decide qual grafia SOBE quando duas colidem: `José` carrega, `Jose` não. Existe
    como função porque a primeira tentativa escreveu o predicado à mão e escreveu-o ERRADO —
    `fold_label(x) == x` parece dizer "não tem acento" e diz outra coisa, porque `fold_label`
    também baixa a caixa: `'Jose'` falhava o teste por causa do J maiúsculo. A comparação certa
    é contra o próprio `casefold`, que isola a diferença que interessa."""
    return fold_label(label) != label.casefold()


#: O SQL equivalente do `fold_label`, e a razão de ser uma FUNÇÃO na base em vez de uma expressão inline.
#:
#: O `unaccent` é declarado STABLE, não IMMUTABLE — porque depende de um dicionário que o
#: administrador pode trocar — e o Postgres RECUSA uma função STABLE dentro de um índice. A forma
#: de dois argumentos fixa o dicionário explicitamente, que é o que torna honesto marcar o wrapper
#: IMMUTABLE. Sem isto, a UNIQUE `(scope, fold(label), node_type)` não pode existir.
#:
#: O `COLLATE "und-x-icu"` não é decoração: é ele que faz o `lower` dobrar maiúsculas acentuadas
#: sob um cluster `C`. Sem ele, `unaccent(lower('JOSÉ'))` dá `josE` e `unaccent(lower('josé'))` dá
#: `jose` — o defeito original, com um passo a mais.
FOLD_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION engram_fold(text) RETURNS text
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
$$ SELECT public.unaccent('public.unaccent', lower($1 COLLATE "und-x-icu")) $$
"""


#: O SQL equivalente de :func:`has_diacritics` — a mesma pergunta, do lado do banco.
HAS_DIACRITICS_SQL = 'engram_fold({col}) <> lower({col} COLLATE "und-x-icu")'
