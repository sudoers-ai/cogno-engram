"""`José` e `Jose` são a mesma pessoa — e as três camadas que isso precisa.

Decisão de PRODUTO do dono (2026-08-25): num CRM que recebe WhatsApp, o contacto escreve o nome
sem acento metade das vezes, e um grafo que trate os dois como nós distintos parte a vida da
pessoa em duas.
"""

from __future__ import annotations

import pytest

from cogno_engram.folding import _TRANSLIT, fold_label


@pytest.mark.parametrize("a,b", [
    ("José", "Jose"),          # a frase do dono, literalmente
    ("JOSÉ", "josé"),          # o defeito original: sob `C`, lower() dava 'josÉ'
    ("MÜLLER", "muller"),
    ("Ção", "cao"),
    ("Añez", "anez"),
    ("Łódź", "lodz"),          # transliteração: `ł` não tem decomposição combinante
    ("straße", "STRASSE"),     # casefold, que lower() não faz
    ("Œuvre", "oeuvre"),
])
def test_these_are_the_SAME_label(a, b):
    assert fold_label(a) == fold_label(b), f"{a!r} e {b!r} deviam ser o mesmo rótulo"


@pytest.mark.parametrize("a,b", [
    ("Ivan", "Ivana"),         # prefixo não é identidade
    ("José", "Josué"),         # dois nomes, ambos com acento — dobrar não é apagar
    ("Ana", "Ane"),
])
def test_these_are_DIFFERENT_labels(a, b):
    assert fold_label(a) != fold_label(b)


def test_the_fold_is_a_KEY_not_a_name():
    """O que sai daqui compara; o que se GUARDA e se MOSTRA é o rótulo original.

    Confundir os dois faria perder o acento no nome de uma pessoa — o oposto exacto do que esta
    função existe para conseguir."""
    assert fold_label("José") == "jose"
    assert "José" != fold_label("José")


def test_the_translit_table_is_lowercase_only():
    """`fold_label` corre `casefold()` ANTES da tabela, portanto uma chave maiúscula nunca casaria
    — seria uma entrada morta, e uma tabela com entrada morta mente sobre o que cobre."""
    assert all(k == k.casefold() for k in _TRANSLIT), (
        f"chaves que nunca casam: {[k for k in _TRANSLIT if k != k.casefold()]}")


def test_folding_is_idempotent():
    """Um rótulo já dobrado tem de dobrar para si próprio — senão comparar uma chave guardada com
    uma chave nova daria falso negativo consoante quantas vezes passou pela função."""
    for r in ["José", "Łódź", "straße", "Œuvre", "MÜLLER", "ﬁcheiro"]:
        assert fold_label(fold_label(r)) == fold_label(r), r
