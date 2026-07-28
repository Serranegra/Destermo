"""Suite obrigatória da seção 3.3 da especificação v1.1.

DOIS casos da tabela da especificação continuam com o valor esperado errado
(derivação manual abaixo). O caso 3 já estava errado na v1.0 e não foi corrigido
na v1.1; o caso 8 vem marcado como "verificar".

  Caso 3 — secreta "carro", tentativa "rerum"
      A especificação diz `BGBYB`, mas o G está na posição errada.
      pos0 r vs c, pos1 e vs a, pos2 r vs r (G), pos3 u vs r, pos4 m vs o
      Passada 1 -> "..G.."; estoque restante {c:1, a:1, r:1, o:1}
      Passada 2 -> pos0 'r' consome o r restante (Y); e/u/m não existem (B)
      Correto: "YBGBB"

  Caso 8 — secreta "termo", tentativa "metro"
      Passada 1 -> pos1 e==e (G), pos4 o==o (G); estoque restante {t:1, r:1, m:1}
      Passada 2 -> pos0 'm' (Y), pos2 't' (Y), pos3 'r' (Y)
      Correto: "YGYYG"

O ponto testado em cada caso (o que a especificação queria cobrir) continua o mesmo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from termo.feedback import (
    N_PADROES,
    PADRAO_VITORIA,
    calcular_codigo,
    calcular_feedback,
    codigo_para_padrao,
    normalizar,
    normalizar_padrao,
    padrao_para_codigo,
)

CASOS_OBRIGATORIOS = [
    # (secreta, tentativa, esperado, o que testa)
    ("arara", "ratos", "YYBBB", "amarelos sem verdes; estoque múltiplo"),
    ("arara", "arara", "GGGGG", "acerto total"),
    ("carro", "rerum", "YBGBB", "duplicada na tentativa, secreta tem 2 R"),
    ("banco", "barra", "GGBBB", "duplicada na tentativa, secreta tem 1 R"),
    ("saúde", "saude", "GGGGG", "acento não conta: secreta acentuada"),
    ("saude", "saúde", "GGGGG", "normalização simétrica"),
    ("açude", "acude", "GGGGG", "cedilha normaliza para c"),
    ("termo", "metro", "YGYYG", "verdes têm prioridade sobre amarelos"),
]


@pytest.mark.parametrize(
    "secreta,tentativa,esperado,descricao",
    CASOS_OBRIGATORIOS,
    ids=[c[3] for c in CASOS_OBRIGATORIOS],
)
def test_casos_obrigatorios(secreta, tentativa, esperado, descricao):
    assert calcular_feedback(tentativa, secreta) == esperado


def test_regra_de_ouro_estoque_unico():
    """Tentativa com 2 ocorrências, secreta com 1: apenas uma fica colorida."""
    # secreta "porta" tem um único 'o'; tentativa tem dois e nenhum casa de posição
    # -> só o primeiro vira amarelo, o segundo fica preto
    assert calcular_feedback("oxoxx", "porta") == "YBBBB"
    # agora um dos 'o' casa de posição: o verde consome o estoque na passada 1
    # e o 'o' anterior (que numa implementação ingênua roubaria o estoque) fica preto
    assert calcular_feedback("oovxx", "porta") == "BGBBB"
    # verde à direita de uma repetição, mesma ideia
    assert calcular_feedback("ovoxx", "ovaxx") == "GGBGG"


def test_acentos_nao_afetam_as_cores():
    """v1.1: o Termo preenche os acentos sozinho e não os considera nas dicas."""
    assert calcular_feedback("saude", "saúde") == "GGGGG"
    assert calcular_feedback("cafes", "cafés") == "GGGGG"
    assert calcular_feedback("cafés", "cafes") == "GGGGG"
    # e o mesmo vale para amarelos e pretos
    assert calcular_feedback("terco", "terço") == "GGGGG"
    assert calcular_feedback("corte", "terço") == calcular_feedback("corte", "terco")


def test_normalizar():
    assert normalizar("saúde") == "saude"
    assert normalizar("terço") == "terco"
    assert normalizar("ágora") == "agora"
    assert normalizar("piquê") == "pique"
    assert normalizar("SAÚDE") == "saude"
    # já normalizada é ponto fixo, e o comprimento em caracteres se mantém
    for palavra in ("saúde", "terço", "ababá", "abunã", "leruê"):
        assert normalizar(normalizar(palavra)) == normalizar(palavra)
        assert len(normalizar(palavra)) == len(palavra) == 5
        assert normalizar(palavra).isascii()


def test_palavra_igual_sempre_vitoria():
    for palavra in ("arara", "saúde", "termo", "aaaaa", "abcde"):
        assert calcular_feedback(palavra, palavra) == "GGGGG"
        assert calcular_codigo(palavra, palavra) == PADRAO_VITORIA


def test_codificacao_exemplo_da_especificacao():
    assert padrao_para_codigo("YGBBB") == 135
    assert padrao_para_codigo("BBBBB") == 0
    assert padrao_para_codigo("GGGGG") == N_PADROES - 1 == 242


def test_codificacao_ida_e_volta():
    for codigo in range(N_PADROES):
        assert padrao_para_codigo(codigo_para_padrao(codigo)) == codigo


def test_normalizar_padrao():
    assert normalizar_padrao("ygbbb") == "YGBBB"
    assert normalizar_padrao(" Y G B B B ") == "YGBBB"
    assert normalizar_padrao("🟨🟩⬛⬛⬛") == "YGBBB"
    assert normalizar_padrao("12000") == "YGBBB"
    with pytest.raises(ValueError):
        normalizar_padrao("YGBB")
    with pytest.raises(ValueError):
        normalizar_padrao("YGBBZ")


def test_tamanho_invalido():
    with pytest.raises(ValueError):
        calcular_feedback("abcd", "abcde")
    with pytest.raises(ValueError):
        calcular_feedback("abcde", "abcdef")


def test_numero_de_verdes_e_amarelos_e_simetrico():
    """|G|+|Y| de (a contra b) deve ser igual a |G|+|Y| de (b contra a).

    O total de letras 'casadas' é uma propriedade do multiconjunto interseção,
    logo não depende de qual palavra é a tentativa. Pega erros de consumo.
    """
    palavras = ["arara", "carro", "banco", "termo", "metro", "saúde", "ratos", "porta"]
    for a in palavras:
        for b in palavras:
            coloridas_ab = sum(c != "B" for c in calcular_feedback(a, b))
            coloridas_ba = sum(c != "B" for c in calcular_feedback(b, a))
            assert coloridas_ab == coloridas_ba, (a, b)


def test_verde_implica_letra_na_posicao():
    palavras = ["arara", "carro", "banco", "termo", "metro", "saúde"]
    for a in palavras:
        for b in palavras:
            for i, marca in enumerate(calcular_feedback(a, b)):
                if marca == "G":
                    assert a[i] == b[i]
                elif marca == "Y":
                    assert a[i] != b[i] and a[i] in b
