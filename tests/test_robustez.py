"""Perplexidade, família de priors e os dois sanity checks da matriz de regret.

A perplexidade tem oráculo exato — uniforme sobre K palavras vale K —, então ela
é testada e fica de fora da assinatura do cache (mesma divisão de trabalho que
`Motor.entropias`: hash para o que não tem oráculo, teste para o que tem).

Os checks da matriz rodam sobre uma grade minúscula, com léxico e matriz de
verdade mas três mundos e duas candidatas: o que se quer verificar é a álgebra do
arrependimento, não o valor numérico de nenhuma abertura.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from termo.lexico import Lexico, calcular_prior
from termo.matriz import carregar_matriz
from termo.robustez import (
    ROTULO_PADRAO,
    Cenario,
    conferir,
    matriz_de_regret,
    perplexidade,
    prior_familia,
    temperatura_para,
)


# ------------------------------------------------------------- perplexidade


@pytest.mark.parametrize("k", [1, 2, 5, 100, 6046])
def test_perplexidade_do_uniforme_e_o_proprio_k(k):
    """A propriedade que dá sentido à unidade: uniforme sobre K vale K."""
    assert perplexidade(np.full(k, 1.0 / k)) == pytest.approx(k)


def test_perplexidade_ignora_peso_zero():
    """Palavra com peso 0 não está no suporte e não pode inflar M."""
    prior = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
    assert perplexidade(prior) == pytest.approx(4.0)


def test_perplexidade_e_maxima_no_uniforme():
    """Qualquer desbalanceamento reduz o número efetivo de palavras."""
    uniforme = np.full(8, 1.0 / 8)
    torto = np.array([0.5] + [0.5 / 7] * 7)
    assert perplexidade(torto) < perplexidade(uniforme) == pytest.approx(8.0)


def test_perplexidade_do_degenerado_e_um():
    """Toda a massa numa palavra só: o mundo tem uma palavra efetiva."""
    assert perplexidade(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_perplexidade_rejeita_prior_vazio():
    with pytest.raises(ValueError):
        perplexidade(np.zeros(5))


# ------------------------------------------------- família de priors e busca


def test_prior_familia_trunca_e_normaliza():
    icf = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prior = prior_familia(icf, 3, 1.0)
    assert prior[3:].tolist() == [0.0, 0.0]      # fora do corte, peso zero
    assert prior.sum() == pytest.approx(1.0)     # renormalizado dentro do corte
    assert prior[0] > prior[1] > prior[2]        # ICF menor = mais comum = mais peso


def test_prior_familia_no_uniforme_recupera_o_corte():
    icf = np.arange(10.0)
    assert perplexidade(prior_familia(icf, 4, math.inf)) == pytest.approx(4.0)


def test_temperatura_para_atinge_o_alvo():
    """A bisseção é sobre uma função monótona; o alvo tem que sair de volta."""
    icf = Lexico.carregar(1.0).icf
    for alvo in (300.0, 1200.0, 4800.0):
        t = temperatura_para(icf, alvo)
        assert perplexidade(prior_familia(icf, icf.size, t)) == pytest.approx(
            alvo, rel=1e-3
        )


def test_temperatura_para_no_teto_devolve_infinito():
    """M >= N só é atingível pelo uniforme exato, não por T grande."""
    icf = np.arange(50.0)
    assert math.isinf(temperatura_para(icf, 50.0, 50))
    assert math.isinf(temperatura_para(icf, 999.0, 50))


def test_m_cresce_com_a_temperatura():
    icf = Lexico.carregar(1.0).icf
    valores = [
        perplexidade(prior_familia(icf, icf.size, t)) for t in (0.25, 0.5, 1, 2, 4)
    ]
    assert valores == sorted(valores)


def test_prior_familia_com_t_1_reproduz_o_prior_do_projeto():
    """A família contém o prior padrão como caso particular (N=todas, T=1)."""
    lexico = Lexico.carregar(1.0)
    familia = prior_familia(lexico.icf, len(lexico), 1.0)
    assert np.allclose(familia, calcular_prior(lexico.icf, 1.0))


# --------------------------------------------------------- sanity checks


@pytest.fixture(scope="module")
def matriz_pequena():
    """Grade mínima: três mundos, duas candidatas. Só a álgebra importa aqui."""
    lexico = Lexico.carregar(1.0)
    matriz = carregar_matriz(lexico.palavras)
    n = len(lexico)
    cenarios = []
    for temperatura, rotulo in ((0.6, ""), (1.0, ROTULO_PADRAO), (math.inf, "")):
        prior = prior_familia(lexico.icf, n, temperatura)
        cenarios.append(
            Cenario(perplexidade(prior), n, temperatura, rotulo, prior)
        )
    return matriz_de_regret(lexico, matriz, cenarios, extras=("tarso", "tosar"))


def test_check1_tarso_e_a_referencia_em_bits_sob_t1(matriz_pequena):
    """§6: regret(tarso, mundo do T=1) contra a referência em bits tem que ser 0.

    `tarso` É a abertura que o nível 2 escolhe sob T=1 — é a definição dela. Se
    esta não zerar, o pipeline está lendo o prior errado em algum lugar, e nenhum
    outro número do módulo vale.
    """
    coluna = next(
        c["descricao"] for c in matriz_pequena["cenarios"]
        if c["rotulo"] == ROTULO_PADRAO
    )
    assert matriz_pequena["cenarios"][1]["referencia_bits"] == "tarso"
    assert matriz_pequena["regret_vs_bits"]["tarso"][coluna] == pytest.approx(0.0)


def test_check2_regret_nao_negativo(matriz_pequena):
    """§6: contra a referência em TENTATIVAS nada pode ficar abaixo de zero.

    Ela é o mínimo do conjunto avaliado, então um negativo aqui seria erro de
    contabilidade — diferente do `regret_vs_bits`, que é negativo de propósito.
    """
    for palavra, linha in matriz_pequena["regret"].items():
        for mundo, valor in linha.items():
            assert valor >= -1e-9, f"regret({palavra}, {mundo}) = {valor}"


def test_conferir_nao_acha_violacao(matriz_pequena):
    assert conferir(matriz_pequena) == []


def test_referencia_de_cada_coluna_tem_regret_zero(matriz_pequena):
    """A abertura eleita de uma coluna não se arrepende de nada nela."""
    for coluna in matriz_pequena["cenarios"]:
        eleita = coluna["referencia_tentativas"]
        assert matriz_pequena["regret"][eleita][coluna["descricao"]] == pytest.approx(
            0.0
        )


def test_regret_e_a_diferenca_dos_esperados(matriz_pequena):
    """Contabilidade: regret = E[g] - E[referência], célula a célula."""
    for coluna in matriz_pequena["cenarios"]:
        chave = coluna["descricao"]
        for palavra in matriz_pequena["candidatas"]:
            esperado = matriz_pequena["esperados"][palavra][chave]
            assert matriz_pequena["regret"][palavra][chave] == pytest.approx(
                esperado - coluna["esperado_tentativas"]
            )
            assert matriz_pequena["regret_vs_bits"][palavra][chave] == pytest.approx(
                esperado - coluna["esperado_bits"]
            )
