"""Testes de integração: léxico, prior, matriz, filtragem, entropia e endgame."""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from termo import entropia
from termo.entropia import Motor
from termo.feedback import (
    calcular_codigo,
    calcular_feedback,
    codigo_para_padrao,
    normalizar,
    padrao_possivel,
)
from termo.lexico import Lexico, calcular_prior
from termo.matriz import construir_matriz


@pytest.fixture(scope="module")
def lexico():
    return Lexico.carregar(1.0)


@pytest.fixture(scope="module")
def motor(lexico):
    return Motor(lexico)


# ------------------------------------------------------------------- léxico


def test_lexico_tem_o_tamanho_da_especificacao(lexico):
    """v1.1: só `lexico`, sem conjugações, normalizado e deduplicado."""
    assert len(lexico) == 6046
    assert lexico.palavras == sorted(lexico.palavras)
    assert len(lexico.exibicao) == len(lexico.palavras)


def test_formas_normalizadas_sao_unicas(lexico):
    """O ponto da dedupe (§2.3): candidatas indistinguíveis travariam o solver."""
    assert len(set(lexico.palavras)) == len(lexico)
    for normalizada, acentuada in zip(lexico.palavras, lexico.exibicao):
        assert normalizar(acentuada) == normalizada
        assert normalizada.isascii() and normalizada.islower()


def test_dedupe_mantem_a_variante_mais_comum(lexico):
    """De cada grupo fica a de menor ICF — 'terço' é mais comum que 'terçô'."""
    assert lexico.mostrar(lexico.indice_de("terco")) == "terço"
    assert lexico.mostrar(lexico.indice_de("agora")) == "agora"
    # e as variantes descartadas continuam encontráveis pela forma normalizada
    for variante in ("terço", "terçó", "terçô", "terco"):
        assert lexico.indice_de(variante) == lexico.indice_de("terco")


def test_conjugacoes_foram_excluidas(lexico):
    """Formas verbais que só vinham do arquivo `conjugações` sumiram.

    Não é uma varredura de "toda forma verbal": palavras como "abafa" continuam
    no léxico porque constam do dicionário geral por si próprias. O critério da
    v1.1 é a fonte, não a morfologia.
    """
    for conjugada in ("fugia", "curas", "zarpa", "andei", "amava"):
        assert conjugada not in lexico.indice, conjugada
    # e a primeira palavra do jogo (5 jan 2022, §2.7) tem que estar
    assert "festa" in lexico.indice


def test_todas_as_palavras_tem_icf(lexico):
    """A especificação (4.2) afirma cobertura de 100% — vale checar."""
    assert lexico.ausentes_no_icf == []
    assert lexico.icf.min() == pytest.approx(6.14, abs=0.01)  # "muito"


def test_prior_e_uma_distribuicao(lexico):
    assert lexico.prior.sum() == pytest.approx(1.0)
    assert (lexico.prior > 0).all()
    # ICF baixo = palavra comum = prior alto
    assert lexico.prior[lexico.indice_de("muito")] > lexico.prior[lexico.indice_de("termo")]
    mais_rara = int(np.argmax(lexico.icf))
    assert lexico.prior[lexico.indice_de("termo")] > lexico.prior[mais_rara]


def test_temperatura_infinita_recupera_uniforme(lexico):
    uniforme = calcular_prior(lexico.icf, math.inf)
    assert uniforme.sum() == pytest.approx(1.0)
    assert uniforme.std() == pytest.approx(0.0)
    # T grande se aproxima do uniforme de forma contínua
    assert calcular_prior(lexico.icf, 1000.0).std() < calcular_prior(lexico.icf, 1.0).std()


def test_prior_estavel_para_t_pequeno(lexico):
    """T pequeno faz ICF/T explodir; o log-sum-exp tem que segurar."""
    prior = calcular_prior(lexico.icf, 0.01)
    assert np.isfinite(prior).all()
    assert prior.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------- matriz


def test_matriz_vetorizada_bate_com_python_puro(lexico):
    random.seed(11)
    amostra = sorted(random.sample(lexico.palavras, 250))
    matriz = construir_matriz(amostra, bloco=64, verboso=False)
    for i, tentativa in enumerate(amostra):
        for j, secreta in enumerate(amostra):
            assert matriz[i, j] == calcular_codigo(tentativa, secreta), (tentativa, secreta)


def test_diagonal_da_matriz_e_vitoria(motor):
    diagonal = motor.matriz[np.arange(len(motor.lexico)), np.arange(len(motor.lexico))]
    assert (diagonal == 242).all()


def test_matriz_cabe_em_uint8(motor):
    assert motor.matriz.dtype == np.uint8
    assert motor.matriz.max() <= 242


# --------------------------------------------------------------- filtragem


def test_secreta_sempre_sobrevive_a_filtragem(motor):
    random.seed(3)
    todas = motor.todas_candidatas()
    for _ in range(30):
        secreta = random.randrange(len(motor.lexico))
        tentativa = random.randrange(len(motor.lexico))
        codigo = int(motor.matriz[tentativa, secreta])
        restantes = motor.filtrar(todas, motor.lexico.palavras[tentativa], codigo)
        assert secreta in set(restantes.tolist())


def test_filtragem_de_palavra_fora_do_lexico(motor):
    """Sem linha na matriz, o caminho lento tem que dar o mesmo resultado."""
    fora = "zzzzz"
    assert fora not in motor.lexico.indice
    todas = motor.todas_candidatas()
    restantes = motor.filtrar(todas, fora, 0)  # BBBBB
    esperadas = [
        i for i in range(len(motor.lexico))
        if calcular_feedback(fora, motor.lexico.palavras[i]) == "BBBBB"
    ]
    assert restantes.tolist() == esperadas


# ---------------------------------------------------------------- entropia


def test_entropia_bate_com_referencia_lenta(motor):
    candidatas = motor.filtrar(motor.todas_candidatas(), "tarso", 0)
    entropias = motor.entropias(candidatas)
    palavras, prior = motor.lexico.palavras, motor.lexico.prior

    random.seed(5)
    for g in random.sample(range(len(motor.lexico)), 40):
        baldes = defaultdict(float)
        for c in candidatas:
            baldes[calcular_feedback(palavras[g], palavras[c])] += prior[c]
        total = sum(baldes.values())
        esperada = -sum(
            (v / total) * math.log2(v / total) for v in baldes.values() if v > 0
        )
        assert entropias[g] == pytest.approx(esperada, abs=1e-9)


def test_caminho_rapido_de_entropia_bate_com_o_geral(motor, monkeypatch):
    """Com poucas candidatas a entropia usa outro algoritmo (`_entropias_poucas`).

    Ele existe só por velocidade — 10x com 4 candidatas, e é onde a busca do nível
    3 passa a maior parte do tempo —, então o que se exige dele é ser
    indistinguível do caminho geral, não "parecido". `LIMIAR_POUCAS = 0` desliga.
    """
    random.seed(23)
    conjuntos = [
        np.array(sorted(random.sample(range(len(motor.lexico)), tamanho)), dtype=np.int32)
        for tamanho in (2, 3, 5, 8, 12)
    ]
    rapidas = [motor.entropias(c) for c in conjuntos]

    monkeypatch.setattr(entropia, "LIMIAR_POUCAS", 0)
    for candidatas, rapida in zip(conjuntos, rapidas):
        geral = motor.entropias(candidatas)
        assert rapida == pytest.approx(geral, abs=1e-12), len(candidatas)


def test_entropia_rapida_aguenta_prior_degenerado(lexico):
    """T minúsculo zera grupos inteiros, e log₂(0) é o jeito óbvio de errar isto."""
    degenerado = lexico.com_temperatura(0.01)
    motor = Motor(degenerado)
    candidatas = np.arange(len(degenerado) - 6, len(degenerado), dtype=np.int32)
    assert degenerado.prior[candidatas].sum() == 0.0  # o cenário existe mesmo
    entropias = motor.entropias(candidatas)
    assert np.isfinite(entropias).all()


def test_entropia_e_zero_com_uma_candidata(motor):
    assert (motor.entropias(np.array([7], dtype=np.int32)) == 0).all()


def test_entropia_nunca_passa_do_maximo_teorico(motor):
    """H <= log2(nº de padrões distintos) <= log2(243)."""
    candidatas = motor.filtrar(motor.todas_candidatas(), "tarso", 0)
    assert motor.entropias(candidatas).max() <= math.log2(243) + 1e-9


# ----------------------------------------------------------------- endgame


def test_uma_candidata_e_chutada(motor):
    i = motor.lexico.indice_de("termo")
    sugestao = motor.escolher(np.array([i], dtype=np.int32), tentativa=2)
    assert sugestao.palavra == "termo" and sugestao.e_candidata


def test_duas_candidatas_chutam_a_mais_provavel(motor):
    par = np.array(
        [motor.lexico.indice_de("termo"), motor.lexico.indice_de("ababá")],
        dtype=np.int32,
    )
    assert motor.escolher(par, tentativa=2).palavra == "termo"


def test_ultima_tentativa_nunca_queima_uma_nao_candidata(motor):
    candidatas = motor.filtrar(motor.todas_candidatas(), "tarso", 0)[:40]
    sugestao = motor.escolher(candidatas, tentativa=6)
    assert sugestao.e_candidata
    assert sugestao.indice == max(candidatas, key=lambda i: motor.lexico.prior[i])


@pytest.mark.parametrize("temperatura", [math.inf, 0.01, 1.0])
def test_ultima_tentativa_chuta_a_mais_comum_em_qualquer_t(lexico, temperatura):
    """A aposta cega é sempre a palavra mais comum — o T não muda isso.

    Com T=inf o prior é uniforme e com T=0,01 ele satura em zero: nos dois o
    `argmax(prior)` devolvia a primeira do índice, não a mais comum.
    """
    motor = Motor(lexico.com_temperatura(temperatura))
    candidatas = motor.filtrar(motor.todas_candidatas(), "tarso", 0)[:40]
    sugestao = motor.escolher(candidatas, tentativa=6)
    assert sugestao.indice == min(candidatas, key=lambda i: lexico.icf[i])
    # e as alternativas seguem na mesma moeda: da mais comum para a mais rara
    icfs = [lexico.icf[lexico.indice_de(p)] for p, _, _ in sugestao.alternativas]
    assert icfs == sorted(icfs) and lexico.icf[sugestao.indice] <= icfs[0]


def test_penultima_tentativa_com_c_pequeno_chuta_candidata(motor):
    candidatas = motor.filtrar(motor.todas_candidatas(), "tarso", 0)[:3]
    assert motor.escolher(candidatas, tentativa=5).e_candidata
    # com C grande a regra não se aplica: volta a valer a entropia
    grande = motor.filtrar(motor.todas_candidatas(), "tarso", 0)
    assert len(grande) > motor.limiar_endgame
    motor.escolher(grande, tentativa=5)  # não deve levantar


def test_abertura_e_deterministica(motor):
    primeira = motor.abertura(usar_cache=False)
    segunda = motor.abertura(usar_cache=False)
    assert primeira.palavra == segunda.palavra
    assert primeira.entropia == pytest.approx(segunda.entropia)


# ------------------------------------------------------- padrões impossíveis


def test_padroes_realizaveis_sao_aceitos(motor):
    """Todo padrão que o léxico realmente produz tem que passar no detector."""
    random.seed(13)
    for i in random.sample(range(len(motor.lexico)), 120):
        tentativa = motor.lexico.palavras[i]
        for codigo in np.unique(motor.matriz[i]):
            padrao = codigo_para_padrao(int(codigo))
            assert padrao_possivel(tentativa, padrao), (tentativa, padrao)


def test_padroes_impossiveis_sao_rejeitados():
    # 4 verdes + 1 amarelo: sobra uma única posição, e é a da própria letra
    assert not padrao_possivel("tarso", "GGGGY")
    assert not padrao_possivel("tarso", "YGGGG")
    # preto antes de amarelo na mesma letra: a passada 2 pinta da esquerda
    assert not padrao_possivel("oovxx", "BYBBB")
    assert padrao_possivel("oovxx", "YBBBB")
    # padrões banais continuam válidos
    assert padrao_possivel("tarso", "BBBBB")
    assert padrao_possivel("tarso", "GGGGG")
    assert padrao_possivel("tarso", "GGGGB")


# ------------------------------------------------------------- jogo completo


def test_partidas_completas_sempre_terminam(motor):
    """Nenhum jogo pode zerar C nem estourar as 6 tentativas com o motor."""
    from benchmark import jogar
    from termo.estrategias import Entropia

    estrategia = Entropia(motor)
    random.seed(17)
    for secreta in random.sample(range(len(motor.lexico)), 25):
        n, jogadas = jogar(motor, estrategia, secreta)
        assert n is not None, motor.lexico.palavras[secreta]
        assert jogadas[-1] == secreta
        assert 1 <= n <= 6
