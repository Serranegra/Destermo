"""Testes do espaço de tentativa ampliado (§2.5).

A propriedade que sustenta o desenho todo é uma só: as candidatas são o PREFIXO
das sondas. Se ela vale, nenhum índice de candidata muda de significado ao ligar
`--ampliado`, e o resto do motor — filtragem, prior, endgame — continua correto
sem saber que o espaço cresceu. Quase todo teste aqui é uma consequência dela.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from termo.entropia import Motor
from termo.feedback import calcular_codigo, calcular_feedback, normalizar
from termo.lexico import Lexico
from termo.matriz import construir_matriz

# Formas verbais que o teste `test_conjugacoes_foram_excluidas` garante estarem
# FORA do léxico — aqui elas são o que se espera encontrar no acréscimo.
CONJUGADAS = ("fugia", "curas", "zarpa", "andei", "amava")


@pytest.fixture(scope="module")
def lexico():
    return Lexico.carregar(1.0, ampliado=True)


@pytest.fixture(scope="module")
def motor(lexico):
    return Motor(lexico)


@pytest.fixture(scope="module")
def motor_padrao():
    """O motor de sempre, para comparar espaço a espaço."""
    return Motor(Lexico.carregar(1.0))


# -------------------------------------------------------------- estrutura


def test_candidatas_sao_prefixo_das_sondas(lexico):
    """A invariante do módulo. Tudo o mais depende dela."""
    assert lexico.n_sondas > len(lexico)
    assert lexico.sondas[: len(lexico)] == lexico.palavras
    assert lexico.sondas_exibicao[: len(lexico)] == lexico.exibicao
    for palavra in lexico.palavras:
        assert lexico.indice_sonda[palavra] == lexico.indice[palavra]


def test_sondas_sao_unicas_e_normalizadas(lexico):
    assert len(set(lexico.sondas)) == lexico.n_sondas
    for normalizada, acentuada in zip(lexico.sondas, lexico.sondas_exibicao):
        assert normalizar(acentuada) == normalizada
        assert normalizada.isascii() and normalizada.islower()


def test_conjugacoes_entram_como_sonda_e_nao_como_resposta(lexico):
    for conjugada in CONJUGADAS:
        assert conjugada not in lexico.indice, conjugada
        indice = lexico.indice_de(conjugada)
        assert indice >= len(lexico)
        assert not lexico.e_candidata(indice)


def test_prior_das_sondas_extras_e_zero(lexico):
    """Não é um chute de raridade: uma conjugação NÃO pode ser a palavra do dia."""
    prior = lexico.prior_sondas
    assert len(prior) == lexico.n_sondas
    assert np.array_equal(prior[: len(lexico)], lexico.prior)
    assert (prior[len(lexico) :] == 0.0).all()
    assert prior.sum() == pytest.approx(1.0)


def test_com_temperatura_preserva_o_espaco_ampliado(lexico):
    outro = lexico.com_temperatura(5.0)
    assert outro.ampliado
    assert outro.sondas == lexico.sondas
    assert (outro.prior_sondas[len(outro) :] == 0.0).all()
    assert outro.prior_sondas[: len(outro)] == pytest.approx(outro.prior)


def test_lexico_sem_ampliar_tem_os_dois_espacos_iguais(motor_padrao):
    lexico = motor_padrao.lexico
    assert not lexico.ampliado
    assert lexico.n_sondas == len(lexico)
    assert lexico.sondas is lexico.palavras
    assert lexico.indice_sonda is lexico.indice


# ----------------------------------------------------------------- matriz


def test_matriz_retangular_bate_com_python_puro(lexico):
    """Recorte pequeno com linhas dos dois lados da fronteira das candidatas."""
    random.seed(17)
    secretas = sorted(random.sample(lexico.palavras, 120))
    extras = sorted(random.sample(lexico.sondas[len(lexico) :], 80))
    sondas = secretas + extras

    matriz = construir_matriz(sondas, secretas, bloco=32, verboso=False)
    assert matriz.shape == (len(sondas), len(secretas))
    for i, tentativa in enumerate(sondas):
        for j, secreta in enumerate(secretas):
            assert matriz[i, j] == calcular_codigo(tentativa, secreta), (
                tentativa,
                secreta,
            )


def test_matriz_ampliada_tem_a_matriz_padrao_no_topo(motor, motor_padrao):
    """Prefixo nos índices, prefixo na matriz — senão a filtragem divergiria."""
    n = len(motor.lexico)
    assert motor.matriz.shape == (motor.lexico.n_sondas, n)
    assert np.array_equal(motor.matriz[:n], motor_padrao.matriz)


def test_linhas_extras_batem_com_o_feedback(motor):
    random.seed(23)
    n = len(motor.lexico)
    for i in random.sample(range(n, motor.lexico.n_sondas), 20):
        tentativa = motor.lexico.sondas[i]
        for j in random.sample(range(n), 25):
            assert motor.matriz[i, j] == calcular_codigo(
                tentativa, motor.lexico.palavras[j]
            )


# -------------------------------------------------------------- filtragem


def test_filtrar_por_conjugacao_usa_a_matriz_e_acerta(motor):
    """Antes de ampliar isto caía no caminho lento; agora tem linha própria."""
    tentativa = "fugia"
    assert tentativa in motor.lexico.indice_sonda
    todas = motor.todas_candidatas()
    restantes = motor.filtrar(todas, tentativa, 0)  # BBBBB
    esperadas = [
        i
        for i in range(len(motor.lexico))
        if calcular_feedback(tentativa, motor.lexico.palavras[i]) == "BBBBB"
    ]
    assert restantes.tolist() == esperadas
    assert len(restantes) < len(todas)


def test_filtragem_das_candidatas_nao_muda(motor, motor_padrao):
    """Ampliar mexe no que se pode JOGAR, nunca no que pode ser a resposta."""
    random.seed(5)
    for palavra in random.sample(motor.lexico.palavras, 15):
        for codigo in (0, 121, 242):
            assert np.array_equal(
                motor.filtrar(motor.todas_candidatas(), palavra, codigo),
                motor_padrao.filtrar(motor_padrao.todas_candidatas(), palavra, codigo),
            )


# --------------------------------------------------------------- entropia


def _estado(motor, tentativa="tarso", codigo=0):
    return motor.filtrar(motor.todas_candidatas(), tentativa, codigo)


def test_entropia_das_candidatas_nao_muda(motor, motor_padrao):
    candidatas = _estado(motor)
    entropias = motor.entropias(candidatas)
    assert len(entropias) == motor.lexico.n_sondas
    assert entropias[: len(motor.lexico)] == pytest.approx(
        motor_padrao.entropias(_estado(motor_padrao))
    )


def test_ampliar_nunca_piora_a_melhor_entropia(motor, motor_padrao):
    """Mais opções e o mesmo critério: o máximo só pode subir."""
    for tentativa, codigo in (("tarso", 0), ("serao", 121), ("termo", 18)):
        candidatas = _estado(motor, tentativa, codigo)
        if len(candidatas) < 2:
            continue
        melhor = motor.entropias(candidatas).max()
        antes = motor_padrao.entropias(_estado(motor_padrao, tentativa, codigo)).max()
        assert melhor >= antes - 1e-12


def test_no_empate_a_candidata_ganha_da_sonda(motor):
    """Uma conjugação só é escolhida se separar ESTRITAMENTE melhor (§4.5)."""
    random.seed(31)
    n = len(motor.lexico)
    for tentativa, codigo in (("tarso", 0), ("cabra", 1), ("termo", 18)):
        candidatas = _estado(motor, tentativa, codigo)
        if len(candidatas) < 3:
            continue
        entropias = motor.entropias(candidatas)
        escolhido = int(motor.ordenar_por_entropia(entropias, candidatas)[0])
        if escolhido >= n:
            assert entropias[escolhido] > entropias[:n].max()


def test_endgame_nunca_sugere_uma_sonda(motor):
    """Com o tabuleiro no fim, chutar o que não pode ser a resposta é desperdício."""
    candidatas = _estado(motor)[:4]
    for tentativa in (motor.n_max_tentativas - 1, motor.n_max_tentativas):
        sugestao = motor.escolher(candidatas, tentativa)
        assert sugestao.e_candidata
        assert sugestao.indice in set(candidatas.tolist())


def test_sugestao_de_sonda_vem_rotulada_como_nao_candidata(motor):
    """A CLI mostra 'não é candidata'; o índice fora do prefixo tem que dizer isso."""
    candidatas = _estado(motor)
    sugestao = motor.escolher(candidatas, tentativa=2)
    assert sugestao.palavra == motor.lexico.mostrar(sugestao.indice)
    assert sugestao.e_candidata == motor.lexico.e_candidata(sugestao.indice)
