"""Testes do nível 3: valor exato, monotonia da busca e integração com o nível 2.

O teste central é o cruzamento com uma referência independente: num léxico
minúsculo, a busca com beam total e profundidade sobrando tem que dar exatamente
o mesmo número que uma enumeração exaustiva escrita à parte, em Python puro.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from termo import nivel3
from termo.entropia import Motor
from termo.feedback import calcular_feedback, padrao_para_codigo
from termo.lexico import Lexico, calcular_prior
from termo.matriz import construir_matriz
from termo.nivel3 import MotorNivel3


# --------------------------------------------------------------- léxico mínimo


def _mini_lexico(palavras: list[str], temperatura: float) -> Lexico:
    """Léxico artificial de poucas palavras, com ICF crescente (0, 1, 2, ...).

    O nível 3 não sabe de onde vem o prior, então um ICF sintético serve — e com
    T=inf ele vira uniforme, que é o caso em que a referência exaustiva (escrita
    sem pesos) é comparável.
    """
    icf = np.arange(len(palavras), dtype=np.float64)
    return Lexico(
        palavras=palavras,
        exibicao=palavras,
        icf=icf,
        ausentes_no_icf=[],
        temperatura=temperatura,
        prior=calcular_prior(icf, temperatura),
        indice={p: i for i, p in enumerate(palavras)},
    )


@pytest.fixture(scope="module")
def palavras_mini():
    random.seed(29)
    return sorted(random.sample(Lexico.carregar().palavras, 14))


@pytest.fixture(scope="module")
def mini_uniforme(palavras_mini):
    """Motor de nível 3 exaustivo (beam total, profundidade sobrando) e uniforme."""
    lexico = _mini_lexico(palavras_mini, math.inf)
    motor = Motor(lexico, matriz=construir_matriz(palavras_mini, verboso=False))
    return MotorNivel3(motor, beam=len(palavras_mini), profundidade=8)


def _valor_exato(
    candidatas: tuple[str, ...], pool: tuple[str, ...], rodadas: int, memo
) -> float:
    """Referência independente: E[tentativas] ótimo, pesos uniformes, exaustivo.

    Reimplementa o objetivo do zero — feedback em string, listas em Python, sem
    matriz, sem beam, sem poda. Se as duas contas concordarem, o erro teria que
    estar nas duas ao mesmo tempo.
    """
    if rodadas <= 0:
        return nivel3.PENALIDADE_DERROTA
    n = len(candidatas)
    if n == 1:
        return 1.0
    if (candidatas, rodadas) in memo:
        return memo[(candidatas, rodadas)]
    melhor = math.inf
    for palpite in pool:
        baldes = defaultdict(list)
        for candidata in candidatas:
            baldes[calcular_feedback(palpite, candidata)].append(candidata)
        if len(baldes) == 1:
            continue  # não separa nada: jogada inútil
        total = 0.0
        for padrao, sub in baldes.items():
            if padrao == "GGGGG":
                continue  # acertou agora, nenhuma tentativa adicional
            total += len(sub) * _valor_exato(tuple(sub), pool, rodadas - 1, memo)
        melhor = min(melhor, 1.0 + total / n)
    memo[(candidatas, rodadas)] = melhor
    return melhor


def test_busca_exaustiva_bate_com_referencia_independente(mini_uniforme, palavras_mini):
    pool = tuple(palavras_mini)
    memo: dict = {}
    todas = mini_uniforme.todas_candidatas()
    assert mini_uniforme.valor(todas) == pytest.approx(
        _valor_exato(pool, pool, 6, memo), abs=1e-9
    )

    # E também em sub-conjuntos, que são os estados que aparecem de verdade.
    random.seed(31)
    for tamanho in (3, 4, 5, 7):
        sorteadas = sorted(random.sample(range(len(pool)), tamanho))
        indices = np.array(sorteadas, dtype=np.int32)
        subconjunto = tuple(pool[i] for i in indices)
        assert mini_uniforme.valor(indices) == pytest.approx(
            _valor_exato(subconjunto, pool, 6, memo), abs=1e-9
        ), subconjunto


def test_orcamento_de_rodadas_muda_o_valor(mini_uniforme, palavras_mini):
    """Com poucas rodadas o valor sobe: parte da massa vira derrota."""
    pool = tuple(palavras_mini)
    memo: dict = {}
    todas = mini_uniforme.todas_candidatas()
    for rodadas in (1, 2, 3):
        assert mini_uniforme.valor(todas, rodadas=rodadas) == pytest.approx(
            _valor_exato(pool, pool, rodadas, memo), abs=1e-9
        ), rodadas
    # mais orçamento nunca piora, e com 1 rodada o valor tem que refletir a derrota
    valores = [mini_uniforme.valor(todas, rodadas=r) for r in range(1, 7)]
    assert valores == sorted(valores, reverse=True)
    assert valores[0] > nivel3.PENALIDADE_DERROTA - 1


# ------------------------------------------------------------------- casos base


def test_uma_candidata_custa_uma_tentativa(mini_uniforme):
    assert mini_uniforme.valor(np.array([3], dtype=np.int32)) == 1.0


def test_duas_candidatas_custam_um_mais_o_peso_da_outra(palavras_mini):
    """Chutar a mais provável: acerta agora com p1, na seguinte com p2."""
    matriz = construir_matriz(palavras_mini, verboso=False)
    lexico = _mini_lexico(palavras_mini, 1.0)
    motor3 = MotorNivel3(Motor(lexico, matriz=matriz))
    par = np.array([0, 1], dtype=np.int32)
    pesos = lexico.prior[par] / lexico.prior[par].sum()
    assert motor3.valor(par) == pytest.approx(1.0 + pesos.min())
    # uniforme é o caso clássico: 50% de chance em cada uma
    uniforme = MotorNivel3(Motor(_mini_lexico(palavras_mini, math.inf), matriz=matriz))
    assert uniforme.valor(par) == pytest.approx(1.5)


def test_prior_degenerado_cai_no_uniforme(palavras_mini):
    """T minúsculo zera o softmax fora do topo; o valor não pode virar lixo.

    Com todos os pesos em zero as probabilidades tinham que somar 1 vindas da
    contagem, não do prior — senão o custo de qualquer jogada seria 1,0 e a busca
    escolheria no ruído.
    """
    lexico = _mini_lexico(palavras_mini, 0.01)
    motor3 = MotorNivel3(
        Motor(lexico, matriz=construir_matriz(palavras_mini, verboso=False)),
        beam=len(palavras_mini),
        profundidade=8,
    )
    # ICF sintético 0,1,2,... com T=0.01: da 8ª palavra em diante exp(-800)
    # simplesmente não existe em float64 e o peso é exatamente zero.
    fora_do_topo = np.arange(8, len(palavras_mini), dtype=np.int32)
    assert lexico.prior[fora_do_topo].sum() == 0.0  # o cenário existe de verdade

    pool = tuple(palavras_mini)
    esperado = _valor_exato(tuple(pool[i] for i in fora_do_topo), pool, 6, {})
    assert motor3.valor(fora_do_topo) == pytest.approx(esperado, abs=1e-9)


def test_conjunto_vazio_e_erro(mini_uniforme):
    with pytest.raises(ValueError):
        mini_uniforme.valor(np.array([], dtype=np.int32))
    with pytest.raises(ValueError):
        mini_uniforme.escolher(np.array([], dtype=np.int32))


def test_orcamento_grande_nao_estoura_a_chave_do_memo(palavras_mini):
    """A chave da memoização não pode impor um teto escondido às rodadas.

    `rodadas` vem de `n_max_tentativas`, que é parâmetro do motor — quem montasse
    um jogo de mais de 255 rodadas levava um "bytes must be in range(0, 256)"
    vindo de dentro do cache, sem relação nenhuma com o que pediu.
    """
    lexico = _mini_lexico(palavras_mini, math.inf)
    matriz = construir_matriz(palavras_mini, verboso=False)
    motor3 = MotorNivel3(Motor(lexico, matriz=matriz), beam=4, profundidade=1)
    candidatas = np.arange(4, dtype=np.int32)

    referencia = motor3.valor(candidatas, rodadas=6)
    for rodadas in (255, 256, 1000):
        # Orçamento grande não muda nada: com 4 candidatas 6 rodadas já sobram.
        assert motor3.valor(candidatas, rodadas=rodadas) == pytest.approx(referencia)

    folgado = MotorNivel3(
        Motor(lexico, matriz=matriz, n_max_tentativas=300), beam=4, profundidade=1
    )
    assert folgado.escolher(candidatas, tentativa=1).palavra in palavras_mini


def test_configuracao_invalida_e_erro(mini_uniforme):
    with pytest.raises(ValueError):
        MotorNivel3(mini_uniforme.motor, beam=0)
    with pytest.raises(ValueError):
        MotorNivel3(mini_uniforme.motor, profundidade=-1)


# ------------------------------------------------------ monotonia e nível 2


@pytest.fixture(scope="module")
def motor():
    return Motor(Lexico.carregar(1.0))


@pytest.fixture(scope="module")
def estado(motor):
    """Estado real de meio de jogo: 'tarso' com o 'r' verde deixa 51 candidatas.

    Grande o bastante para os dois níveis discordarem, pequeno o bastante para a
    busca rodar em segundos.
    """
    return motor.filtrar(motor.todas_candidatas(), "tarso", padrao_para_codigo("BBGBB"))


def test_busca_nunca_fica_acima_da_politica_gulosa(motor, estado):
    """Cada nível de profundidade só pode baixar o valor: o beam contém o guloso."""
    motor3 = MotorNivel3(motor, beam=6, profundidade=0)
    guloso = motor3.valor_guloso(estado)
    valores = [motor3.valor(estado, profundidade=p) for p in range(3)]
    assert valores[0] <= guloso + 1e-9
    for anterior, seguinte in zip(valores, valores[1:]):
        assert seguinte <= anterior + 1e-9


def test_o_nivel_3_bate_a_entropia_em_estado_real(motor, estado):
    """O ponto do nível 3: maximizar bits não minimiza tentativas (§4.1).

    Com o prior de T=1 a diferença não é sutil — a jogada de maior entropia perde
    para uma candidata provável, que ganha o jogo com probabilidade não-nula.
    """
    motor3 = MotorNivel3(motor, beam=8, profundidade=1)
    escolha3 = motor3.escolher(estado, tentativa=2)
    escolha2 = motor.escolher(estado, tentativa=2)
    assert escolha3.valor_esperado < motor3.valor_guloso(estado)
    assert escolha3.palavra != escolha2.palavra
    # a entropia do escolhido é menor que a do nível 2 — é o trade-off explícito
    assert escolha3.entropia < escolha2.entropia


def test_alternativas_saem_ordenadas_por_valor_esperado(motor, estado):
    sugestao = MotorNivel3(motor, beam=6, profundidade=0).escolher(estado, tentativa=2)
    valores = [sugestao.valor_esperado]
    valores += [valor for _, valor, _ in sugestao.alternativas]
    assert valores == sorted(valores)
    assert all(math.isfinite(valor) for valor in valores)


def test_nivel_2_nao_preenche_valor_esperado(motor, estado):
    """O campo novo em `Sugestao` é do nível 3; o nível 2 segue reportando bits."""
    assert motor.escolher(estado, tentativa=2).valor_esperado is None


# --------------------------------------------------------------------- endgame


def test_endgame_e_delegado_ao_nivel_2(motor, estado):
    """Na última tentativa não se queima jogada, por mais que ela separe bem."""
    motor3 = MotorNivel3(motor, beam=4, profundidade=0)
    assert motor3.escolher(estado, tentativa=6).e_candidata
    tres = estado[:3]
    assert motor3.escolher(tres, tentativa=5).e_candidata


def test_uma_e_duas_candidatas_seguem_o_nivel_2(motor):
    par = np.array(
        [motor.lexico.indice_de("termo"), motor.lexico.indice_de("ababá")],
        dtype=np.int32,
    )
    motor3 = MotorNivel3(motor, beam=4, profundidade=0)
    sugestao = motor3.escolher(par, tentativa=2)
    assert sugestao.palavra == "termo"
    assert sugestao.valor_esperado == pytest.approx(motor3.valor(par))
    unica = motor3.escolher(par[:1], tentativa=2)
    assert unica.valor_esperado == 1.0


# ---------------------------------------------------------------------- cache


def test_abertura_vai_e_volta_do_cache(palavras_mini, tmp_path, monkeypatch):
    """Mesmo teste de cache do nível 2, mas com chave em (T, beam, profundidade)."""
    monkeypatch.setattr(nivel3, "ARQ_ABERTURA", tmp_path / "aberturas_nivel3.json")
    lexico = _mini_lexico(palavras_mini, 1.0)
    matriz = construir_matriz(palavras_mini, verboso=False)
    motor3 = MotorNivel3(Motor(lexico, matriz=matriz), beam=5, profundidade=1)

    assert not motor3.abertura_em_cache()
    primeira = motor3.abertura()
    assert motor3.abertura_em_cache()
    segunda = motor3.abertura()
    assert (segunda.palavra, segunda.valor_esperado) == (
        primeira.palavra, primeira.valor_esperado
    )
    assert "em cache" in segunda.motivo

    # Outra configuração é outra chave: não herda a resposta da primeira.
    outro = MotorNivel3(Motor(lexico, matriz=matriz), beam=5, profundidade=2)
    assert not outro.abertura_em_cache()


def test_escolha_com_cache_devolve_o_mesmo(motor, estado):
    motor3 = MotorNivel3(motor, beam=4, profundidade=0)
    direta = motor3.escolher(estado, tentativa=3, n_alternativas=0)
    for _ in range(2):
        assert motor3.escolher_com_cache(estado, 3).indice == direta.indice


# --------------------------------------------------------------- jogo completo


def _partida(motor, motor3, secreta: int, abertura: int) -> int | None:
    """Joga do jeito da CLI: abertura do nível 2, nível 3 daí em diante.

    A abertura do nível 3 custa minutos e está coberta pelo teste de cache; o que
    importa aqui é o resto da partida.
    """
    candidatas = motor.todas_candidatas()
    tentativa = abertura
    for rodada in range(1, 7):
        if tentativa == secreta:
            return rodada
        codigo = motor.matriz[tentativa, secreta]
        candidatas = candidatas[motor.matriz[tentativa, candidatas] == codigo]
        assert len(candidatas) > 0, motor.lexico.palavras[secreta]
        tentativa = motor3.escolher_com_cache(candidatas, rodada + 1).indice
    return None


def test_partidas_completas_terminam(motor):
    """O nível 3 resolve a bateria realista em 6 tentativas, sem zerar C.

    A amostra sai das palavras mais comuns (a bateria do §5.3) e não do léxico
    inteiro, e isso é parte do teste: o objetivo do nível 3 é ponderado pelo
    prior, então ele SACRIFICA palavras raríssimas de propósito — com T=1 a
    candidata mais provável de um conjunto carrega uns dois terços da massa, e
    gastar uma jogada separando o resto piora o valor esperado. "criva" (ICF 19,
    contra 8 de "prima") é uma das que ele perde; o Termo nunca a sortearia.
    """
    motor3 = MotorNivel3(motor, beam=5, profundidade=0)
    abertura = motor.abertura().indice

    random.seed(19)
    for secreta in random.sample(motor.lexico.mais_comuns(1500).tolist(), 8):
        assert _partida(motor, motor3, secreta, abertura) is not None, (
            motor.lexico.palavras[secreta]
        )
