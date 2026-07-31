#!/usr/bin/env python
"""Benchmark das estratégias (seção 5 da especificação).

Pergunta central: a estratégia de entropia compensa o custo computacional, ou
uma heurística simples chega perto?

    python benchmark.py                       # bateria realista, 4 estratégias
    python benchmark.py --bateria completo    # stress test: léxico inteiro
    python benchmark.py --varredura-t         # tentativas médias em função de T
    python benchmark.py --n 300               # amostra menor, para iterar rápido
    python benchmark.py --nivel3              # head-to-head nível 2 vs nível 3
    python benchmark.py --serao               # por que os fóruns apontam `serão`
    python benchmark.py --catalogo            # melhor abertura de cada cenário (lento)

Resultados vão para `resultados/*.json`; os gráficos saem de `analise.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from termo.entropia import N_MAX_TENTATIVAS, Motor
from termo.estrategias import (
    AberturaFixa,
    Aleatoria,
    Entropia,
    Estrategia,
    FrequenciaDeLetras,
    MaisProvavel,
    construir_nivel3,
)
from termo.lexico import Lexico, calcular_prior
from termo.nivel3 import BEAM, PROFUNDIDADE, MotorNivel3

DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"

# Bateria realista: as N palavras de menor ICF. O Termo nunca vai sortear
# "leruê", então medir contra o léxico inteiro puniria todas as estratégias
# e inflaria artificialmente a média (seção 5.3). A faixa da v1.1 é ~800-1.500,
# reduzida porque o léxico encolheu de 8.996 para 6.046.
TAMANHO_REALISTA = 1500

# O nível 3 custa uma árvore de decisão por jogada, não uma varredura de entropia.
# Amostra menor por padrão; o cache de escolhas amortiza o resto (§4.1, nível 3).
TAMANHO_NIVEL3 = 300

TEMPERATURAS_VARREDURA = [0.5, 1.0, 2.0, 5.0, 10.0, math.inf]

# Experimento do `serão`: a abertura que os fóruns de Termo apontam como ótima e
# que não é a melhor em nível nenhum aqui. Os cortes são os tamanhos plausíveis
# para a "lista de respostas" que o consenso supõe sem declarar; o mundo restrito
# é onde as aberturas se enfrentam jogando.
PALAVRA_FORUNS = "serão"
CORTES_SERAO = [200, 300, 500, 1000, 1500]
MUNDO_SERAO = 300
N_TOPO_SERAO = 12          # fundo do ranking mostrado: o suficiente para incluir `serão`
N_ABERTURAS_SERAO = 3      # quantas rivais jogam contra ele no mundo restrito
TEMPERATURAS_SERAO = [math.inf, 2.0, 1.0, 0.5, 0.3, 0.1]

# Matriz de arrependimento: quanto cada abertura perde para a melhor DAQUELE mundo.
# Ninguém sabe em qual dos três está jogando, então a coluna que importa é o pior
# caso — e uma abertura medíocre em todo lugar pode ganhar de uma ótima num só.
MUNDOS_ARREPENDIMENTO = [300, 1500, None]  # None = léxico inteiro

# Grade (N, T): os dois botões de uma vez. As bordas reproduzem tudo que já se
# sabe — (todas, T=1) é o padrão do projeto, (300, T→∞) é a hipótese do fórum.
CORTES_GRADE = [100, 200, 300, 500, 1000, 1500, 3000, None]
TEMPERATURAS_GRADE = [math.inf, 5.0, 2.0, 1.0, 0.5, 0.3, 0.1]

# Catálogo de ótimos: a melhor abertura de CADA cenário, pelos dois objetivos.
# Roda separado do `--serao` porque a última linha do mundo fechado é a busca
# completa do nível 3 sobre as 6.046 — sozinha, ~11 min.
CORTES_CATALOGO = [100, 200, 300, 500, 1000, 1500, 3000, None]
CORTES_CATALOGO_ABERTO = [300, 1500, None]
TEMPERATURAS_CATALOGO = [math.inf, 2.0, 1.0, 0.5, 0.3, 0.1]


# ------------------------------------------------------------------ simulação


def jogar(
    motor: Motor, estrategia: Estrategia, secreta: int, n_max: int = N_MAX_TENTATIVAS,
    candidatas_iniciais: np.ndarray | None = None,
) -> tuple[int | None, list[int]]:
    """Joga uma partida completa. Devolve (nº de tentativas ou None, jogadas).

    `candidatas_iniciais` restringe o que o solver admite como RESPOSTA sem tocar
    no que ele pode digitar (o padrão é o léxico inteiro nos dois papéis).
    """
    estrategia.reiniciar()
    candidatas = (
        motor.todas_candidatas() if candidatas_iniciais is None else candidatas_iniciais
    )
    jogadas: list[int] = []

    for rodada in range(1, n_max + 1):
        tentativa = estrategia.escolher(candidatas, rodada)
        jogadas.append(tentativa)
        if tentativa == secreta:
            return rodada, jogadas
        codigo = motor.matriz[tentativa, secreta]
        candidatas = candidatas[motor.matriz[tentativa, candidatas] == codigo]
        if len(candidatas) == 0:  # não deve acontecer: a secreta está no léxico
            raise RuntimeError(f"candidatas zeraram jogando contra {secreta}")

    return None, jogadas


# ------------------------------------------------------------------- métricas


@dataclass
class Resultado:
    estrategia: str
    bateria: str
    n_jogos: int
    media_tentativas: float          # entre as partidas vencidas
    media_penalizada: float          # derrota conta como n_max + 1
    taxa_vitoria: float
    distribuicao: dict[int, int]     # tentativas -> nº de jogos
    derrotas: list[str]              # palavras que a estratégia não resolveu
    piores_casos: list[str]          # palavras resolvidas só na última tentativa
    segundos: float
    segundos_por_jogo: float
    temperatura: float | None = None
    extras: dict = field(default_factory=dict)


def avaliar(
    motor: Motor,
    estrategia: Estrategia,
    secretas: np.ndarray,
    nome_bateria: str,
    n_max: int = N_MAX_TENTATIVAS,
    verboso: bool = True,
    candidatas_iniciais: np.ndarray | None = None,
) -> Resultado:
    palavras = motor.lexico.exibicao  # relatórios saem na forma acentuada
    tentativas: list[int] = []
    derrotas: list[str] = []
    piores: list[str] = []
    inicio = time.perf_counter()

    for posicao, secreta in enumerate(secretas):
        n, _ = jogar(motor, estrategia, int(secreta), n_max, candidatas_iniciais)
        if n is None:
            derrotas.append(palavras[secreta])
        else:
            tentativas.append(n)
            if n == n_max:
                piores.append(palavras[secreta])
        if verboso and (posicao + 1) % 250 == 0:
            decorrido = time.perf_counter() - inicio
            print(
                f"    {posicao + 1}/{len(secretas)}  "
                f"média={statistics.fmean(tentativas or [0]):.3f}  "
                f"{decorrido:.0f}s",
                flush=True,
            )

    segundos = time.perf_counter() - inicio
    n_jogos = len(secretas)
    penalizadas = tentativas + [n_max + 1] * len(derrotas)
    return Resultado(
        estrategia=estrategia.nome,
        bateria=nome_bateria,
        n_jogos=n_jogos,
        media_tentativas=statistics.fmean(tentativas) if tentativas else float("nan"),
        media_penalizada=statistics.fmean(penalizadas),
        taxa_vitoria=len(tentativas) / n_jogos,
        distribuicao=dict(sorted(Counter(tentativas).items())),
        derrotas=derrotas[:50],
        piores_casos=piores[:20],
        segundos=segundos,
        segundos_por_jogo=segundos / n_jogos,
        temperatura=motor.lexico.temperatura,
    )


def formatar_tabela(resultados: list[Resultado], n_max: int = N_MAX_TENTATIVAS) -> str:
    colunas = [f"{n}" for n in range(1, n_max + 1)]
    cabecalho = (
        f"{'estratégia':<20} {'média':>7} {'penal.':>7} {'vitória':>8} "
        + " ".join(f"{c:>5}" for c in colunas)
        + f" {'s/jogo':>8}"
    )
    linhas = [cabecalho, "-" * len(cabecalho)]
    for r in resultados:
        distribuicao = " ".join(
            f"{r.distribuicao.get(n, 0):>5}" for n in range(1, n_max + 1)
        )
        linhas.append(
            f"{r.estrategia:<20} {r.media_tentativas:>7.3f} {r.media_penalizada:>7.3f} "
            f"{r.taxa_vitoria:>7.1%} {distribuicao} {r.segundos_por_jogo:>8.4f}"
        )
    return "\n".join(linhas)


# -------------------------------------------------------------------- baterias


def conjunto_secretas(lexico: Lexico, bateria: str, n: int | None) -> np.ndarray:
    if bateria == "realista":
        return np.sort(lexico.mais_comuns(n or TAMANHO_REALISTA)).astype(np.int32)
    if bateria == "completo":
        if n is None or n >= len(lexico):
            return np.arange(len(lexico), dtype=np.int32)
        # Amostra equiespaçada: preserva o perfil do léxico inteiro, ao contrário
        # de cortar pelas mais comuns (isso seria a bateria realista de novo).
        return np.linspace(0, len(lexico) - 1, n).astype(np.int32)
    raise ValueError(f"bateria desconhecida: {bateria}")


def mundo_restrito(
    lexico: Lexico, matriz: np.ndarray, n: int, temperatura: float = math.inf
) -> tuple[Lexico, np.ndarray]:
    """O mundo que o consenso dos fóruns supõe: só as `n` palavras mais comuns.

    Respostas E palpites saem dessa lista curta, e dentro dela o peso vem de `T`
    (o padrão, T→∞, é uniforme). São dois botões independentes, e é essa a
    correção que o dial sozinho não dá: `n` é um corte DURO na lista de respostas,
    `T` é um peso CONTÍNUO sobre quem sobrou. O prior é renormalizado dentro do
    corte, então cada mundo é um jogo completo e não uma fatia de outro.

    Os dois extremos já são conhecidos: `(n=todas, T=1)` é a configuração padrão
    do projeto e `(n=300, T→∞)` é a hipótese do fórum.
    """
    if n >= len(lexico):
        # Sem corte não há o que fatiar, e a matriz inteira são 37 MB por cópia.
        return lexico.com_temperatura(temperatura), matriz

    indices = np.sort(lexico.mais_comuns(n))
    palavras = [lexico.palavras[i] for i in indices]
    icf = lexico.icf[indices]
    restrito = Lexico(
        palavras=palavras,
        exibicao=[lexico.exibicao[i] for i in indices],
        icf=icf,
        ausentes_no_icf=[],
        temperatura=temperatura,
        prior=calcular_prior(icf, temperatura),
        indice={p: i for i, p in enumerate(palavras)},
    )
    return restrito, matriz[np.ix_(indices, indices)].copy()


def mundo_aberto(lexico: Lexico, n: int) -> tuple[Lexico, np.ndarray]:
    """Só as `n` mais comuns podem ser a RESPOSTA; o léxico inteiro continua digitável.

    É a diferença que a matriz de arrependimento exige. `tarso` não está entre as
    300 mais comuns e mesmo assim é uma jogada perfeitamente legal no term.ooo —
    num mundo fechado (`mundo_restrito`) ela simplesmente não existe, e aberturas
    de mundos diferentes não teriam como se enfrentar.

    O que muda de mundo para mundo é a crença sobre o sorteio, não o teclado:
    o corte entra no prior (uniforme sobre as `n`, zero no resto) e no conjunto
    inicial de candidatas, nunca no espaço de palpites.
    """
    candidatas = np.sort(lexico.mais_comuns(n)).astype(np.int32)
    prior = np.zeros(len(lexico), dtype=np.float64)
    prior[candidatas] = 1.0 / n
    return replace(lexico, temperatura=math.inf, prior=prior), candidatas


def _posicao(ordem: np.ndarray, indice: int) -> int:
    """Colocação de `indice` num ranking já ordenado (1 = primeiro)."""
    return int(np.where(ordem == indice)[0][0]) + 1


def salvar(resultados: list[Resultado], nome: str) -> Path:
    return salvar_bruto([asdict(r) for r in resultados], nome)


def salvar_bruto(objeto, nome: str) -> Path:
    DIR_RESULTADOS.mkdir(exist_ok=True)
    caminho = DIR_RESULTADOS / nome
    caminho.write_text(
        json.dumps(objeto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return caminho


# ------------------------------------------------------------------ execuções


def rodar_comparacao(bateria: str, n: int | None, temperatura: float,
                     semente: int) -> list[Resultado]:
    lexico = Lexico.carregar(temperatura)
    motor = Motor(lexico)
    secretas = conjunto_secretas(lexico, bateria, n)
    print(f"\nbateria '{bateria}': {len(secretas)} palavras secretas, T={temperatura}")

    estrategias = [
        Aleatoria(semente),
        FrequenciaDeLetras(lexico),
        MaisProvavel(lexico),
        Entropia(motor),
    ]
    resultados = []
    for estrategia in estrategias:
        print(f"\n  {estrategia.nome} ...")
        resultados.append(avaliar(motor, estrategia, secretas, bateria))
    return resultados


def rodar_nivel3(bateria: str, n: int | None, temperatura: float,
                 beam: int, profundidade: int) -> list[Resultado]:
    """Head-to-head nível 2 vs nível 3 (§4.1) sobre exatamente as mesmas secretas.

    Fica num arquivo próprio em vez de entrar na tabela das quatro estratégias:
    a pergunta aqui é outra — não "entropia compensa?", mas "o proxy da entropia
    custa quantas tentativas em relação ao objetivo real?".
    """
    lexico = Lexico.carregar(temperatura)
    motor = Motor(lexico)
    secretas = conjunto_secretas(lexico, bateria, n or TAMANHO_NIVEL3)
    print(f"\nbateria '{bateria}': {len(secretas)} palavras secretas, T={temperatura}, "
          f"beam={beam}, profundidade={profundidade}")

    resultados = []
    for estrategia in (Entropia(motor), construir_nivel3(motor, beam, profundidade)):
        print(f"\n  {estrategia.nome} ...")
        resultados.append(avaliar(motor, estrategia, secretas, bateria))
    resultados[-1].extras = {"beam": beam, "profundidade": profundidade}

    dois, tres = resultados
    delta = tres.media_penalizada - dois.media_penalizada  # negativo = nível 3 ganha
    print(f"\n  nível 3 - nível 2: {delta:+.4f} tentativas (média penalizada), "
          f"{tres.segundos / max(dois.segundos, 1e-9):.0f}x o custo de CPU")
    return resultados


def rodar_serao(n: int | None, temperatura: float, beam: int, profundidade: int
                ) -> tuple[list[Resultado], dict]:
    """Por que os fóruns apontam `serão` e o solver não.

    O consenso não é bobo: ele otimiza um problema vizinho. Este experimento
    isola as três diferenças, e cada uma sozinha já tira `serão` do topo.

      1. o léxico de palpites  as dez aberturas à frente dele são obscuras;
                               `serão` é a 83ª palavra mais comum do léxico
      2. a lista de respostas  cortada nas ~300 mais comuns, ele passa a PRIMEIRO
                               em entropia — a recomendação é reproduzível
      3. o objetivo            e ainda assim perde em tentativas, mesmo ali

    A terceira é a correção do 3B1B (§4.1) aplicada ao consenso popular. As duas
    primeiras são ranking; a terceira só se decide jogando, então as partidas
    voltam como `Resultado` e o resto vai no dicionário de extras.

    Só que "qual mundo é o certo" não é uma pergunta respondida: a lista de
    respostas do Termo é uma hipótese nossa. As três medições finais tratam a
    escolha do mundo como parte do problema em vez de premissa — arrependimento de
    pior caso entre mundos, a grade (N, T) inteira, e a fronteira frequência ×
    informação de onde tudo isso vem.

    `n` vale só para a bateria realista do fim. Os cortes e o mundo restrito são
    a definição do experimento, não uma amostragem: mexer neles é mudar a
    pergunta, e é para isso que servem `CORTES_SERAO` e `MUNDO_SERAO`.
    """
    lexico = Lexico.carregar(temperatura)
    motor = Motor(lexico)
    matriz = motor.matriz
    alvo = lexico.indice_de(PALAVRA_FORUNS)
    ordem_icf = np.argsort(lexico.icf, kind="stable")
    frequencia = {int(j): k + 1 for k, j in enumerate(ordem_icf)}
    partidas: list[Resultado] = []

    # ---------------------------------------------- 1. o léxico de palpites
    print(f"\n[1/6] ranking de entropia do léxico completo (T={temperatura:g}) ...")
    entropias = motor.entropias(motor.todas_candidatas())
    ordem = np.argsort(-entropias, kind="stable")
    topo = [
        {
            "palavra": lexico.mostrar(int(j)),
            "entropia": float(entropias[j]),
            "frequencia": frequencia[int(j)],
        }
        for j in ordem[:N_TOPO_SERAO]
    ]
    for posicao, linha in enumerate(topo, 1):
        print(f"  {posicao:2d}. {linha['palavra']:8s} H={linha['entropia']:.3f}  "
              f"frequência {linha['frequencia']:>5}/{len(lexico)}")
    print(f"  '{PALAVRA_FORUNS}': {_posicao(ordem, alvo)}º em entropia, "
          f"{frequencia[alvo]}º em frequência")

    # ------------------------------------------- 2a. a lista de respostas
    print("\n[2/6] cortando a lista de respostas nas N mais comuns ...")
    uniforme = Motor(lexico.com_temperatura(math.inf), matriz=matriz)
    cortes = []
    for corte in CORTES_SERAO:
        candidatas = np.sort(lexico.mais_comuns(corte)).astype(np.int32)
        curtas = set(candidatas.tolist())
        ordem_n = np.argsort(-uniforme.entropias(candidatas), kind="stable")
        # Duas leituras: o fórum restringe as RESPOSTAS, mas na prática também os
        # PALPITES — ninguém digita `tirão`. A segunda coluna é o cenário completo.
        dentro = [int(j) for j in ordem_n if int(j) in curtas]
        cortes.append({
            "n": corte,
            "melhor_no_lexico": lexico.mostrar(int(ordem_n[0])),
            "posicao_no_lexico": _posicao(ordem_n, alvo),
            "melhor_na_lista": lexico.mostrar(dentro[0]),
            "posicao_na_lista": dentro.index(alvo) + 1,
        })
        c = cortes[-1]
        print(f"  N={corte:<5} palpites=léxico: {c['melhor_no_lexico']:8s} "
              f"('{PALAVRA_FORUNS}' {c['posicao_no_lexico']:>3}º)   "
              f"palpites=as próprias N: {c['melhor_na_lista']:8s} "
              f"('{PALAVRA_FORUNS}' {c['posicao_na_lista']:>3}º)")

    # ------------------------------- 2b. o dial T não reproduz o corte duro
    print("\n[3/6] o mesmo efeito, tentado pelo dial T (prior contínuo) ...")
    varredura = []
    for t in TEMPERATURAS_SERAO:
        m = Motor(lexico.com_temperatura(t), matriz=matriz)
        ordem_t = np.argsort(-m.entropias(m.todas_candidatas()), kind="stable")
        varredura.append({
            "t": t,
            "melhor": lexico.mostrar(int(ordem_t[0])),
            "posicao": _posicao(ordem_t, alvo),
        })
        print(f"  T={t:<5g} melhor={varredura[-1]['melhor']:8s} "
              f"'{PALAVRA_FORUNS}' {varredura[-1]['posicao']}º")

    # ------------------------------------------------------- 3. o objetivo
    print(f"\n[4/6] mundo restrito de {MUNDO_SERAO} palavras: bits contra tentativas ...")
    curto, matriz_curta = mundo_restrito(lexico, matriz, MUNDO_SERAO)
    motor_curto = Motor(curto, matriz=matriz_curta)
    alvo_curto = curto.indice_de(PALAVRA_FORUNS)
    ordem_curta = np.argsort(
        -motor_curto.entropias(motor_curto.todas_candidatas()), kind="stable"
    )
    print(f"  entropia: {[curto.mostrar(int(j)) for j in ordem_curta[:5]]}  "
          f"('{PALAVRA_FORUNS}' {_posicao(ordem_curta, alvo_curto)}º)")

    busca = MotorNivel3(motor_curto, beam, profundidade)
    escolha = busca.escolher(busca.todas_candidatas(), tentativa=1, n_alternativas=5)
    print(f"  nível 3: {escolha.palavra} (E={escolha.valor_esperado:.3f})  "
          "alternativas: "
          + ", ".join(f"{p} {v:.3f}" for p, v, _ in escolha.alternativas))

    # As rivais saem da própria busca, não de uma lista escrita à mão.
    rivais = dict.fromkeys([escolha.palavra] + [p for p, _, _ in escolha.alternativas])
    aberturas = [p for p in rivais if p != PALAVRA_FORUNS][:N_ABERTURAS_SERAO]
    aberturas.append(PALAVRA_FORUNS)
    for palavra in aberturas:
        # Motor novo por abertura: o cache de escolhas é indexado pelo conjunto de
        # candidatas, e aberturas diferentes levam a conjuntos diferentes.
        m = Motor(curto, matriz=matriz_curta)
        partidas.append(avaliar(m, AberturaFixa(m, palavra),
                                motor_curto.todas_candidatas(),
                                f"restrita-{MUNDO_SERAO}", verboso=False))

    secretas = conjunto_secretas(lexico, "realista", n)
    print(f"\n  e no jogo de verdade ({len(lexico)} possíveis, "
          f"{len(secretas)} secretas realistas) ...")
    for palavra in (motor.abertura().palavra, PALAVRA_FORUNS):
        m = Motor(lexico, matriz=matriz)
        partidas.append(
            avaliar(m, AberturaFixa(m, palavra), secretas, "realista", verboso=False)
        )

    # De onde a recomendação provavelmente saiu: S, E, R, A, O são as cinco letras
    # mais frequentes, e é isso que o nível 1 mede.
    nivel1 = {
        rotulo: _ranking_nivel1(lexico, candidatas, alvo)
        for rotulo, candidatas in (
            (f"{MUNDO_SERAO} mais comuns", np.sort(lexico.mais_comuns(MUNDO_SERAO))),
            ("léxico completo", motor.todas_candidatas()),
        )
    }
    for rotulo, dados in nivel1.items():
        print(f"  nível 1 sobre {rotulo}: {dados['topo']}  "
              f"('{PALAVRA_FORUNS}' {dados['posicao']}º)")

    # ------------------------------------- 4. e se não soubermos o mundo?
    print(f"\n[5/6] arrependimento: {len(MUNDOS_ARREPENDIMENTO)} mundos, "
          "as mesmas aberturas em cada um ...")
    # As disputantes saem das próprias medições: a abertura do nível 2, a do nível
    # 3 (só se já estiver em cache — recalculá-la são ~9 min), a dos fóruns, a que
    # o nível 3 elegeu no mundo curto e a campeã por entropia de cada mundo.
    rivais = [lexico.mostrar(int(ordem[0])), escolha.palavra, PALAVRA_FORUNS]
    real = MotorNivel3(motor, beam, profundidade)
    if real.abertura_em_cache():
        rivais.insert(1, real.abertura().palavra)
    for corte in MUNDOS_ARREPENDIMENTO:
        mundo, candidatas = mundo_aberto(lexico, corte or len(lexico))
        motor_mundo = Motor(mundo, matriz=matriz)
        melhor = np.argmax(motor_mundo.entropias(candidatas))
        rivais.append(lexico.mostrar(int(melhor)))
    aberturas = list(dict.fromkeys(rivais))
    print(f"  aberturas em disputa: {', '.join(aberturas)}")
    jogos, arrependimento = _arrependimento(lexico, matriz, aberturas)
    partidas += jogos

    print("\n[6/6] grade (N, T) e a fronteira frequência × informação ...")
    grade = _grade_n_t(lexico, matriz, PALAVRA_FORUNS)
    nuvem = _nuvem_h_icf(lexico, entropias, alvo)
    print(f"  fronteira de Pareto: {len(nuvem['fronteira'])} palavras, "
          f"'{PALAVRA_FORUNS}' "
          + ("está nela" if any(f[0] == nuvem["alvo"] for f in nuvem["fronteira"])
             else "não está nela"))

    return partidas, {
        "palavra": PALAVRA_FORUNS,
        "temperatura": temperatura,
        "posicao_no_lexico": _posicao(ordem, alvo),
        "posicao_em_frequencia": frequencia[alvo],
        "ranking_do_lexico": topo,
        "cortes": cortes,
        "varredura_t": varredura,
        "mundo_restrito": {
            "n": MUNDO_SERAO,
            "entropia": [curto.mostrar(int(j)) for j in ordem_curta[:5]],
            "posicao_por_entropia": _posicao(ordem_curta, alvo_curto),
            "nivel3": {
                "abertura": escolha.palavra,
                "valor_esperado": escolha.valor_esperado,
                "beam": beam,
                "profundidade": profundidade,
                "alternativas": [[p, v] for p, v, _ in escolha.alternativas],
            },
        },
        "nivel1": nivel1,
        "arrependimento": arrependimento,
        "grade": grade,
        "nuvem": nuvem,
    }


def _arrependimento(lexico: Lexico, matriz: np.ndarray, aberturas: list[str],
                    ) -> tuple[list[Resultado], dict]:
    """Cada abertura em cada mundo, e o que ela perde para a melhor dali.

    O veredito de uma linha só ("`serão` perde por 0,04") supõe que se sabe em que
    mundo se está. Ninguém sabe: a lista de respostas do Termo é uma hipótese, não
    um dado. Sob incerteza a pergunta muda de "qual é a melhor?" para "qual é a
    menos pior no seu pior mundo?" — que é minimax de arrependimento, e pode dar
    outra resposta, porque uma abertura ótima num mundo e péssima nos outros perde
    para uma que fica sempre perto.

    Mundos ABERTOS (`mundo_aberto`): o corte é sobre as respostas, e as seis
    aberturas continuam digitáveis nos três. Um mundo fechado não serviria aqui —
    `tarso` nem existiria no de 300.
    """
    medias: dict[str, dict[str, float]] = {}
    partidas: list[Resultado] = []
    for n in MUNDOS_ARREPENDIMENTO:
        mundo, candidatas = mundo_aberto(lexico, n or len(lexico))
        rotulo = f"N={len(candidatas)}"
        print(f"  mundo {rotulo} ({len(candidatas)} respostas possíveis, "
              f"{len(lexico)} palpites) ...")
        for palavra in aberturas:
            motor = Motor(mundo, matriz=matriz)
            resultado = avaliar(motor, AberturaFixa(motor, palavra), candidatas,
                                rotulo, verboso=False, candidatas_iniciais=candidatas)
            partidas.append(resultado)
            medias.setdefault(palavra, {})[rotulo] = resultado.media_penalizada
            print(f"    {palavra:8s} {resultado.media_penalizada:.4f}")

    rotulos = [f"N={n or len(lexico)}" for n in MUNDOS_ARREPENDIMENTO]
    melhor_do_mundo = {r: min(medias[p][r] for p in aberturas) for r in rotulos}
    arrependimento = {
        p: {r: medias[p][r] - melhor_do_mundo[r] for r in rotulos} for p in aberturas
    }
    pior_caso = {p: max(arrependimento[p].values()) for p in aberturas}
    vencedora = min(pior_caso, key=lambda p: pior_caso[p])
    print(f"\n  menor arrependimento de pior caso: {vencedora} "
          f"({pior_caso[vencedora]:.4f} tentativa)")
    return partidas, {
        "mundos": rotulos,
        "aberturas": aberturas,
        "medias": medias,
        "melhor_do_mundo": melhor_do_mundo,
        "arrependimento": arrependimento,
        "pior_caso": pior_caso,
        "vencedora_minimax": vencedora,
    }


def _grade_n_t(lexico: Lexico, matriz: np.ndarray, alvo_palavra: str) -> dict:
    """Melhor abertura em cada (N, T): o corte duro e o prior contínuo, juntos.

    Uma célula é um mundo completo de `mundo_restrito`, então a grade contém como
    casos particulares tudo que as duas varreduras de uma dimensão só mostraram —
    e a diagonal entre elas, que é onde o desacordo com os fóruns vive.

    O valor da célula é a distância em bits entre a melhor abertura dali e a
    palavra em questão. Zero significa "é ela a melhor". Bits são comparáveis
    entre células; colocação num ranking não é, porque o denominador muda com N.
    """
    celulas = []
    for n in CORTES_GRADE:
        tamanho = n or len(lexico)
        for t in TEMPERATURAS_GRADE:
            mundo, matriz_mundo = mundo_restrito(lexico, matriz, tamanho, t)
            motor = Motor(mundo, matriz=matriz_mundo)
            entropias = motor.entropias(motor.todas_candidatas())
            ordem = np.argsort(-entropias, kind="stable")
            alvo = mundo.indice_de(alvo_palavra)
            celulas.append({
                "n": tamanho,
                "t": t,
                "melhor": mundo.mostrar(int(ordem[0])),
                "entropia_melhor": float(entropias[ordem[0]]),
                "entropia_alvo": float(entropias[alvo]),
                "distancia": float(entropias[ordem[0]] - entropias[alvo]),
                "posicao": _posicao(ordem, alvo),
            })
        linha = [c for c in celulas if c["n"] == tamanho]
        print(f"  N={tamanho:<5} " + "  ".join(
            f"T={c['t']:g}:{c['melhor']}" for c in linha
        ))
    return {
        "cortes": [n or len(lexico) for n in CORTES_GRADE],
        "temperaturas": TEMPERATURAS_GRADE,
        "celulas": celulas,
    }


def _nuvem_h_icf(lexico: Lexico, entropias: np.ndarray, alvo: int,
                 topo: int = 10) -> dict:
    """Entropia contra frequência, palavra a palavra — e a fronteira entre as duas.

    O fenômeno inteiro é uma troca: as aberturas que mais informam são raras, e as
    comuns informam menos. A fronteira de Pareto (nenhuma palavra é ao mesmo tempo
    mais comum E mais informativa) é onde essa troca fica explícita, e é nela que
    `serão` está — o consenso dos fóruns não é um erro de cálculo, é um ponto
    diferente da mesma curva.
    """
    # ICF é ~ -log(frequência no corpus): menor = mais comum.
    ordem_icf = np.argsort(lexico.icf, kind="stable")
    fronteira, teto = [], -np.inf
    for i in ordem_icf:  # do mais comum ao mais raro
        if entropias[i] > teto:
            teto = float(entropias[i])
            fronteira.append(int(i))
    ordem_h = np.argsort(-entropias, kind="stable")
    return {
        "icf": [round(float(v), 3) for v in lexico.icf],
        "entropia": [round(float(v), 4) for v in entropias],
        "fronteira": [
            [lexico.mostrar(i), round(float(lexico.icf[i]), 3),
             round(float(entropias[i]), 4)]
            for i in fronteira
        ],
        "anotar": [
            [lexico.mostrar(int(i)), round(float(lexico.icf[i]), 3),
             round(float(entropias[i]), 4)]
            for i in list(ordem_h[:topo]) + [alvo]
        ],
        "alvo": lexico.mostrar(alvo),
        "cortes_icf": {
            str(n): round(float(lexico.icf[lexico.mais_comuns(n)].max()), 3)
            for n in CORTES_SERAO
        },
    }


def _otimos(motor: Motor, candidatas: np.ndarray, beam: int,
            profundidade: int) -> dict:
    """A melhor abertura de um mundo pelos DOIS objetivos: bits e tentativas.

    Serve aos dois tipos de mundo sem saber de qual se trata. Num mundo fechado o
    léxico do motor já é a lista curta, e num aberto ele é o léxico inteiro com o
    prior zerado fora do corte — nos dois casos os palpites são "o léxico do
    motor" e as candidatas são o que veio no argumento.
    """
    entropias = motor.entropias(candidatas)
    ordem = np.argsort(-entropias, kind="stable")
    escolha = MotorNivel3(motor, beam, profundidade).escolher(
        candidatas, tentativa=1, n_alternativas=3
    )
    return {
        "bits": motor.lexico.mostrar(int(ordem[0])),
        "bits_valor": float(entropias[ordem[0]]),
        "tentativas": escolha.palavra,
        "e": escolha.valor_esperado,
        "alternativas": [[p, round(v, 4)] for p, v, _ in escolha.alternativas],
    }


def rodar_catalogo(temperatura: float, beam: int, profundidade: int
                   ) -> tuple[list[Resultado], dict]:
    """A melhor abertura de cada cenário — o catálogo por trás da tabela do README.

    "Qual é a abertura ótima do Termo" não tem resposta sem três premissas
    declaradas: o objetivo (bits ou tentativas), quais palavras podem CAIR e
    quais você aceita DIGITAR. Este comando varre as três e mostra que quase
    nenhuma palavra sobrevive à troca de premissa.

    Não joga partida nenhuma — é tudo escolha de abertura —, então devolve a lista
    de resultados vazia e só o dicionário interessa.
    """
    lexico = Lexico.carregar(temperatura)
    motor = Motor(lexico)
    matriz = motor.matriz

    print(f"\n[1/3] mundo completo ({len(lexico)} respostas), variando o prior ...")
    completo = []
    for t in TEMPERATURAS_CATALOGO:
        m = Motor(lexico.com_temperatura(t), matriz=matriz)
        entropias = m.entropias(m.todas_candidatas())
        melhor = int(np.argmax(entropias))
        completo.append({"t": t, "bits": lexico.mostrar(melhor),
                         "bits_valor": float(entropias[melhor])})
        print(f"  T={t:<5g} bits: {completo[-1]['bits']}")

    # O nível 3 do mundo completo com T=1 é a abertura da CLI, que vem em cache;
    # com T→∞ é a última linha do mundo fechado, calculada lá embaixo.
    busca = MotorNivel3(motor, beam, profundidade)
    nivel3_padrao = None
    if busca.abertura_em_cache():
        sugestao = busca.abertura()
        nivel3_padrao = {"t": temperatura, "palavra": sugestao.palavra,
                         "e": sugestao.valor_esperado}
        print(f"  T={temperatura:g} tentativas: {sugestao.palavra} "
              f"(E={sugestao.valor_esperado:.4f}, em cache)")

    print("\n[2/3] mundo ABERTO — só as N mais comuns caem, o léxico todo é digitável")
    aberto = []
    for corte in CORTES_CATALOGO_ABERTO:
        n = corte or len(lexico)
        mundo, candidatas = mundo_aberto(lexico, n)
        dados = _otimos(Motor(mundo, matriz=matriz), candidatas, beam, profundidade)
        aberto.append({"n": n, **dados})
        print(f"  N={n:<5} bits: {dados['bits']:8s} {dados['bits_valor']:.3f}"
              f"    tentativas: {dados['tentativas']:8s} E={dados['e']:.4f}",
              flush=True)

    print("\n[3/3] mundo FECHADO — respostas e palpites saem da mesma lista curta")
    fechado = []
    for corte in CORTES_CATALOGO:
        n = corte or len(lexico)
        mundo, matriz_mundo = mundo_restrito(lexico, matriz, n)
        inicio = time.perf_counter()
        motor_mundo = Motor(mundo, matriz=matriz_mundo)
        dados = _otimos(motor_mundo, motor_mundo.todas_candidatas(), beam,
                        profundidade)
        fechado.append({"n": n, **dados})
        print(f"  N={n:<5} bits: {dados['bits']:8s} {dados['bits_valor']:.3f}"
              f"    tentativas: {dados['tentativas']:8s} E={dados['e']:.4f}"
              f"   [{time.perf_counter() - inicio:.0f}s]", flush=True)

    nivel1 = _ranking_nivel1(lexico, motor.todas_candidatas(),
                             lexico.indice_de(PALAVRA_FORUNS))
    print(f"\n  nível 1 (letras) no mundo completo: {nivel1['topo'][0]}")

    palavras = {d["bits"] for d in completo}
    palavras |= {d[k] for d in aberto + fechado for k in ("bits", "tentativas")}
    print(f"  {len(palavras)} palavras distintas são ótimas em algum cenário")

    return [], {
        "temperatura": temperatura,
        "beam": beam,
        "profundidade": profundidade,
        "nivel1": nivel1,
        "mundo_completo": completo,
        "nivel3_padrao": nivel3_padrao,
        "mundo_aberto": aberto,
        "mundo_fechado": fechado,
        "palavras_otimas": sorted(palavras),
    }


def _ranking_nivel1(lexico: Lexico, candidatas: np.ndarray, alvo: int,
                    topo: int = 4) -> dict:
    """Ranking da heurística de frequência de letras (nível 1) sobre `candidatas`."""
    heuristica = FrequenciaDeLetras(lexico)
    contem = heuristica.contem.astype(np.float64)
    ordem = np.argsort(-(contem @ contem[candidatas].sum(axis=0)), kind="stable")
    return {
        "topo": [lexico.mostrar(int(j)) for j in ordem[:topo]],
        "posicao": _posicao(ordem, alvo),
    }


def rodar_varredura_t(bateria: str, n: int | None) -> list[Resultado]:
    """Tentativas médias em função de T (seção 5.5).

    T -> inf recupera a entropia pura, então esta única curva compara o Nível 1
    e o Nível 2 do algoritmo de forma contínua.
    """
    base = Lexico.carregar(1.0)
    matriz = Motor(base).matriz  # construída/carregada uma vez só
    secretas = conjunto_secretas(base, bateria, n)
    resultados = []

    for temperatura in TEMPERATURAS_VARREDURA:
        lexico = base.com_temperatura(temperatura)
        motor = Motor(lexico, matriz=matriz)
        print(f"\n  T={temperatura} ...")
        # O conjunto de secretas é sempre o mesmo (definido pelo ICF, não por T).
        resultados.append(avaliar(motor, Entropia(motor), secretas, bateria))
        r = resultados[-1]
        print(f"    média={r.media_tentativas:.3f}  vitória={r.taxa_vitoria:.1%}")
    return resultados


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    analisador = argparse.ArgumentParser(description="Benchmark do solver de Termo")
    analisador.add_argument("--bateria", choices=("realista", "completo"),
                            default="realista")
    analisador.add_argument("--n", type=int, default=None,
                            help="quantas palavras secretas (padrão: 1500 / todas)")
    analisador.add_argument("--t", dest="temperatura", default="1.0")
    analisador.add_argument("--semente", type=int, default=0)
    analisador.add_argument("--varredura-t", action="store_true",
                            help="varre T em vez de comparar estratégias")
    analisador.add_argument("--nivel3", action="store_true",
                            help="head-to-head nível 2 vs nível 3 (lento)")
    analisador.add_argument("--serao", action="store_true",
                            help="por que os fóruns apontam `serão` e o solver não")
    analisador.add_argument("--catalogo", action="store_true",
                            help="a melhor abertura de cada cenário (lento: ~12 min)")
    analisador.add_argument("--beam", type=int, default=BEAM,
                            help=f"nível 3: palpites por nó (padrão {BEAM})")
    analisador.add_argument("--profundidade", type=int, default=PROFUNDIDADE,
                            help=f"nível 3: níveis de busca (padrão {PROFUNDIDADE})")
    argumentos = analisador.parse_args()
    bruto = str(argumentos.temperatura).lower()
    temperatura = math.inf if bruto in ("inf", "infinito") else float(bruto)

    inicio = time.perf_counter()
    extras: dict | None = None
    if argumentos.varredura_t:
        resultados = rodar_varredura_t(argumentos.bateria, argumentos.n)
        nome = f"varredura_t_{argumentos.bateria}.json"
    elif argumentos.serao:
        resultados, extras = rodar_serao(
            argumentos.n, temperatura, argumentos.beam, argumentos.profundidade
        )
        nome = "serao.json"
    elif argumentos.catalogo:
        resultados, extras = rodar_catalogo(
            temperatura, argumentos.beam, argumentos.profundidade
        )
        nome = "catalogo.json"
    elif argumentos.nivel3:
        resultados = rodar_nivel3(
            argumentos.bateria, argumentos.n, temperatura,
            argumentos.beam, argumentos.profundidade,
        )
        nome = f"comparacao_nivel3_{argumentos.bateria}.json"
    else:
        resultados = rodar_comparacao(
            argumentos.bateria, argumentos.n, temperatura, argumentos.semente
        )
        nome = f"comparacao_{argumentos.bateria}.json"

    if resultados:
        print("\n" + formatar_tabela(resultados))
    if extras is None:
        caminho = salvar(resultados, nome)
    else:
        # Aqui as partidas são só uma das quatro medições; o resto é ranking.
        caminho = salvar_bruto(
            {"experimento": extras, "partidas": [asdict(r) for r in resultados]}, nome
        )
    print(f"\nresultados em {caminho}")
    print(f"tempo total: {time.perf_counter() - inicio:.0f}s")

    if not (argumentos.varredura_t or argumentos.serao or argumentos.catalogo):
        pior = max(resultados, key=lambda r: r.media_tentativas)
        melhor = min(resultados, key=lambda r: r.media_penalizada)
        print(f"\nmelhor (média penalizada): {melhor.estrategia} "
              f"{melhor.media_penalizada:.3f}")
        print(f"pior (média entre vitórias): {pior.estrategia} "
              f"{pior.media_tentativas:.3f}")


if __name__ == "__main__":
    main()
