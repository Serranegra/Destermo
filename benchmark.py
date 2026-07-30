#!/usr/bin/env python
"""Benchmark das estratégias (seção 5 da especificação).

Pergunta central: a estratégia de entropia compensa o custo computacional, ou
uma heurística simples chega perto?

    python benchmark.py                       # bateria realista, 4 estratégias
    python benchmark.py --bateria completo    # stress test: léxico inteiro
    python benchmark.py --varredura-t         # tentativas médias em função de T
    python benchmark.py --n 300               # amostra menor, para iterar rápido
    python benchmark.py --nivel3              # head-to-head nível 2 vs nível 3

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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from termo.entropia import N_MAX_TENTATIVAS, Motor
from termo.estrategias import (
    Aleatoria,
    Entropia,
    Estrategia,
    FrequenciaDeLetras,
    MaisProvavel,
    construir_nivel3,
)
from termo.lexico import Lexico
from termo.nivel3 import BEAM, PROFUNDIDADE

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


# ------------------------------------------------------------------ simulação


def jogar(
    motor: Motor, estrategia: Estrategia, secreta: int, n_max: int = N_MAX_TENTATIVAS
) -> tuple[int | None, list[int]]:
    """Joga uma partida completa. Devolve (nº de tentativas ou None, jogadas)."""
    estrategia.reiniciar()
    candidatas = motor.todas_candidatas()
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
) -> Resultado:
    palavras = motor.lexico.exibicao  # relatórios saem na forma acentuada
    tentativas: list[int] = []
    derrotas: list[str] = []
    piores: list[str] = []
    inicio = time.perf_counter()

    for posicao, secreta in enumerate(secretas):
        n, _ = jogar(motor, estrategia, int(secreta), n_max)
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


def salvar(resultados: list[Resultado], nome: str) -> Path:
    DIR_RESULTADOS.mkdir(exist_ok=True)
    caminho = DIR_RESULTADOS / nome
    caminho.write_text(
        json.dumps([asdict(r) for r in resultados], ensure_ascii=False, indent=2),
        encoding="utf-8",
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
    analisador.add_argument("--beam", type=int, default=BEAM,
                            help=f"nível 3: palpites por nó (padrão {BEAM})")
    analisador.add_argument("--profundidade", type=int, default=PROFUNDIDADE,
                            help=f"nível 3: níveis de busca (padrão {PROFUNDIDADE})")
    argumentos = analisador.parse_args()
    bruto = str(argumentos.temperatura).lower()
    temperatura = math.inf if bruto in ("inf", "infinito") else float(bruto)

    inicio = time.perf_counter()
    if argumentos.varredura_t:
        resultados = rodar_varredura_t(argumentos.bateria, argumentos.n)
        nome = f"varredura_t_{argumentos.bateria}.json"
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

    print("\n" + formatar_tabela(resultados))
    caminho = salvar(resultados, nome)
    print(f"\nresultados em {caminho}")
    print(f"tempo total: {time.perf_counter() - inicio:.0f}s")

    if not argumentos.varredura_t:
        pior = max(resultados, key=lambda r: r.media_tentativas)
        melhor = min(resultados, key=lambda r: r.media_penalizada)
        print(f"\nmelhor (média penalizada): {melhor.estrategia} "
              f"{melhor.media_penalizada:.3f}")
        print(f"pior (média entre vitórias): {pior.estrategia} "
              f"{pior.media_tentativas:.3f}")


if __name__ == "__main__":
    main()
