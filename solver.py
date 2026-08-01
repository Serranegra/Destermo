#!/usr/bin/env python
"""CLI interativa do solver de Termo (seção 7 da especificação).

Esta camada só faz entrada/saída. Toda a lógica vive em `termo/` — trocar a CLI
por um bot, uma API ou uma página web não exige mexer no motor (seção 7.4).

    python solver.py                  # nível 3, T = 1.0, abertura padrão
    python solver.py --nivel 2        # só entropia: milissegundos por jogada
    python solver.py --abertura otima # abre pelo ótimo do nível escolhido
    python solver.py --t 5            # outra temperatura do prior
    python solver.py --t inf          # entropia pura, sem prior de frequência
    python solver.py --ampliado       # deixa sondar com conjugações (§2.5)

O padrão é o nível 3 (§4.1): ele minimiza o número esperado de tentativas em vez
de maximizar bits, o que vale 0,54 tentativa na bateria realista. Cada jogada
custa décimos de segundo.

A abertura é a exceção: ela é fixa em `tarso` nos dois níveis e em qualquer T (ver
`ABERTURA_PADRAO`), porque é a única que não tem regime ruim e porque a primeira
jogada inteira vale ≤ 0,07 tentativa. `--abertura otima` devolve o ótimo do
critério do nível escolhido; da segunda jogada em diante nada muda.

Os dois níveis expõem a mesma interface (`abertura`, `abertura_padrao`,
`escolher`), então daqui para baixo nada sabe qual dos dois está respondendo.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from termo.entropia import ABERTURA_PADRAO, N_MAX_TENTATIVAS, Motor, Sugestao
from termo.feedback import (
    normalizar,
    normalizar_padrao,
    padrao_para_codigo,
    padrao_possivel,
)
from termo.lexico import Lexico
from termo.nivel3 import BEAM, PROFUNDIDADE, MotorNivel3

AJUDA = """
comandos:
  <palavra>          registra a tentativa que você jogou (digite SEM acento —
                     o Termo os preenche sozinho; com acento também funciona)
  voltar             desfaz a última rodada
  listar [n]         mostra as candidatas restantes
  sair               encerra

feedback: 5 caracteres, um por posição
  G / verde    letra certa na posição certa     (aceita também 2 ou 🟩)
  Y / amarelo  letra existe, posição errada     (aceita também 1 ou 🟨)
  B / preto    letra não existe                 (aceita também 0 ou ⬛)
"""

VOLTAR = ("voltar", "desfazer", "undo")
SAIR = ("sair", "quit", "exit")

# Os dois motores têm a mesma interface; a CLI não distingue um do outro.
Cerebro = Motor | MotorNivel3


class Cancelado(Exception):
    """Usuário pediu para desfazer a rodada em andamento."""


def preparar_console() -> None:
    """Windows: o console não usa UTF-8 por padrão e come os acentos."""
    for fluxo in (sys.stdout, sys.stdin):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def ler(prompt: str) -> str:
    try:
        texto = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0) from None
    if texto.lower() in SAIR:
        raise SystemExit(0)
    return texto


def formatar_sugestao(sugestao: Sugestao, rotulo: str) -> str:
    detalhes = []
    if sugestao.entropia > 0:
        detalhes.append(f"{sugestao.entropia:.2f} bits")
    if sugestao.valor_esperado is not None:
        detalhes.append(f"E={sugestao.valor_esperado:.2f} tentativas")
    detalhes.append("é candidata" if sugestao.e_candidata else "não é candidata")
    linhas = [
        f"  {rotulo}: {sugestao.palavra}   ({', '.join(detalhes)})",
        f"  motivo: {sugestao.motivo}",
    ]
    if sugestao.alternativas:
        alternativas = ", ".join(
            f"{palavra}{'*' if e_cand else ''}"
            for palavra, _, e_cand in sugestao.alternativas
        )
        linhas.append(f"  alternativas: {alternativas}   (* = também é candidata)")
    return "\n".join(linhas)


def mostrar_candidatas(motor: Cerebro, candidatas: np.ndarray, limite: int) -> None:
    ordem = candidatas[np.argsort(-motor.lexico.prior[candidatas])]
    nomes = [motor.lexico.mostrar(i) for i in ordem[:limite]]
    print(f"  {len(candidatas)} candidatas (mais prováveis primeiro):")
    print("   " + ", ".join(nomes) + (" ..." if len(candidatas) > limite else ""))


def ler_tentativa(motor: Cerebro, candidatas: np.ndarray, rodada: int) -> str:
    """Devolve a tentativa já normalizada (sem acento) — é o que o motor usa."""
    while True:
        texto = ler(f"\n[{rodada}] tentativa > ").lower()
        if not texto:
            continue
        if texto in VOLTAR:
            raise Cancelado
        if texto in ("?", "ajuda", "help"):
            print(AJUDA)
            continue
        if texto.startswith("listar"):
            partes = texto.split()
            limite = int(partes[1]) if len(partes) > 1 and partes[1].isdigit() else 20
            mostrar_candidatas(motor, candidatas, limite)
            continue
        # O jogador digita sem acento (o Termo preenche sozinho), mas aceitamos
        # entrada acentuada e normalizamos (§7.2).
        tentativa = normalizar(texto)
        if len(tentativa) != 5:
            print("  ! a tentativa precisa ter 5 letras")
            continue
        if tentativa not in motor.lexico.indice_sonda:
            # Seção 7.3: avisar, mas permitir — o Termo aceita palavras que não temos.
            print("  ! essa palavra não está no nosso léxico (seguindo mesmo assim)")
        return tentativa


def ler_feedback(tentativa: str) -> str:
    while True:
        texto = ler(f"[?] feedback de '{tentativa}' > ")
        if texto.lower() in VOLTAR:
            raise Cancelado
        try:
            padrao = normalizar_padrao(texto)
        except ValueError as erro:
            print(f"  ! {erro}. Use 5 chars em G/Y/B.")
            continue
        if not padrao_possivel(tentativa, padrao):
            # Seção 7.3: detectar ANTES de zerar C.
            print(
                f"  ! '{padrao}' é logicamente impossível para '{tentativa}': nenhuma\n"
                "    palavra secreta produziria esse feedback. Confira a digitação."
            )
            continue
        return padrao


AVISO_SEM_CANDIDATAS = """
  ! nenhuma candidata é compatível com esse feedback.
    Ou o feedback foi digitado errado, ou a palavra do dia não está no nosso
    léxico de {n} palavras — ele é uma aproximação de fonte legítima
    (fserb/pt-br), não a lista oficial do Termo.
    A rodada NÃO foi aplicada: reinforme a tentativa e o feedback, ou digite
    'voltar' para desfazer a rodada anterior."""


def rotular(motor: Cerebro, tentativa: str) -> str:
    """Forma acentuada da tentativa, quando ela é uma palavra que conhecemos."""
    indice = motor.lexico.indice_sonda.get(tentativa)
    return motor.lexico.mostrar(indice) if indice is not None else tentativa


def jogar(motor: Cerebro, abertura_fixa: bool = True) -> None:
    candidatas = motor.todas_candidatas()
    historico: list[np.ndarray] = []

    # A palavra fixa ou o ótimo do nível — daqui em diante os dois caminhos são o
    # mesmo motor, e `voltar` também passa por aqui ao desfazer a primeira rodada.
    abertura = motor.abertura_padrao if abertura_fixa else motor.abertura

    if isinstance(motor, MotorNivel3) and not abertura_fixa and (
        not motor.abertura_em_cache()
    ):
        # Só a configuração padrão vem com a abertura versionada; quem mexe em
        # --t/--beam/--profundidade cai aqui. A saída de emergência vem junto — e
        # com a abertura padrão nem se chega neste ramo, que ela não busca nada.
        print("\nAbertura ótima do nível 3 fora do cache para esta configuração:"
              "\nsão ~9 min de busca na árvore, uma vez só (o resultado vai para"
              "\ndata/aberturas_nivel3.json). Para começar já, tire o"
              "\n--abertura otima: a palavra fixa não busca nada.")
    else:
        print("\nCalculando a melhor abertura...")
    print(formatar_sugestao(abertura(), "melhor abertura"))

    rodada = 1
    while rodada <= N_MAX_TENTATIVAS:
        try:
            tentativa = ler_tentativa(motor, candidatas, rodada)
            padrao = ler_feedback(rotular(motor, tentativa))
        except Cancelado:
            if not historico:
                print("  ! não há rodada anterior para desfazer")
                continue
            candidatas = historico.pop()
            rodada -= 1
            print(f"\n  rodada desfeita — de volta à rodada {rodada}, "
                  f"{len(candidatas)} candidatas")
            sugestao = (
                abertura() if rodada == 1 else motor.escolher(candidatas, rodada)
            )
            print(formatar_sugestao(sugestao, "sugestão"))
            continue

        if padrao == "GGGGG":
            print(f"\n  acertou '{rotular(motor, tentativa)}' em {rodada} "
                  f"tentativa(s). Fim.")
            return

        restantes = motor.filtrar(candidatas, tentativa, padrao_para_codigo(padrao))
        if len(restantes) == 0:
            # Seção 7.3: nunca crashar por C vazio — é cenário esperado (ver 2.6).
            print(AVISO_SEM_CANDIDATAS.format(n=len(motor.lexico)))
            continue

        historico.append(candidatas)
        candidatas = restantes
        rodada += 1

        print(f"\n  candidatas restantes: {len(candidatas)}")
        if rodada > N_MAX_TENTATIVAS:
            break
        print(formatar_sugestao(motor.escolher(candidatas, rodada), "sugestão"))

    print("\n  acabaram as 6 tentativas.")


def main() -> None:
    preparar_console()
    analisador = argparse.ArgumentParser(description="Solver de Termo por entropia")
    analisador.add_argument(
        "--t", "--temperatura", dest="temperatura", default="1.0",
        help="temperatura do prior de frequência ('inf' = entropia pura)",
    )
    analisador.add_argument(
        "--nivel", type=int, choices=(2, 3), default=3,
        help="3 = minimiza tentativas esperadas (padrão); 2 = entropia pura",
    )
    analisador.add_argument(
        "--abertura", choices=("padrao", "otima"), default="padrao",
        help=f"'padrao' abre sempre por '{ABERTURA_PADRAO}'; 'otima' abre pelo "
             "ótimo do critério do nível escolhido, que muda com o T",
    )
    analisador.add_argument(
        "--beam", type=int, default=BEAM,
        help=f"nível 3: palpites testados por nó (padrão {BEAM})",
    )
    analisador.add_argument(
        "--profundidade", type=int, default=PROFUNDIDADE,
        help=f"nível 3: níveis de busca abaixo da raiz (padrão {PROFUNDIDADE})",
    )
    analisador.add_argument(
        "--ampliado", action="store_true",
        help="permite sondar com conjugações (8.628 palavras jogáveis, 6.046 "
             "respostas possíveis); baixa e processa a fonte na primeira vez",
    )
    argumentos = analisador.parse_args()
    bruto = argumentos.temperatura.lower()
    temperatura = math.inf if bruto in ("inf", "infinito") else float(bruto)

    lexico = Lexico.carregar(temperatura, argumentos.ampliado)
    motor: Cerebro = Motor(lexico)
    if argumentos.nivel == 3:
        motor = MotorNivel3(motor, argumentos.beam, argumentos.profundidade)
        print("Solver de Termo — nível 3: menor nº esperado de tentativas")
    else:
        print("Solver de Termo — entropia com prior de frequência")
    print("Digite '?' a qualquer momento para ver os comandos.")
    busca = (
        f"   beam={argumentos.beam} profundidade={argumentos.profundidade}"
        if argumentos.nivel == 3
        else ""
    )
    # A abertura vale para os dois níveis, então sai fora do bloco da busca.
    busca += f"   abertura={argumentos.abertura}"
    espaco = (
        f"{len(lexico)} respostas, {lexico.n_sondas} jogáveis"
        if lexico.ampliado
        else f"{len(lexico)} palavras"
    )
    print(f"léxico: {espaco}   T={temperatura}{busca}")
    jogar(motor, argumentos.abertura == "padrao")


if __name__ == "__main__":
    main()
