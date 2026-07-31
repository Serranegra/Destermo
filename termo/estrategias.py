"""Estratégias competidoras do benchmark (seção 5.2 da especificação).

Todas expõem a mesma interface — `escolher(candidatas, rodada) -> índice` — para
que o simulador não precise saber com qual está lidando.

  1. Aleatória            piso de comparação
  2. Frequência de letras a heurística que a maioria dos solvers usa
  3. Mais provável        sempre a candidata de menor ICF
  4. Entropia             o motor da seção 4 (nível 2)
  5. Nível 3              minimiza tentativas esperadas, não bits (opcional)

As três primeiras sempre chutam uma candidata, então a regra de endgame não se
aplica a elas: nunca "queimam" uma tentativa.

A quinta é opcional porque custa ordens de magnitude mais que as outras quatro
juntas — entra no benchmark só quando pedida (`benchmark.py --nivel3`).

`AberturaFixa` não é uma competidora: é uma sonda. Ela força a primeira jogada e
delega o resto ao nível 2, para medir quanto uma abertura vale isolando-a da
política que vem depois (`benchmark.py --serao`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .entropia import Motor
from .lexico import Lexico
from .nivel3 import BEAM, PROFUNDIDADE, MotorNivel3


class Estrategia(ABC):
    nome: str = "?"

    @abstractmethod
    def escolher(self, candidatas: np.ndarray, rodada: int) -> int:
        """Índice da palavra a jogar nesta rodada."""

    def reiniciar(self) -> None:
        """Chamado no início de cada jogo."""


class Aleatoria(Estrategia):
    """Chuta qualquer candidata restante. Piso de comparação."""

    nome = "aleatória"

    def __init__(self, semente: int = 0):
        self.semente = semente
        self._rng = np.random.default_rng(semente)

    def reiniciar(self) -> None:
        self._rng = np.random.default_rng(self.semente)

    def escolher(self, candidatas: np.ndarray, rodada: int) -> int:
        return int(self._rng.choice(candidatas))


class FrequenciaDeLetras(Estrategia):
    """Score = soma das frequências das letras (distintas) entre as candidatas.

    É a abordagem que o artigo do Gabriel Yshay mostrou ser furada, e a que a
    maioria dos solvers de Termo por aí usa. Serve de régua honesta.
    """

    nome = "freq. de letras"

    def __init__(self, lexico: Lexico):
        alfabeto = sorted({letra for palavra in lexico.palavras for letra in palavra})
        posicao = {letra: i for i, letra in enumerate(alfabeto)}
        self.contem = np.zeros((len(lexico), len(alfabeto)), dtype=bool)
        for i, palavra in enumerate(lexico.palavras):
            for letra in palavra:
                self.contem[i, posicao[letra]] = True

    def escolher(self, candidatas: np.ndarray, rodada: int) -> int:
        contem = self.contem[candidatas].astype(np.float64)
        # Frequência = em quantas candidatas restantes cada letra aparece.
        frequencia = contem.sum(axis=0)
        # Letras distintas: uma letra repetida não conta duas vezes.
        return int(candidatas[np.argmax(contem @ frequencia)])


class MaisProvavel(Estrategia):
    """Chuta sempre a candidata de maior prior (menor ICF)."""

    nome = "mais provável"

    def __init__(self, lexico: Lexico):
        self.prior = lexico.prior

    def escolher(self, candidatas: np.ndarray, rodada: int) -> int:
        return int(candidatas[np.argmax(self.prior[candidatas])])


class Entropia(Estrategia):
    """Maximiza a informação esperada, com a regra de endgame da seção 4.5."""

    def __init__(self, motor: Motor):
        self.motor = motor
        self.nome = f"entropia (T={motor.lexico.temperatura:g})"
        self._abertura: int | None = None

    def escolher(self, candidatas: np.ndarray, rodada: int) -> int:
        if rodada == 1:
            if self._abertura is None:
                self._abertura = self.motor.abertura().indice
            return self._abertura
        return self.motor.escolher_com_cache(candidatas, rodada).indice


class Nivel3(Estrategia):
    """Minimiza tentativas esperadas em vez de bits ganhos (§4.1, nível 3).

    Mesma forma da `Entropia`: abertura fixa (não depende de feedback) e escolhas
    memoizadas pelo conjunto de candidatas. Aqui o cache pesa muito mais — cada
    escolha custa uma árvore de decisão inteira.
    """

    def __init__(self, motor: MotorNivel3):
        self.motor = motor
        self.nome = f"nível 3 (K={motor.beam}, P={motor.profundidade})"
        self._abertura: int | None = None

    def escolher(self, candidatas: np.ndarray, rodada: int) -> int:
        if rodada == 1:
            if self._abertura is None:
                self._abertura = self.motor.abertura().indice
            return self._abertura
        return self.motor.escolher_com_cache(candidatas, rodada).indice


class AberturaFixa(Estrategia):
    """Abertura imposta, nível 2 daí em diante.

    Isola o efeito da primeira jogada: como a política pós-abertura é a mesma, a
    diferença de média entre duas instâncias é atribuível só à abertura. É assim
    que se compara uma sugestão de fora — `serão`, digamos — com a do solver.
    """

    def __init__(self, motor: Motor, palavra: str):
        self.motor = motor
        self.indice = motor.lexico.indice_de(palavra)
        self.nome = f"abertura {motor.lexico.mostrar(self.indice)}"

    def escolher(self, candidatas: np.ndarray, rodada: int) -> int:
        if rodada == 1:
            return self.indice
        return self.motor.escolher_com_cache(candidatas, rodada).indice


def construir_todas(
    motor: Motor,
    semente: int = 0,
    nivel3: MotorNivel3 | None = None,
) -> list[Estrategia]:
    """As quatro estratégias da seção 5.2, na ordem da tabela.

    Com um motor de nível 3 em `nivel3`, ele entra como quinta série — mas a
    paleta da marca tem quatro cores, então o gráfico previsto para ele é o
    head-to-head contra a entropia (`benchmark.py --nivel3`), não esta tabela.
    """
    estrategias: list[Estrategia] = [
        Aleatoria(semente),
        FrequenciaDeLetras(motor.lexico),
        MaisProvavel(motor.lexico),
        Entropia(motor),
    ]
    if nivel3 is not None:
        estrategias.append(Nivel3(nivel3))
    return estrategias


def construir_nivel3(
    motor: Motor, beam: int = BEAM, profundidade: int = PROFUNDIDADE
) -> Nivel3:
    """Atalho para o head-to-head nível 2 vs nível 3 sobre o mesmo motor."""
    return Nivel3(MotorNivel3(motor, beam, profundidade))
