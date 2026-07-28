"""Motor de entropia e regra de endgame (seção 4 da especificação).

Para cada tentativa candidata `g`, particiona o conjunto atual de candidatas `C`
pelos padrões de feedback que `g` produziria, pesando cada candidata pelo prior:

    p(padrão) = Σ_{c em C, feedback(g,c)=padrão} prior(c)  / Σ prior(C)
    H(g)      = -Σ p log₂ p

Escolhe-se o `g` de maior H. O espaço de tentativas é o léxico completo — não só
as candidatas restantes —, o que permite "queimar" uma jogada numa palavra
improvável mas muito informativa. O prior entra apenas no peso das candidatas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .feedback import N_PADROES, PADRAO_VITORIA, normalizar
from .lexico import DIR_DADOS, Lexico
from .matriz import carregar_matriz, codigos_contra

ARQ_ABERTURA = DIR_DADOS / "aberturas.json"

# Na penúltima tentativa com no máximo este número de candidatas, chuta-se uma
# candidata em vez da palavra de maior entropia (seção 4.5).
LIMIAR_ENDGAME = 3

# Entropias que diferem menos que isso são consideradas empate (seção 4.5).
EPSILON_EMPATE = 1e-9

N_MAX_TENTATIVAS = 6


@dataclass
class Sugestao:
    """Jogada sugerida e o porquê.

    `palavra` e as alternativas vêm na forma ACENTUADA — é o que o jogador vê na
    tela do jogo (§7.2). O motor por dentro só manipula índices.
    """

    indice: int
    palavra: str
    entropia: float
    motivo: str
    e_candidata: bool
    alternativas: list[tuple[str, float, bool]] = field(default_factory=list)


class Motor:
    """Motor do solver: filtragem de candidatas + escolha da próxima jogada."""

    def __init__(
        self,
        lexico: Lexico,
        matriz: np.ndarray | None = None,
        limiar_endgame: int = LIMIAR_ENDGAME,
        n_max_tentativas: int = N_MAX_TENTATIVAS,
    ):
        self.lexico = lexico
        self.matriz = carregar_matriz(lexico.palavras) if matriz is None else matriz
        self.limiar_endgame = limiar_endgame
        self.n_max_tentativas = n_max_tentativas
        self._cache_escolha: dict[bytes, Sugestao] = {}

    # ------------------------------------------------------------- filtragem

    def todas_candidatas(self) -> np.ndarray:
        return np.arange(len(self.lexico), dtype=np.int32)

    def filtrar(
        self, candidatas: np.ndarray, tentativa: str, codigo: int
    ) -> np.ndarray:
        """Mantém apenas as candidatas compatíveis com o feedback recebido.

        Reutiliza exatamente a mesma função de feedback usada em todo o resto —
        não há uma segunda lógica de filtragem por regras (seção 4.4).
        """
        indice = self.lexico.indice.get(normalizar(tentativa))
        if indice is not None:
            codigos = self.matriz[indice, candidatas]
        else:
            # Tentativa fora do léxico: sem linha na matriz, calcula na hora.
            codigos = codigos_contra(
                tentativa, [self.lexico.palavras[i] for i in candidatas]
            )
        return candidatas[codigos == codigo]

    # -------------------------------------------------------------- entropia

    def entropias(self, candidatas: np.ndarray) -> np.ndarray:
        """Entropia (em bits) de cada palavra do léxico como próxima tentativa."""
        n = len(self.lexico)
        m = len(candidatas)
        if m <= 1:
            return np.zeros(n, dtype=np.float64)

        pesos = self.lexico.prior[candidatas]
        soma = pesos.sum()
        pesos = pesos / soma if soma > 0 else np.full(m, 1.0 / m)

        entropias = np.empty(n, dtype=np.float64)
        # Bloco escolhido para manter os temporários na casa de dezenas de MB.
        bloco = max(1, min(n, 2_000_000 // m))
        deslocamento = np.arange(bloco, dtype=np.int32) * N_PADROES

        for i0 in range(0, n, bloco):
            i1 = min(i0 + bloco, n)
            b = i1 - i0
            sub = self.matriz[i0:i1][:, candidatas]  # (b, m) uint8
            plano = sub.astype(np.int32) + deslocamento[:b, None]
            baldes = np.bincount(
                plano.ravel(),
                weights=np.broadcast_to(pesos, (b, m)).ravel(),
                minlength=b * N_PADROES,
            ).reshape(b, N_PADROES)

            # Padrões com p = 0 não contribuem (e log(0) explode).
            logs = np.log2(baldes, out=np.zeros_like(baldes), where=baldes > 0)
            entropias[i0:i1] = -(baldes * logs).sum(axis=1)

        return entropias

    # --------------------------------------------------------------- escolha

    def _melhor_candidata(self, candidatas: np.ndarray) -> int:
        """Candidata de maior prior (menor ICF)."""
        return int(candidatas[np.argmax(self.lexico.prior[candidatas])])

    def _ordenar_por_entropia(
        self, entropias: np.ndarray, candidatas: np.ndarray
    ) -> np.ndarray:
        """Ordem de preferência: maior entropia, depois ser candidata, depois prior.

        O desempate por "também é candidata" é ganho grátis: entre duas palavras
        que separam igualmente bem, a que pode ser a resposta tem chance não-nula
        de encerrar o jogo (seção 4.5).
        """
        e_candidata = np.zeros(len(self.lexico), dtype=bool)
        e_candidata[candidatas] = True
        arredondada = np.round(entropias / EPSILON_EMPATE) * EPSILON_EMPATE
        # lexsort: a última chave é a primária.
        return np.lexsort((-self.lexico.prior, ~e_candidata, -arredondada))

    def escolher(
        self, candidatas: np.ndarray, tentativa: int = 1, n_alternativas: int = 3
    ) -> Sugestao:
        """Próxima jogada, aplicando a regra de endgame da seção 4.5."""
        m = len(candidatas)
        if m == 0:
            raise ValueError("conjunto de candidatas vazio")

        palavras = self.lexico.exibicao  # o usuário vê a forma acentuada

        if m == 1:
            i = int(candidatas[0])
            return Sugestao(i, palavras[i], 0.0, "única candidata restante", True)

        if m == 2:
            i = self._melhor_candidata(candidatas)
            outra = [palavras[j] for j in candidatas if j != i]
            return Sugestao(
                i, palavras[i], 0.0,
                f"restam 2 candidatas; chuta a mais provável (outra: {outra[0]})",
                True,
                [(palavras[j], 0.0, True) for j in candidatas if j != i],
            )

        ultima = tentativa >= self.n_max_tentativas
        penultima_apertada = (
            tentativa == self.n_max_tentativas - 1 and m <= self.limiar_endgame
        )
        if ultima or penultima_apertada:
            i = self._melhor_candidata(candidatas)
            motivo = (
                "última tentativa: só faz sentido chutar uma candidata"
                if ultima
                else f"penúltima tentativa com {m} candidatas: chuta a mais provável"
            )
            alternativas = [
                (palavras[j], 0.0, True)
                for j in candidatas[np.argsort(-self.lexico.prior[candidatas])][
                    1 : n_alternativas + 1
                ]
            ]
            return Sugestao(i, palavras[i], 0.0, motivo, True, alternativas)

        entropias = self.entropias(candidatas)
        ordem = self._ordenar_por_entropia(entropias, candidatas)
        conjunto = set(candidatas.tolist())
        melhor = int(ordem[0])
        alternativas = [
            (palavras[int(j)], float(entropias[j]), int(j) in conjunto)
            for j in ordem[1 : n_alternativas + 1]
        ]
        return Sugestao(
            melhor,
            palavras[melhor],
            float(entropias[melhor]),
            f"maior entropia entre as {len(palavras)} palavras do léxico "
            f"({m} candidatas restantes)",
            melhor in conjunto,
            alternativas,
        )

    def escolher_com_cache(self, candidatas: np.ndarray, tentativa: int) -> Sugestao:
        """`escolher` memoizado pelo conjunto de candidatas — usado no benchmark.

        Milhares de jogos convergem para os mesmos conjuntos (sobretudo depois da
        abertura fixa, que tem no máximo 243 desdobramentos).
        """
        # A tentativa só importa perto do fim; antes disso o estado é o conjunto.
        rodada = tentativa if tentativa >= self.n_max_tentativas - 1 else 0
        chave = rodada.to_bytes(1, "little") + candidatas.astype(np.int32).tobytes()
        resultado = self._cache_escolha.get(chave)
        if resultado is None:
            resultado = self.escolher(candidatas, tentativa, n_alternativas=0)
            if len(self._cache_escolha) < 500_000:
                self._cache_escolha[chave] = resultado
        return resultado

    # -------------------------------------------------------------- abertura

    def abertura(self, n_alternativas: int = 5, usar_cache: bool = True) -> Sugestao:
        """Melhor primeira jogada. Não depende de feedback nenhum, então é fixa.

        Calculada uma vez e guardada em disco: é de longe a conta mais cara do
        uso diário (8.996 x 8.996 partições).
        """
        chave = f"T={self.lexico.temperatura}"
        cache: dict = {}
        if usar_cache and ARQ_ABERTURA.exists():
            cache = json.loads(ARQ_ABERTURA.read_text(encoding="utf-8"))
            guardada = cache.get(chave)
            if guardada and guardada.get("n") == len(self.lexico):
                return Sugestao(
                    self.lexico.indice_de(guardada["palavra"]),
                    guardada["palavra"],
                    guardada["entropia"],
                    "melhor abertura (em cache)",
                    True,
                    [tuple(alt) for alt in guardada["alternativas"]],
                )

        sugestao = self.escolher(self.todas_candidatas(), tentativa=1,
                                 n_alternativas=max(n_alternativas, 5))
        sugestao.motivo = "melhor abertura do léxico"
        if usar_cache:
            cache[chave] = {
                "n": len(self.lexico),
                "palavra": sugestao.palavra,
                "entropia": sugestao.entropia,
                "alternativas": [list(alt) for alt in sugestao.alternativas],
            }
            ARQ_ABERTURA.parent.mkdir(parents=True, exist_ok=True)
            ARQ_ABERTURA.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        return sugestao


def carregar_motor(temperatura: float = 1.0, **kwargs) -> Motor:
    """Atalho: léxico + matriz + motor prontos para uso."""
    return Motor(Lexico.carregar(temperatura), **kwargs)


if __name__ == "__main__":
    import sys
    import time

    sys.stdout.reconfigure(encoding="utf-8")
    temperatura = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    motor = carregar_motor(temperatura)

    inicio = time.perf_counter()
    abertura = motor.abertura(usar_cache=False)
    print(f"abertura (T={temperatura}): {abertura.palavra}  "
          f"H={abertura.entropia:.4f} bits  [{time.perf_counter() - inicio:.1f}s]")
    for palavra, entropia, e_cand in abertura.alternativas:
        print(f"   {palavra}  H={entropia:.4f}{'  (candidata)' if e_cand else ''}")

    inicio = time.perf_counter()
    motor.abertura(usar_cache=True)
    restantes = motor.filtrar(motor.todas_candidatas(), abertura.palavra, PADRAO_VITORIA)
    print(f"filtragem por GGGGG deixa {len(restantes)} candidata(s)")
    seguinte = motor.filtrar(motor.todas_candidatas(), abertura.palavra, 0)
    inicio = time.perf_counter()
    print(f"após feedback BBBBB: {len(seguinte)} candidatas -> "
          f"{motor.escolher(seguinte, 2).palavra} "
          f"[{time.perf_counter() - inicio:.2f}s]")
