"""Motor de entropia e regra de endgame (seção 4 da especificação).

Para cada tentativa candidata `g`, particiona o conjunto atual de candidatas `C`
pelos padrões de feedback que `g` produziria, pesando cada candidata pelo prior:

    p(padrão) = Σ_{c em C, feedback(g,c)=padrão} prior(c)  / Σ prior(C)
    H(g)      = -Σ p log₂ p

Escolhe-se o `g` de maior H. O espaço de tentativas é o léxico completo — não só
as candidatas restantes —, o que permite "queimar" uma jogada numa palavra
improvável mas muito informativa. O prior entra apenas no peso das candidatas.

Com `Lexico.carregar(ampliado=True)` esse espaço cresce para as 8.628 sondas
(§2.5): as conjugações não podem ser a resposta, mas podem ser digitadas, e como
o prior delas é 0 elas nunca competem com uma candidata em caso de empate. As
linhas da matriz são o espaço de sonda; as colunas, o de candidata. Nada mais no
módulo muda — `candidatas` continua sendo índice de coluna.
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

# Até quantas candidatas vale calcular a entropia por agrupamento par a par em vez
# de pelos 243 baldes. O equilíbrio teórico é m² ≈ 243 (m ≈ 16), e medido dá 10,7x
# em m=4, 5,1x em m=8 e só 1,2x em m=16 — o limiar fica em 12 para ficar dentro da
# faixa em que o ganho é grande, sem depender de a máquina bater igual à medição.
LIMIAR_POUCAS = 12

N_MAX_TENTATIVAS = 6

# A palavra que as interfaces abrem, em qualquer temperatura e nos dois motores.
#
# Não é o ótimo de nenhum critério isolado, e é essa a graça: `tarso` é a de maior
# entropia sob o prior do projeto E a que empata em primeiro no minimax de
# arrependimento sobre sete distribuições de secreta, da lista bem curada à
# uniforme (README, "E se a própria distribuição for desconhecida?"). As ótimas de
# cada T vencem no regime em que foram calculadas e pagam no outro; esta não tem
# regime ruim. Como a abertura inteira vale ≤ 0,07 tentativa, trocar o ótimo local
# pela estabilidade sai barato — e uma abertura que não muda quando o usuário mexe
# no T é uma palavra a menos para decorar.
#
# `abertura()` continua devolvendo o ótimo calculado: é o que o benchmark mede e o
# que `analise.py` tabula por temperatura. A palavra fixa é escolha de interface, e
# mora em `abertura_padrao()`.
ABERTURA_PADRAO = "tarso"
MOTIVO_ABERTURA_PADRAO = "melhor abertura - robusto a palavras comuns e raras"


def _alinhar(prior: np.ndarray, n: int) -> np.ndarray:
    """Prior no tamanho exato do espaço de tentativa, completando com zeros."""
    if len(prior) == n:
        return prior
    if len(prior) > n:
        return prior[:n]
    return np.concatenate([prior, np.zeros(n - len(prior), dtype=prior.dtype)])


@dataclass
class Sugestao:
    """Jogada sugerida e o porquê.

    `palavra` e as alternativas vêm na forma ACENTUADA — é o que o jogador vê na
    tela do jogo (§7.2). O motor por dentro só manipula índices.

    `valor_esperado` só é preenchido pelo nível 3 (`termo/nivel3.py`), que escolhe
    por E[nº de tentativas] em vez de por bits; nele o float das alternativas
    também é um valor esperado, não uma entropia.
    """

    indice: int
    palavra: str
    entropia: float
    motivo: str
    e_candidata: bool
    alternativas: list[tuple[str, float, bool]] = field(default_factory=list)
    valor_esperado: float | None = None


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
        self.matriz = (
            carregar_matriz(lexico.sondas, lexico.palavras)
            if matriz is None
            else matriz
        )
        # Linhas da matriz = espaço de tentativa. Num motor montado com uma matriz
        # de recorte (os testes fazem isso), é ela que manda, não o léxico — daí o
        # prior de sondas ser realinhado a ela em vez de assumido do tamanho certo.
        self.n_sondas = self.matriz.shape[0]
        self.prior_sondas = _alinhar(lexico.prior_sondas, self.n_sondas)
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
        # A linha da matriz é o espaço de SONDA: uma conjugação jogada no modo
        # ampliado tem linha pronta e não cai no caminho lento abaixo.
        indice = self.lexico.indice_sonda.get(normalizar(tentativa))
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
        """Entropia (em bits) de cada sonda como próxima tentativa."""
        n = self.n_sondas
        m = len(candidatas)
        if m <= 1:
            return np.zeros(n, dtype=np.float64)

        pesos = self.lexico.prior[candidatas]
        soma = pesos.sum()
        pesos = pesos / soma if soma > 0 else np.full(m, 1.0 / m)

        if m <= LIMIAR_POUCAS:
            return self._entropias_poucas(candidatas, pesos)

        entropias = np.empty(n, dtype=np.float64)
        # Bloco escolhido para manter os temporários na casa de dezenas de MB.
        bloco = max(1, min(n, 2_000_000 // m))
        deslocamento = np.arange(bloco, dtype=np.int32) * N_PADROES
        # Alocado uma vez por chamada, não por bloco. Reaproveitar entre blocos é
        # seguro porque só se escreve onde baldes > 0, e nas outras posições o
        # produto é 0 · (lixo finito) = 0 — o log nunca produz inf/nan aqui.
        logs = np.zeros((bloco, N_PADROES), dtype=np.float64)

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
            vista = logs[:b]
            np.log2(baldes, out=vista, where=baldes > 0)
            entropias[i0:i1] = -(baldes * vista).sum(axis=1)

        return entropias

    def entropia_de(self, palpite: int, candidatas: np.ndarray) -> float:
        """Entropia (bits) de UMA sonda, sem varrer as outras 6.045.

        Mesma conta de `entropias`, numa linha só da matriz. Existe para anotar os
        bits de uma jogada imposta de fora (`abertura_padrao`), onde a varredura
        inteira seria desperdício.
        """
        m = len(candidatas)
        if m <= 1:
            return 0.0
        pesos = self.lexico.prior[candidatas]
        soma = pesos.sum()
        pesos = pesos / soma if soma > 0 else np.full(m, 1.0 / m)
        baldes = np.bincount(
            self.matriz[palpite, candidatas], weights=pesos, minlength=N_PADROES
        )
        positivos = baldes[baldes > 0]
        return float(-(positivos * np.log2(positivos)).sum())

    def _entropias_poucas(
        self, candidatas: np.ndarray, pesos: np.ndarray
    ) -> np.ndarray:
        """Mesma entropia, por agrupamento par a par em vez dos 243 baldes.

        Com poucas candidatas o caminho geral é quase todo desperdício: ele monta
        uma tabela (6.046 x 243) para preencher no máximo `m` colunas por linha —
        11 MB alocados e zerados para processar algumas dezenas de milhares de
        células úteis. Aqui a conta é reescrita por elemento:

            H = -Σ_padrão P·log₂P = -Σ_c w_c · log₂(massa do grupo de c)

        (cada grupo aparece uma vez por candidata que o compõe, e Σ_{c∈g} w_c = P_g).
        Custa n·m² em vez de n·243, sem tabela intermediária. Abaixo de
        `LIMIAR_POUCAS` isso é uma ordem de grandeza — e é justamente onde a busca
        do nível 3 passa a maior parte do tempo: 87% das varreduras dela são em
        conjuntos de até 8 candidatas.
        """
        sub = self.matriz[:, candidatas]  # (n, m) uint8
        massa = np.zeros(sub.shape, dtype=np.float64)
        for j in range(len(candidatas)):
            # Candidatas que caem no mesmo padrão que a j-ésima, linha a linha.
            massa += pesos[j] * (sub == sub[:, j : j + 1])
        # Prior degenerado pode zerar um grupo inteiro; lá w_c também é 0, então o
        # termo é 0 — trocar a massa por 1 mantém log₂ = 0 e evita 0·(-inf) = nan.
        segura = np.where(massa > 0, massa, 1.0)
        return -(pesos * np.log2(segura)).sum(axis=1)

    # --------------------------------------------------------------- escolha

    def melhor_candidata(self, candidatas: np.ndarray) -> int:
        """Candidata de maior prior (menor ICF)."""
        return int(candidatas[np.argmax(self.lexico.prior[candidatas])])

    def ordenar_por_entropia(
        self, entropias: np.ndarray, candidatas: np.ndarray
    ) -> np.ndarray:
        """Ordem de preferência: maior entropia, depois ser candidata, depois prior.

        O desempate por "também é candidata" é ganho grátis: entre duas palavras
        que separam igualmente bem, a que pode ser a resposta tem chance não-nula
        de encerrar o jogo (seção 4.5). É também o que segura o espaço ampliado no
        lugar: uma conjugação só é escolhida quando separa ESTRITAMENTE melhor que
        qualquer palavra do léxico — no empate ela perde nas duas chaves, porque
        não é candidata e tem prior 0.

        Pública porque o nível 3 (`termo/nivel3.py`) usa esta mesma ordem como
        "move ordering" da busca: os primeiros da lista são os candidatos a jogada
        ótima e os que mais ajudam a poda.
        """
        e_candidata = np.zeros(self.n_sondas, dtype=bool)
        e_candidata[candidatas] = True
        arredondada = np.round(entropias / EPSILON_EMPATE) * EPSILON_EMPATE
        # lexsort: a última chave é a primária.
        return np.lexsort((-self.prior_sondas, ~e_candidata, -arredondada))

    def escolher(
        self, candidatas: np.ndarray, tentativa: int = 1, n_alternativas: int = 3
    ) -> Sugestao:
        """Próxima jogada, aplicando a regra de endgame da seção 4.5."""
        m = len(candidatas)
        if m == 0:
            raise ValueError("conjunto de candidatas vazio")

        # O usuário vê a forma acentuada. Indexado pelo espaço de SONDA, que é o
        # de onde sai a jogada — as candidatas são o prefixo dele.
        palavras = self.lexico.sondas_exibicao

        if m == 1:
            i = int(candidatas[0])
            return Sugestao(i, palavras[i], 0.0, "única candidata restante", True)

        if m == 2:
            i = self.melhor_candidata(candidatas)
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
            i = self.melhor_candidata(candidatas)
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
        ordem = self.ordenar_por_entropia(entropias, candidatas)
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
            f"maior entropia entre as {self.n_sondas} palavras jogáveis "
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
        uso diário (a matriz inteira particionada contra as 6.046 candidatas).

        O espaço de tentativa entra na chave do cache porque é exatamente o que a
        conta varre: a abertura das 6.046 e a das 8.628 são respostas a perguntas
        diferentes e não podem se sobrescrever no arquivo. O sufixo só aparece no
        modo ampliado — assim as entradas já versionadas continuam valendo.
        """
        chave = f"T={self.lexico.temperatura}"
        if self.n_sondas != len(self.lexico):
            chave += f";S={self.n_sondas}"
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

    def abertura_padrao(self, n_alternativas: int = 5) -> Sugestao:
        """A abertura que as interfaces jogam: `ABERTURA_PADRAO`, em qualquer T.

        O ótimo continua sendo calculado — é dele que saem as alternativas, e é ele
        que responde quando a palavra fixa não está no léxico em uso (um recorte de
        teste, outra fonte de dados). O que muda é qual das duas vai na frente.
        """
        indice = self.lexico.indice_sonda.get(ABERTURA_PADRAO)
        otima = self.abertura(n_alternativas)
        if indice is None or indice >= self.n_sondas:
            return otima
        if otima.indice == indice:
            otima.motivo = MOTIVO_ABERTURA_PADRAO
            return otima

        # A ótima do critério vira a primeira alternativa: é a informação que ela
        # ainda carrega depois de perder a vez.
        alternativas = [(otima.palavra, otima.entropia, True)]
        alternativas += [
            alt for alt in otima.alternativas if normalizar(alt[0]) != ABERTURA_PADRAO
        ]
        palavra = self.lexico.sondas_exibicao[indice]
        return Sugestao(
            indice,
            palavra,
            self.entropia_de(indice, self.todas_candidatas()),
            MOTIVO_ABERTURA_PADRAO,
            indice < len(self.lexico),
            alternativas[:max(n_alternativas, 1)],
        )


def carregar_motor(
    temperatura: float = 1.0, ampliado: bool = False, **kwargs
) -> Motor:
    """Atalho: léxico + matriz + motor prontos para uso."""
    return Motor(Lexico.carregar(temperatura, ampliado), **kwargs)


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
