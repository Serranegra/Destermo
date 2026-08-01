"""Nível 3: minimiza diretamente o nº ESPERADO de tentativas (§4.1).

O nível 2 (`termo/entropia.py`) maximiza a informação de UMA jogada — é um proxy.
O objetivo real do jogo é outro: minimizar E[nº de tentativas até acertar]. O
3B1B publicou uma correção mostrando que os dois não coincidem, e a v1.1 deixou o
nível 3 fora de escopo por custo computacional (§4.1, §12).

Este módulo o implementa como uma camada ACIMA do nível 2, sem tocar nele:

    V(S, r) = min_g [ 1 + Σ_{padrão ≠ GGGGG} P(padrão | S) · V(S_padrão, r-1) ]
    V(S, 0) = PENALIDADE_DERROTA  acabaram as rodadas sem acertar

    P(padrão | S) = Σ_{c ∈ S_padrão} prior(c) / Σ_{c ∈ S} prior(c)

Daí saem os dois casos de contorno, sem nada de novo:

    V({w}, r≥1)   = 1                  basta chutar a única candidata
    V({a,b}, r≥2) = 1 + p_menor        a mais provável; o feedback revela a outra

O `r` (rodadas restantes) não é decoração. Sem ele o solver otimiza um jogo de
tentativas ilimitadas e cai na espiral clássica do Termo — "prima, urina, brida,
crica, criva": cinco candidatas prováveis que se distinguem por uma letra só. Cada
jogada dessas maximiza a chance de acertar AGORA e o valor esperado adora isso,
mas o Termo tem 6 rodadas e a partida se perde. Com o limite na recursão a busca
enxerga a parede e gasta uma jogada separando o grupo.

O prior é o mesmo do nível 2, então T continua sendo o dial contínuo de §4.2: com
T → ∞ os pesos ficam uniformes e V vira a contagem clássica de tentativas, sem
prior de frequência. Mas há uma diferença de fundo entre os dois níveis, e ela
aparece no que cada um faz com o prior:

  nível 2  usa o prior só para PESAR a entropia; a jogada é sempre a que mais
           separa, e as candidatas raras contam quase como as comuns
  nível 3  otimiza o valor esperado SOB o prior, então sacrifica candidatas raras
           de propósito quando isso baixa a média

Com T=1 o prior do projeto é agressivo — a candidata mais provável de um conjunto
de 50 carrega uns dois terços da massa —, e o nível 3 responde a isso chutando
candidatas prováveis em vez de sondas informativas. É a decisão certa para o
objetivo declarado: ele ganha na bateria realista (§5.3) e perde algumas palavras
raríssimas que o Termo não sortearia. Quem quiser o comportamento avesso a risco
tem o dial: T maior achata o prior e a busca volta a preferir sondas.

Quatro coisas tornam a recursão viável num léxico de 6.046 palavras:

  beam         só os K palpites de maior entropia entram na busca em cada nó. O
               nível 2 é um bom "move ordering": a jogada ótima quase sempre está
               no topo da lista de entropia. A candidata mais provável entra
               sempre, o que garante que todo nó tenha ao menos um palpite útil.
  profundidade abaixo de P níveis a recursão cai na política GULOSA do nível 2
               (beam de 1). Como uma política concreta é sempre um limite
               SUPERIOR de V, o valor só melhora quando P cresce — e P=0 com
               beam=1 reproduz o próprio nível 2. A busca é "anytime".
  memoização   estados iguais aparecem muitas vezes; a chave é o conjunto de
               candidatas mais (rodadas restantes, profundidade restante), que
               são o que muda o valor.
  poda         a soma parcial só cresce, então um palpite é abandonado no meio
               assim que passa do melhor custo já encontrado no nó.

A regra de endgame de §4.5 continua valendo e é delegada ao motor do nível 2: na
última tentativa só faz sentido chutar uma candidata. A busca chega à mesma
conclusão sozinha, mas delegar mantém a mensagem ao usuário idêntica nos dois
níveis — e é de graça.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import math
import textwrap

import numpy as np

from .entropia import (
    MOTIVO_ABERTURA_PADRAO,
    Motor,
    Sugestao,
    gravar_cache_json,
    ler_cache_json,
)
from .entropia import carregar_motor as carregar_nivel2
from .feedback import PADRAO_VITORIA
from .lexico import DIR_DADOS

ARQ_ABERTURA = DIR_DADOS / "aberturas_nivel3.json"

# Quantos palpites por nó entram na busca. 10 já cobre a jogada ótima na quase
# totalidade dos nós; o custo é linear em K e a qualidade satura rápido.
BEAM = 10

# Níveis abaixo da raiz que ainda usam o beam. 1 = filhos com busca, netos pela
# política gulosa. Cada +1 multiplica o custo por ~K.
PROFUNDIDADE = 1

# Quanto custa acabar as rodadas sem acertar, em "tentativas equivalentes". A
# média penalizada do benchmark (§5.4) cobra 7 numa derrota, o que equivale a 1
# aqui — barato demais: o solver trocaria a vitória por meia tentativa de ganho.
# 7 é "a partida inteira jogada em vão", e a essa altura qualquer valor acima de
# ~2 já basta para a busca nunca escolher perder.
PENALIDADE_DERROTA = 7.0

# Acima de quantas candidatas `escolher_com_teto` desiste da busca e responde o
# nível 2. É um parâmetro de INTERFACE, não do solver — ver o método para a
# medição que fixa o valor. `escolher` continua sem teto nenhum.
TETO_INTERATIVO = 250

# Quantos estados a memoização da busca guarda antes de ser zerada, no caminho
# interativo. Os caches do motor foram desenhados para um PROCESSO CURTO — uma
# bateria do benchmark, uma partida na CLI —, onde crescer sem limite é o certo:
# o processo morre no fim e o teto de 500.000 de `_palpites` nunca é alcançado.
# Servindo uma página, o mesmo motor vive enquanto o servidor viver, e aí crescer
# sem limite tem outro nome. Medido com aberturas variadas (o pior caso para a
# diversidade de estados), o par `_memo`+`_cache_palpites` cresce ~12 MB a cada
# 100 partidas — o que num dia de tráfego passa de gigabyte.
#
# Zerar por inteiro, e não despejar o mais antigo: manter idade por entrada
# custaria mais que o cache economiza, e o que se perde é recalculável por
# definição. 30.000 entradas são ~21 MB, e as poucas dezenas de estados quentes
# de um dia (todo mundo joga a MESMA secreta) voltam ao cache na primeira
# consulta depois da limpeza.
TETO_MEMO = 30_000


def _sem_docstring(arvore: ast.AST) -> ast.AST:
    """Tira as docstrings da árvore — texto não muda o que a busca decide."""
    for no in ast.walk(arvore):
        corpo = getattr(no, "body", None)
        if not isinstance(corpo, list) or not corpo:
            continue
        primeiro = corpo[0]
        if (
            isinstance(primeiro, ast.Expr)
            and isinstance(primeiro.value, ast.Constant)
            and isinstance(primeiro.value.value, str)
        ):
            no.body = corpo[1:] or [ast.Pass()]
    return arvore


@functools.cache
def assinatura_busca() -> str:
    """Hash do código que decide o valor da busca (§0.3: cache derivado obsoleto).

    A abertura do nível 3 custa ~9 min e vai versionada em disco. O risco é o da
    §0.3 da especificação — um cache derivado sobreviver a uma mudança que o
    invalida —, e aconteceu de verdade durante o desenvolvimento: a primeira
    abertura foi calculada sob o objetivo sem limite de rodadas e continuou
    parecendo válida depois. A matriz não tem esse problema porque a assinatura
    dela é sobre a entrada; aqui a "entrada" inclui o algoritmo, então é ele que
    entra no hash.

    Comparar por AST, sem docstrings, é o que separa "mudei a regra" de "mudei o
    texto": reformatar, quebrar linha diferente ou reescrever a explicação não
    custa 9 min de recálculo; mexer na regra do beam, na poda ou na fórmula do
    custo, sim. Renomear uma variável TAMBÉM invalida — `ast.unparse` preserva
    identificadores —, o que é falso positivo, mas do lado seguro: recalcular à
    toa custa tempo, publicar uma abertura errada custa credibilidade.

    O que fica de fora, de propósito: `Motor.entropias` e `_entropias_poucas`. Elas
    têm oráculo — os testes comparam com uma referência escrita à mão e com o
    outro caminho —, então uma mudança de valor lá é ruidosa, não silenciosa. E
    são justamente as funções que se mexe por desempenho: pôr no hash cobraria os
    9 min a cada otimização que não muda resultado nenhum. Hash para o que não tem
    oráculo, teste para o que tem.
    """
    alvos = (
        MotorNivel3._particionar,
        MotorNivel3._palpites,
        MotorNivel3._pesos,
        MotorNivel3._custo,
        MotorNivel3._valor,
        MotorNivel3.escolher,
        Motor.ordenar_por_entropia,
        Motor.melhor_candidata,
    )
    partes = [
        ast.unparse(_sem_docstring(ast.parse(textwrap.dedent(inspect.getsource(f)))))
        for f in alvos
    ]
    return hashlib.sha256("\n".join(partes).encode("utf-8")).hexdigest()[:12]


class MotorNivel3:
    """Busca na árvore de decisão com o objetivo real: E[nº de tentativas].

    Envolve um `Motor` do nível 2 e reaproveita tudo dele — matriz de padrões,
    prior, entropias, filtragem e a regra de endgame. A interface pública é a
    mesma (`abertura`, `escolher`, `escolher_com_cache`), então quem consome o
    solver não precisa saber com qual nível está falando.
    """

    def __init__(
        self,
        motor: Motor,
        beam: int = BEAM,
        profundidade: int = PROFUNDIDADE,
    ):
        if beam < 1:
            raise ValueError("beam deve ser >= 1")
        # O teto é folgado de propósito: uma partida de Termo tem 6 rodadas, então
        # nada além de ~6 muda o resultado. Vale só para recusar cedo, com uma
        # mensagem legível, quem passar um número absurdo.
        if not 0 <= profundidade <= 64:
            raise ValueError("profundidade deve estar entre 0 e 64")
        self.motor = motor
        self.beam = beam
        self.profundidade = profundidade
        self._memo: dict[bytes, float] = {}
        self._cache_palpites: dict[bytes, np.ndarray] = {}
        self._cache_escolha: dict[bytes, Sugestao] = {}

    # ------------------------------------------------------------- delegação

    @property
    def lexico(self):
        return self.motor.lexico

    @property
    def matriz(self) -> np.ndarray:
        return self.motor.matriz

    def todas_candidatas(self) -> np.ndarray:
        return self.motor.todas_candidatas()

    def filtrar(
        self, candidatas: np.ndarray, tentativa: str, codigo: int
    ) -> np.ndarray:
        return self.motor.filtrar(candidatas, tentativa, codigo)

    # -------------------------------------------------------------- partição

    def _particionar(
        self, palpite: int, candidatas: np.ndarray
    ) -> list[tuple[int, np.ndarray]]:
        """Agrupa as candidatas pelo padrão que `palpite` produziria em cada uma.

        Uma linha da matriz + um argsort estável: os sub-conjuntos saem na mesma
        ordem crescente de índice que entrou, o que mantém a chave da memoização
        canônica sem precisar reordenar nada.
        """
        codigos = self.matriz[palpite, candidatas]
        ordem = np.argsort(codigos, kind="stable")
        ordenadas = candidatas[ordem]
        valores = codigos[ordem]
        cortes = np.flatnonzero(valores[1:] != valores[:-1]) + 1
        inicios = np.concatenate(([0], cortes))
        return list(zip(valores[inicios].tolist(), np.split(ordenadas, cortes)))

    def _palpites(
        self, candidatas: np.ndarray, k: int, mais_provavel: bool = True
    ) -> np.ndarray:
        """Índices dos k palpites a testar neste nó.

        Nos nós de busca a candidata de maior prior entra mesmo fora do top-k: com
        o prior de T=1 ela costuma ser a jogada ótima e nem sempre está entre as
        de maior entropia — foi por isso que 3B1B teve que publicar a correção.

        Na cauda gulosa ela NÃO entra: lá o beam de 1 tem que ser exatamente a
        política do nível 2, senão o valor deixa de ser comparável com ela.

        O resultado é memoizado à parte do valor: a ordem depende só do conjunto,
        enquanto o valor depende também das rodadas e da profundidade restantes.
        Sem este cache, o mesmo conjunto reaparecendo noutra rodada refaz a
        varredura do léxico inteiro — que é 95% do custo da busca.

        O `.copy()` da fatia não é zelo: `ordenar_por_entropia` devolve a ordem
        das 6.046 (ou 8.629) sondas, e `[:k]` é uma VIEW que mantém esse array de
        48 KB vivo enquanto estiver no cache. Guardar a view custava 48 KB por
        entrada para armazenar dez inteiros — medido, 9.874 das 11.823 entradas
        de uma bateria de 30 partidas retinham a base, 478 MB dos 504 MB que o
        processo crescia. Com a cópia, a mesma bateria cresce 6 MB. Num motor de
        `cache_resource`, que vive enquanto o servidor viver, a diferença é entre
        um cache e um vazamento.
        """
        chave = (
            k.to_bytes(2, "little")
            + bytes((mais_provavel,))
            + candidatas.astype(np.int32).tobytes()
        )
        guardado = self._cache_palpites.get(chave)
        if guardado is not None:
            return guardado

        entropias = self.motor.entropias(candidatas)
        ordem = self.motor.ordenar_por_entropia(entropias, candidatas)[:k].copy()
        if mais_provavel:
            melhor = self.motor.melhor_candidata(candidatas)
            if melhor not in ordem:
                ordem = np.append(ordem, melhor)
        if len(self._cache_palpites) < 500_000:
            self._cache_palpites[chave] = ordem
        return ordem

    # ----------------------------------------------------------------- valor

    def _pesos(self, candidatas: np.ndarray) -> tuple[np.ndarray, float]:
        """(pesos alinhados a `candidatas`, soma deles).

        Com T minúsculo o softmax do prior chega a zerar tudo que não é o topo, e
        um conjunto inteiro pode somar 0. Aí cai no uniforme, que é o que "sem
        informação de frequência" significa — é o mesmo que `Motor.entropias` faz.
        """
        pesos = self.lexico.prior[candidatas]
        soma = float(pesos.sum())
        if soma <= 0.0:
            pesos = np.ones(len(candidatas), dtype=np.float64)
            soma = float(len(candidatas))
        return pesos, soma

    def _custo(
        self,
        palpite: int,
        candidatas: np.ndarray,
        rodadas: int,
        restante: int,
        limite: float = math.inf,
    ) -> float:
        """E[tentativas] de jogar `palpite` agora e seguir jogando bem depois.

        `rodadas` é o orçamento DEPOIS desta jogada. Devolve `inf` se o palpite
        não separa nada (todas as candidatas cairiam no mesmo padrão — informação
        zero) ou se a soma parcial já passou de `limite`, caso em que não vale
        terminar a conta.
        """
        baldes = self._particionar(palpite, candidatas)
        if len(baldes) == 1:
            return math.inf

        prior = self.lexico.prior
        soma = float(prior[candidatas].sum())
        uniforme = soma <= 0.0  # prior degenerado; ver `_pesos`
        acumulado = 0.0
        for codigo, sub in baldes:
            if codigo == PADRAO_VITORIA:
                continue  # acertou nesta jogada: nenhuma tentativa adicional
            if uniforme:
                probabilidade = len(sub) / len(candidatas)
            else:
                probabilidade = float(prior[sub].sum()) / soma
            acumulado += probabilidade * self._valor(sub, rodadas, restante)
            if 1.0 + acumulado >= limite:
                return math.inf
        return 1.0 + acumulado

    def _valor(self, candidatas: np.ndarray, rodadas: int, restante: int) -> float:
        """V(S, r): tentativas esperadas jogando bem, com `rodadas` disponíveis."""
        if rodadas <= 0:
            return PENALIDADE_DERROTA  # acabaram as tentativas e não acertou

        m = len(candidatas)
        if m == 1:
            return 1.0
        if m == 2:
            # Chutar a mais provável é ótimo aqui, e por isso o caso é exato: um
            # palpite de fora custaria 2 cheios, e este custa 1 + p_menor < 2. Com
            # uma rodada só, errar já é derrota — e é o mesmo 1 + p_menor · V(·, 0).
            pesos, soma = self._pesos(candidatas)
            perde = 1.0 if rodadas >= 2 else PENALIDADE_DERROTA
            return 1.0 + perde * float(pesos.min()) / soma

        # Dois bytes por campo, não um: `rodadas` vem de `n_max_tentativas`, que é
        # parâmetro do motor, e um `Motor(n_max_tentativas=300)` derrubava a busca
        # com "bytes must be in range(0, 256)" — erro que não diz nada a quem o vê.
        chave = (
            rodadas.to_bytes(2, "little")
            + restante.to_bytes(2, "little")
            + candidatas.astype(np.int32).tobytes()
        )
        guardado = self._memo.get(chave)
        if guardado is not None:
            return guardado

        # Acabou o orçamento de busca: segue a política gulosa do nível 2, que é
        # uma política concreta e portanto um limite superior honesto de V. Cada
        # nível a mais só pode baixar o valor, porque o beam contém o palpite que
        # a política gulosa escolheria.
        busca = restante > 0
        palpites = self._palpites(
            candidatas, self.beam if busca else 1, mais_provavel=busca
        )

        melhor = math.inf
        for palpite in palpites:
            custo = self._custo(
                int(palpite), candidatas, rodadas - 1, max(restante - 1, 0), melhor
            )
            if custo < melhor:
                melhor = custo
        # `melhor` é finito: `palpites[0]` é o de maior entropia do léxico, e com
        # m >= 3 candidatas existe palpite que separa (cada candidata separa a si
        # mesma), então a entropia máxima é > 0 e o palpite não é descartado.
        self._memo[chave] = melhor
        return melhor

    def valor(
        self,
        candidatas: np.ndarray,
        rodadas: int | None = None,
        profundidade: int | None = None,
    ) -> float:
        """V(S, r) com a profundidade de busca configurada (ou uma pontual).

        `rodadas` é o orçamento a partir desta jogada, inclusive; o padrão são as
        6 do Termo, isto é, o começo de uma partida.
        """
        if len(candidatas) == 0:
            raise ValueError("conjunto de candidatas vazio")
        rodadas = self.motor.n_max_tentativas if rodadas is None else rodadas
        profundidade = self.profundidade if profundidade is None else profundidade
        return self._valor(np.asarray(candidatas), rodadas, profundidade + 1)

    def valor_guloso(self, candidatas: np.ndarray, rodadas: int | None = None) -> float:
        """V(S, r) seguindo a política de entropia do nível 2 — a régua a bater.

        Mesmo objetivo, mesma penalidade de derrota, medidos sobre a política
        gulosa. `valor()` nunca fica acima disto.
        """
        return self.valor(candidatas, rodadas, profundidade=-1)

    def valor_com_abertura(
        self, palpite: int, candidatas: np.ndarray, rodadas: int | None = None
    ) -> float:
        """E[tentativas] jogando `palpite` agora e seguindo o nível 2 depois.

        Complemento de `valor_guloso`, que deixa a própria política escolher a
        primeira jogada. Aqui ela é imposta, que é o que permite comparar
        aberturas ENTRE SI sob a mesma política e o mesmo prior — e é exato, não
        uma média de simulação: a esperança é sobre o prior, palavra a palavra.
        """
        rodadas = self.motor.n_max_tentativas if rodadas is None else rodadas
        return self._custo(int(palpite), np.asarray(candidatas), rodadas - 1, 0)

    # --------------------------------------------------------------- escolha

    def escolher(
        self, candidatas: np.ndarray, tentativa: int = 1, n_alternativas: int = 3
    ) -> Sugestao:
        """Próxima jogada segundo o objetivo do nível 3.

        Os casos de contorno (1 ou 2 candidatas) e a regra de endgame (§4.5) são
        do motor do nível 2 — lá o nível 3 não tem nada a acrescentar: a resposta
        é a mesma e as mensagens ficam consistentes entre os dois níveis.
        """
        m = len(candidatas)
        if m == 0:
            raise ValueError("conjunto de candidatas vazio")

        # Rodadas ainda disponíveis, contando esta. É o que impede a busca de
        # otimizar um jogo sem parede (ver o cabeçalho do módulo).
        rodadas = max(self.motor.n_max_tentativas - tentativa + 1, 1)

        if m <= 2 or tentativa >= self.motor.n_max_tentativas or (
            tentativa == self.motor.n_max_tentativas - 1
            and m <= self.motor.limiar_endgame
        ):
            sugestao = self.motor.escolher(candidatas, tentativa, n_alternativas)
            if m <= 2:  # base exata da recursão, vale anotar o valor
                sugestao.valor_esperado = self.valor(candidatas, rodadas)
            return sugestao

        palavras = self.lexico.sondas_exibicao  # a jogada sai do espaço de sonda
        palpites = self._palpites(candidatas, self.beam)
        # Uma varredura a mais, só na raiz, para relatar os bits da jogada
        # escolhida — irrelevante perto das centenas de nós da árvore.
        entropias = self.motor.entropias(candidatas)

        # Sem poda na raiz: são só K palpites e queremos o custo de todos para
        # poder mostrar as alternativas com o valor de cada uma.
        custos = [
            (
                self._custo(int(g), candidatas, rodadas - 1, self.profundidade),
                int(g),
            )
            for g in palpites
        ]
        custos = [(custo, g) for custo, g in custos if math.isfinite(custo)]
        custos.sort(key=lambda par: par[0])
        if not custos:
            # Nenhum palpite do beam separa nada. Não deveria acontecer com m >= 3
            # (ver `_valor`), mas é a fronteira pública do módulo: melhor devolver
            # a jogada do nível 2 do que estourar um IndexError na cara do usuário.
            return self.motor.escolher(candidatas, tentativa, n_alternativas)

        conjunto = set(candidatas.tolist())
        valor, melhor = custos[0]
        alternativas = [
            (palavras[g], custo, g in conjunto)
            for custo, g in custos[1 : n_alternativas + 1]
        ]
        return Sugestao(
            melhor,
            palavras[melhor],
            float(entropias[melhor]),
            f"menor nº esperado de tentativas (E={valor:.3f}) entre os "
            f"{len(palpites)} melhores palpites por entropia "
            f"({m} candidatas, {rodadas} rodadas, profundidade {self.profundidade})",
            melhor in conjunto,
            alternativas,
            valor_esperado=valor,
        )

    def escolher_com_teto(
        self,
        candidatas: np.ndarray,
        tentativa: int = 1,
        teto: int = TETO_INTERATIVO,
        n_alternativas: int = 3,
    ) -> Sugestao:
        """`escolher`, mas com um limite de tempo de resposta em vez de nenhum.

        Acima de `teto` candidatas a busca é abandonada e responde o nível 2. A
        troca é assimétrica, e é por isso que ela vale a pena: o custo da busca
        cresce com o tamanho do conjunto, e a vantagem dela ENCOLHE. Medido sobre
        os 131 estados possíveis da rodada 2 (a abertura é fixa, então o espaço é
        enumerável), com o E[tentativas] exato de cada jogada sob o prior:

            teto   delegados   prob.   pior latência   perda em E[tentativas]
            sem         0       0,0%       12,13 s            0
            400         2      13,2%        5,33 s            0,0068
            250         5      22,8%        3,43 s            0,0072
            150         9      29,1%        2,36 s            0,0126
             60        25      56,5%        0,46 s            0,1020

        Ou seja: em 250 se abre mão de 0,007 tentativa — 1,3% da vantagem de 0,54
        que o nível 3 tem sobre o nível 2 — para cortar o pior caso em 3,5x.

        E o teto não é só sobre conforto. Sem ele, uma jogada que quase não separa
        (uma palavra fora do léxico marcada toda preta, que a interface aceita)
        deixa ~5.800 candidatas, e aí a raiz da busca é a mesma da abertura: mais
        de dez minutos presos numa thread do servidor. Numa página aberta ao
        público isso é um travamento a um clique de distância.

        Não é o padrão de `escolher` de propósito. O nível 3 sem teto é o que o
        benchmark mede e o que o README documenta; onde vale esperar (a CLI, uma
        bateria), o método a chamar continua sendo `escolher`. O teto é a escolha
        de quem tem alguém do outro lado esperando, e por isso a interface é que o
        pede — não `termo/` que o impõe.

        É também aqui que a memoização é aparada (`TETO_MEMO`), pelo mesmo
        motivo: só o caminho interativo tem um motor que vive para sempre.
        """
        if len(self._memo) > TETO_MEMO:
            self._memo.clear()
            self._cache_palpites.clear()
        if teto is not None and len(candidatas) > teto:
            return self.motor.escolher(candidatas, tentativa, n_alternativas)
        return self.escolher(candidatas, tentativa, n_alternativas)

    def escolher_com_cache(self, candidatas: np.ndarray, tentativa: int) -> Sugestao:
        """`escolher` memoizado pelo conjunto de candidatas — usado no benchmark.

        Mesma ideia do nível 2, e aqui pesa muito mais: cada escolha custa uma
        árvore inteira, não uma varredura de entropia. A rodada entra na chave
        sempre — aqui ela é parte do estado, não só um detalhe de endgame.
        """
        chave = tentativa.to_bytes(1, "little") + candidatas.astype(np.int32).tobytes()
        resultado = self._cache_escolha.get(chave)
        if resultado is None:
            resultado = self.escolher(candidatas, tentativa, n_alternativas=0)
            if len(self._cache_escolha) < 500_000:
                self._cache_escolha[chave] = resultado
        return resultado

    # -------------------------------------------------------------- abertura

    def _chave_cache(self) -> str:
        """Tudo que muda a resposta entra na chave.

        Além da configuração da busca, a penalidade de derrota e o nº de rodadas
        definem o objetivo: mexer neles muda o valor de toda a árvore, e uma
        abertura calculada sob o objetivo antigo não vale mais nada. O `B` fecha o
        resto — é a assinatura do próprio algoritmo (ver `assinatura_busca`).

        O espaço de tentativa entra pelo mesmo motivo, e só no modo ampliado: as
        entradas já versionadas custaram 16 min cada, no custo da época, e não
        podem ser invalidadas por uma opção que ninguém pediu.
        """
        sondas = (
            "" if self.motor.n_sondas == len(self.lexico)
            else f";S={self.motor.n_sondas}"
        )
        return (
            f"T={self.lexico.temperatura};K={self.beam};P={self.profundidade}"
            f";D={PENALIDADE_DERROTA:g};R={self.motor.n_max_tentativas}"
            f";B={assinatura_busca()}{sondas}"
        )

    def _abertura_guardada(self) -> tuple[dict, dict | None]:
        """(cache inteiro, entrada desta configuração se ainda válida)."""
        cache = ler_cache_json(ARQ_ABERTURA)
        guardada = cache.get(self._chave_cache())
        if guardada and guardada.get("n") == len(self.lexico):
            return cache, guardada
        return cache, None

    def abertura_em_cache(self) -> bool:
        """A abertura desta configuração já está no disco?

        Quem chama `abertura()` sem cache espera minutos, então vale avisar antes.
        """
        return self._abertura_guardada()[1] is not None

    def abertura(self, n_alternativas: int = 5, usar_cache: bool = True) -> Sugestao:
        """Melhor primeira jogada pelo objetivo do nível 3.

        É de longe a conta mais cara do projeto: a raiz tem as 6.046 candidatas e
        cada palpite do beam abre ~150 sub-árvores — 9 min de busca (eram 16 antes
        de `_entropias_poucas`). Vai para o disco em `data/aberturas_nivel3.json`,
        com a chave de `_chave_cache`.
        """
        cache, guardada = self._abertura_guardada() if usar_cache else ({}, None)
        if guardada is not None:
            return Sugestao(
                self.lexico.indice_de(guardada["palavra"]),
                guardada["palavra"],
                guardada["entropia"],
                "melhor abertura do nível 3 (em cache)",
                True,
                [tuple(alt) for alt in guardada["alternativas"]],
                valor_esperado=guardada["valor_esperado"],
            )

        sugestao = self.escolher(
            self.todas_candidatas(), tentativa=1,
            n_alternativas=max(n_alternativas, 5),
        )
        sugestao.motivo = (
            f"melhor abertura do nível 3 "
            f"(E={sugestao.valor_esperado:.3f} tentativas)"
        )
        if usar_cache:
            cache[self._chave_cache()] = {
                "n": len(self.lexico),
                "palavra": sugestao.palavra,
                "entropia": sugestao.entropia,
                "valor_esperado": sugestao.valor_esperado,
                "alternativas": [list(alt) for alt in sugestao.alternativas],
            }
            gravar_cache_json(ARQ_ABERTURA, cache)
        return sugestao

    def abertura_padrao(self, n_alternativas: int = 5) -> Sugestao:
        """`ABERTURA_PADRAO` também aqui, com o E[tentativas] dela anotado.

        A palavra é a mesma dos dois motores — o nível 3 não tem por que abrir
        diferente do nível 2 numa escolha que vale ≤ 0,07 tentativa e que o usuário
        digita todo dia. O que ele acrescenta é o número: o E da abertura fixa sai
        da lista de alternativas da abertura ótima, que já é uma tabela de
        E[tentativas] por abertura sob esta mesma política. Sem cache dela (T fora
        das versionadas), a sugestão sai sem o E em vez de custar ~9 min de busca.

        Só a primeira jogada muda: da segunda em diante é `escolher`, igual.
        """
        sugestao = self.motor.abertura_padrao(n_alternativas)
        if sugestao.motivo != MOTIVO_ABERTURA_PADRAO:
            # Léxico sem a palavra fixa. O nível 2 respondeu com o ótimo DELE; aqui
            # o ótimo é outro, e é o nosso que vale.
            return self.abertura(n_alternativas)
        _, guardada = self._abertura_guardada()
        if guardada is None:
            return sugestao

        tabela = [(guardada["palavra"], guardada["valor_esperado"], True)]
        tabela += [tuple(alt) for alt in guardada["alternativas"]]
        sugestao.valor_esperado = dict(
            (palavra, valor) for palavra, valor, _ in tabela
        ).get(sugestao.palavra)
        if sugestao.valor_esperado is not None:
            # Com o E anotado, as alternativas passam a ser as da tabela — o float
            # delas tem que estar na mesma moeda do campo principal (§ `Sugestao`),
            # e as que vieram do nível 2 estão em bits. A ótima preterida encabeça
            # a lista: é a informação que ela ainda carrega depois de perder a vez.
            sugestao.alternativas = [
                alt for alt in tabela if alt[0] != sugestao.palavra
            ][:n_alternativas]
        return sugestao


def carregar_motor(
    temperatura: float = 1.0,
    beam: int = BEAM,
    profundidade: int = PROFUNDIDADE,
    **kwargs,
) -> MotorNivel3:
    """Atalho: léxico + matriz + motor do nível 3 prontos para uso."""
    return MotorNivel3(carregar_nivel2(temperatura, **kwargs), beam, profundidade)


if __name__ == "__main__":
    import sys
    import time

    sys.stdout.reconfigure(encoding="utf-8")
    temperatura = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    motor3 = carregar_motor(temperatura)
    motor2 = motor3.motor

    # Um estado real de meio de jogo: abertura do nível 2 com feedback BBBBB.
    abertura2 = motor2.abertura()
    candidatas = motor2.filtrar(motor2.todas_candidatas(), abertura2.palavra, 0)
    print(f"após '{abertura2.palavra}' + BBBBB: {len(candidatas)} candidatas")

    inicio = time.perf_counter()
    guloso = motor3.valor_guloso(candidatas)
    print(f"  V da política gulosa (nível 2): {guloso:.4f} "
          f"[{time.perf_counter() - inicio:.1f}s]")

    inicio = time.perf_counter()
    sugestao = motor3.escolher(candidatas, tentativa=2)
    print(f"  nível 3 joga '{sugestao.palavra}': E={sugestao.valor_esperado:.4f} "
          f"[{time.perf_counter() - inicio:.1f}s]")
    print(f"  nível 2 joga '{motor2.escolher(candidatas, 2).palavra}'")
    for palavra, valor, e_cand in sugestao.alternativas:
        print(f"     {palavra}  E={valor:.4f}{'  (candidata)' if e_cand else ''}")
