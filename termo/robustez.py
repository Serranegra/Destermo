"""Qual abertura escolher sem saber de que distribuição as secretas saem.

O projeto inteiro — níveis 1, 2 e 3 — responde à pergunta "qual é a melhor
abertura?" depois de fixar um prior: softmax(-ICF/T) com T=1. Este módulo tira
essa premissa do lugar de dado e a põe no lugar de incógnita.

O eixo é o NÚMERO EFETIVO DE PALAVRAS:

    M(prior) = 2 ** H(prior)          H em bits

M responde "quantas palavras esse prior de fato considera possíveis". Uniforme
sobre K palavras dá exatamente K, e é a única leitura de "concentração" que não
depende da forma da distribuição — o que é justamente a hipótese que a §3 põe
à prova.

O critério é minimax de arrependimento (Savage): para cada abertura candidata,
o pior que ela faz em relação à melhor daquele mundo; vence quem tem o pior caso
menos ruim. Não é o mesmo que "melhor em média sobre os mundos" — minimax não
supõe que se saiba quanto cada mundo é provável, o que é exatamente a situação.

RESSALVA QUE NÃO É DETALHE: minimax de arrependimento é refém do conjunto de
mundos admitidos. Um M implausível na grade domina o máximo e sequestra a
resposta sozinho. O piso M=300 daqui é uma ESCOLHA, não um dado — a de que o
Termo não sorteia de uma lista mais curta que isso. `piso_empirico` existe para
trocar essa escolha por medição assim que houver um registro das secretas já
sorteadas: a mais rara delas obriga a lista a ter ao menos aquele tamanho.

Tudo em nível 2. O nível 3 é ~47x mais CPU e inviabilizaria a grade; a vencedora
é levada ao nível 3 só no fim, para reportar o número dela.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import math
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .entropia import Motor
from .lexico import DIR_DADOS, Lexico, calcular_prior
from .nivel3 import BEAM, PROFUNDIDADE, MotorNivel3, _sem_docstring

ARQ_CACHE = DIR_DADOS / "robustez.json"

# Registro das secretas já sorteadas pelo Termo, uma por linha. Não vem no
# repositório; se aparecer, vira o piso empírico de M (ver `piso_empirico`).
ARQ_SORTEADAS = DIR_DADOS / "secretas_sorteadas.txt"

# Pontos da grade, espaçados em log. O topo é o léxico inteiro (uniforme) e o
# piso é a escolha discutida no cabeçalho.
PONTOS_M = [300, 600, 1200, 2400, 4800, 6046]

# Rótulos de leitura. O do Wordle vem da proporção do jogo original: 2.315
# respostas para 12.972 palpites válidos, ~18% — que aqui daria ~1.090, e 1200 é
# o ponto da grade mais próximo disso.
ROTULOS = {
    300: "lista bem curada",
    1200: "estilo Wordle",
    6046: "não sei nada",
}
ROTULO_PADRAO = "prior ICF, padrão do projeto"

# Três formas diferentes com a mesma concentração, para o teste de colapso (§3).
M_COLAPSO = 1200
CORTES_COLAPSO = [6046, 3000, 1500]


# --------------------------------------------------------------- perplexidade


def perplexidade(prior: np.ndarray) -> float:
    """M = 2**H: quantas palavras o prior efetivamente considera possíveis.

    Uniforme sobre K palavras dá exatamente K — é o que dá sentido à unidade e o
    que o teste fixa. Peso zero não contribui: 0·log0 = 0 no limite.
    """
    massa = np.asarray(prior, dtype=np.float64)
    massa = massa[massa > 0]
    if massa.size == 0:
        raise ValueError("prior sem massa positiva")
    entropia = -(massa * np.log2(massa)).sum()
    return float(2.0**entropia)


def prior_familia(icf: np.ndarray, n: int, temperatura: float) -> np.ndarray:
    """prior_{N,T}: softmax(-ICF/T) entre as N mais comuns, zero no resto.

    As duas alavancas fazem coisas diferentes com a mesma aparência: N é um corte
    DURO na lista de respostas, T é um peso CONTÍNUO sobre quem sobrou. Elas
    podem produzir o mesmo M por caminhos distintos, e é disso que trata a §3.
    """
    prior = np.zeros(icf.size, dtype=np.float64)
    indices = np.argsort(icf, kind="stable")[:n]
    prior[indices] = calcular_prior(icf[indices], temperatura)
    return prior


def temperatura_para(
    icf: np.ndarray, alvo: float, n: int | None = None, tolerancia: float = 1e-4
) -> float:
    """Busca binária do T que dá perplexidade `alvo` com as `n` mais comuns.

    M cresce monotonicamente com T — de 1 (T→0, toda a massa na palavra mais
    comum) até n (T→∞, uniforme) —, então a bisseção é bem posta. Alvo igual ou
    acima do teto devolve infinito, que é o uniforme exato e não uma aproximação.
    """
    n = icf.size if n is None else n
    if alvo >= n:
        return math.inf
    if alvo <= 1.0:
        raise ValueError("M alvo tem que ser > 1")

    baixo, alto = 1e-6, 1.0
    while perplexidade(prior_familia(icf, n, alto)) < alvo:
        alto *= 2.0
        if alto > 1e9:  # inalcançável na prática; o teto é o uniforme
            return math.inf

    for _ in range(200):
        meio = math.sqrt(baixo * alto)  # bisseção em log: T varre ordens de grandeza
        if perplexidade(prior_familia(icf, n, meio)) < alvo:
            baixo = meio
        else:
            alto = meio
        if alto / baixo - 1.0 < tolerancia:
            break
    return math.sqrt(baixo * alto)


def piso_empirico(lexico: Lexico, caminho: Path = ARQ_SORTEADAS) -> dict | None:
    """Piso de M medido, se houver registro das secretas já sorteadas.

    A palavra mais rara já sorteada obriga a lista de respostas a ter ao menos a
    posição dela em frequência — abaixo disso o mundo é incompatível com o que o
    jogo já fez. Isso troca o piso ESCOLHIDO da grade por um piso MEDIDO, que é a
    única forma honesta de tirar o minimax da mão de quem monta a grade.

    Devolve None quando o arquivo não existe, que é o caso hoje.
    """
    if not caminho.exists():
        return None
    sorteadas = [
        linha.strip() for linha in caminho.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]
    posicao = {int(j): k + 1 for k, j in enumerate(np.argsort(lexico.icf, kind="stable"))}
    conhecidas, ausentes = [], []
    for palavra in sorteadas:
        try:
            conhecidas.append((posicao[lexico.indice_de(palavra)], palavra))
        except KeyError:
            ausentes.append(palavra)
    if not conhecidas:
        return None
    rank, palavra = max(conhecidas)
    return {
        "piso": rank,
        "palavra_mais_rara": palavra,
        "n_sorteadas": len(conhecidas),
        "fora_do_lexico": ausentes,
    }


# ------------------------------------------------------------------- cenários


@dataclass
class Cenario:
    """Um mundo da grade: uma distribuição de onde as secretas sairiam."""

    m: float
    n: int
    temperatura: float
    rotulo: str
    prior: np.ndarray = field(repr=False)

    @property
    def descricao(self) -> str:
        t = "∞" if math.isinf(self.temperatura) else f"{self.temperatura:.3g}"
        return f"M={self.m:.0f} (N={self.n}, T={t})"


def montar_grade(lexico: Lexico, pontos: list[int] | None = None) -> list[Cenario]:
    """Um cenário por ponto de M, mais o do prior padrão do projeto.

    Todos com N = léxico inteiro: o corte fica de fora porque a §3 testa se ele
    muda alguma coisa além de M. O ponto do T=1 entra mesmo caindo entre dois
    pontos da grade — é a posição atual do projeto, e o relatório precisa dela
    para dizer quanto se perde ao supor o mundo errado.
    """
    pontos = PONTOS_M if pontos is None else pontos
    n = len(lexico)
    cenarios = []
    for alvo in pontos:
        temperatura = temperatura_para(lexico.icf, alvo, n)
        prior = prior_familia(lexico.icf, n, temperatura)
        cenarios.append(Cenario(perplexidade(prior), n, temperatura,
                                ROTULOS.get(alvo, ""), prior))

    padrao = prior_familia(lexico.icf, n, 1.0)
    cenarios.append(Cenario(perplexidade(padrao), n, 1.0, ROTULO_PADRAO, padrao))
    cenarios.sort(key=lambda c: c.m)
    return cenarios


def com_prior(lexico: Lexico, prior: np.ndarray) -> Lexico:
    """Cópia do léxico com outro prior — os arrays pesados seguem compartilhados.

    Trocar `lexico.prior` no lugar seria mais curto e é armadilha: o motor e a
    busca guardam referências ao léxico, e um cenário vazaria para o seguinte
    pela porta dos fundos.
    """
    return replace(lexico, temperatura=math.nan, prior=prior)


def abertura_nivel2(motor: Motor, prior: np.ndarray) -> tuple[int, float]:
    """A abertura ótima do nível 2 sob `prior`, e a entropia dela.

    É a escolha do nível 2 e nada mais: maior entropia sobre o léxico inteiro,
    com o desempate de `ordenar_por_entropia`. É esta palavra que serve de
    referência da coluna no arrependimento — não o mínimo do conjunto de
    candidatas, que subestimaria o arrependimento de todo mundo.
    """
    motor = Motor(com_prior(motor.lexico, prior), matriz=motor.matriz)
    candidatas = motor.todas_candidatas()
    entropias = motor.entropias(candidatas)
    melhor = int(motor.ordenar_por_entropia(entropias, candidatas)[0])
    return melhor, float(entropias[melhor])


# ------------------------------------------------------------- teste de colapso


def teste_de_colapso(
    lexico: Lexico, matriz: np.ndarray, alvo: float = M_COLAPSO,
    cortes: list[int] | None = None,
) -> dict:
    """§3: mesmo M por caminhos diferentes dá a mesma abertura?

    Se der, "concentração" é tudo que importa e a grade pode ter um eixo só (M).
    Se não der, a forma da distribuição importa por si, e reduzir (N,T) a M
    perderia informação — o que precisa ser dito em voz alta, não escondido.

    É barato (uma varredura de entropia por par) e decide o desenho do resto,
    então roda antes de tudo.
    """
    cortes = CORTES_COLAPSO if cortes is None else cortes
    motor = Motor(lexico, matriz=matriz)
    pontos, priores = [], []
    for n in cortes:
        temperatura = temperatura_para(lexico.icf, alvo, n)
        prior = prior_familia(lexico.icf, n, temperatura)
        indice, entropia = abertura_nivel2(motor, prior)
        priores.append(prior)
        pontos.append({
            "n": n,
            "temperatura": temperatura,
            "m": perplexidade(prior),
            "abertura": lexico.mostrar(indice),
            "entropia": entropia,
        })
    aberturas = sorted({p["abertura"] for p in pontos})

    # "Deu palavra diferente" e "deu palavra MUITO diferente" não são a mesma
    # notícia. Se a vencedora de um ponto perde por 0,001 bit no ponto vizinho, o
    # que houve foi empate técnico decidido pelo desempate, e reduzir tudo a M
    # continua defensável. O gap abaixo é o que separa os dois casos.
    gap = 0.0
    for prior, ponto in zip(priores, pontos):
        submotor = Motor(com_prior(lexico, prior), matriz=matriz)
        entropias = submotor.entropias(submotor.todas_candidatas())
        valores = [float(entropias[lexico.indice_de(p)]) for p in aberturas]
        gap = max(gap, max(valores) - min(valores))

    return {
        "alvo": alvo,
        "pontos": pontos,
        "colapsa": len(aberturas) == 1,
        "aberturas": aberturas,
        "gap_bits": gap,
    }


# ---------------------------------------------------------- matriz de regret


def matriz_de_regret(
    lexico: Lexico, matriz: np.ndarray, cenarios: list[Cenario],
    extras: tuple[str, ...] = ("tarso", "tosar"),
) -> dict:
    """Arrependimento de cada abertura candidata em cada mundo da grade.

        regret(g, M) = E_M[tentativas | g] - E_M[tentativas | g*_M]

    `g*_M` é a abertura que o NÍVEL 2 escolheria naquele mundo, e E é exato:
    `valor_com_abertura` soma sobre o prior, palavra a palavra, em vez de
    simular partidas. Duas colunas quaisquer são portanto comparáveis sem
    depender de semente nem de bateria.

    DUAS referências por coluna, porque uma só mente:

      g*_bits        a que o nível 2 escolheria — é o "benchmark da coluna" no
                     sentido de "o que o solver de hoje faria ali"
      g*_tentativas  a de menor E entre o beam de entropia mais as candidatas —
                     é a régua correta para uma métrica medida em tentativas

    Contra `g*_bits` o arrependimento sai NEGATIVO com frequência, e isso não é
    bug: a referência é ótima em BITS e a métrica é TENTATIVAS, que é a correção
    do 3B1B e a razão de o nível 3 existir. Só que um regret negativo estraga o
    minimax — quem bate a referência em toda coluna termina com pior caso zero
    por construção, e "zero" viraria uma propriedade da régua, não da abertura.
    Por isso o minimax roda contra `g*_tentativas`, que devolve uma matriz
    não-negativa e um vencedor que significa alguma coisa.

    `g*_tentativas` é o melhor do beam, não o melhor do léxico: varrer as 6.046
    aberturas por coluna custaria ~8 h. É o mesmo beam do nível 3, com a mesma
    justificativa — a jogada ótima quase sempre está no topo da entropia — e uma
    consequência que vale dizer na direção certa: uma referência mais fraca que a
    verdadeira encolhe a diferença, então os arrependimentos reportados são
    limites INFERIORES dos verdadeiros, nunca superiores.
    """
    base_candidatas = list(extras)
    esperados: dict[str, dict[str, float]] = {}
    colunas = []

    for cenario in cenarios:
        # O memo da busca é indexado por (rodadas, restante, candidatas) e NÃO
        # pelo prior: reaproveitá-lo entre cenários devolveria o valor do mundo
        # anterior. Uma busca nova por coluna é o que mantém isso correto.
        submotor = Motor(com_prior(lexico, cenario.prior), matriz=matriz)
        busca = MotorNivel3(submotor, BEAM, PROFUNDIDADE)
        todas = busca.todas_candidatas()
        entropias = submotor.entropias(todas)
        ordem = submotor.ordenar_por_entropia(entropias, todas)

        beam = [int(j) for j in ordem[:BEAM]]
        avaliar = list(dict.fromkeys(beam + [lexico.indice_de(p)
                                             for p in base_candidatas]))
        valores = {
            lexico.mostrar(i): busca.valor_com_abertura(i, todas) for i in avaliar
        }
        for palavra, valor in valores.items():
            esperados.setdefault(palavra, {})[cenario.descricao] = valor

        ref_bits = lexico.mostrar(int(ordem[0]))
        ref_tentativas = min(valores, key=lambda p: valores[p])
        colunas.append({
            "m": cenario.m, "n": cenario.n, "temperatura": cenario.temperatura,
            "rotulo": cenario.rotulo, "descricao": cenario.descricao,
            "referencia_bits": ref_bits,
            "esperado_bits": valores[ref_bits],
            "entropia_bits": float(entropias[int(ordem[0])]),
            "referencia_tentativas": ref_tentativas,
            "esperado_tentativas": valores[ref_tentativas],
        })
        print(f"  {cenario.descricao:32s} bits={ref_bits:8s} E={valores[ref_bits]:.4f}"
              f"   tentativas={ref_tentativas:8s} E={valores[ref_tentativas]:.4f}",
              flush=True)

    # As duas referências de cada coluna entram como candidatas — inclusive nas
    # colunas onde não foram eleitas, senão a linha delas teria buraco.
    candidatas = list(dict.fromkeys(
        base_candidatas
        + [c["referencia_bits"] for c in colunas]
        + [c["referencia_tentativas"] for c in colunas]
    ))
    for cenario, coluna in zip(cenarios, colunas):
        faltando = [p for p in candidatas
                    if coluna["descricao"] not in esperados.get(p, {})]
        if not faltando:
            continue
        busca = MotorNivel3(
            Motor(com_prior(lexico, cenario.prior), matriz=matriz),
            BEAM, PROFUNDIDADE,
        )
        todas = busca.todas_candidatas()
        for palavra in faltando:
            esperados.setdefault(palavra, {})[coluna["descricao"]] = (
                busca.valor_com_abertura(lexico.indice_de(palavra), todas)
            )
        print(f"  {coluna['descricao']:32s} +{len(faltando)} célula(s)", flush=True)

    # A referência em tentativas só fica definitiva agora. Uma palavra eleita por
    # OUTRA coluna entra aqui no passe de preenchimento, e pode bater a referência
    # provisória — foi o que aconteceu com `corta` e `carto` em M=6046, e o
    # resultado eram regrets negativos. Tomar o mínimo sobre tudo que foi avaliado
    # não é "usar o mínimo das candidatas" no sentido proibido: o conjunto é o
    # beam de entropia MAIS as candidatas, então ampliá-lo só melhora a
    # referência, e referência melhor significa arrependimento maior, nunca menor.
    for coluna in colunas:
        avaliadas = {
            palavra: linha[coluna["descricao"]]
            for palavra, linha in esperados.items()
            if coluna["descricao"] in linha
        }
        melhor = min(avaliadas, key=lambda p: avaliadas[p])
        if melhor != coluna["referencia_tentativas"]:
            print(f"  {coluna['descricao']:32s} referência revista: "
                  f"{coluna['referencia_tentativas']} -> {melhor}")
        coluna["referencia_tentativas"] = melhor
        coluna["esperado_tentativas"] = avaliadas[melhor]
        coluna["n_avaliadas"] = len(avaliadas)

    regret = {
        p: {c["descricao"]: esperados[p][c["descricao"]] - c["esperado_tentativas"]
            for c in colunas}
        for p in candidatas
    }
    regret_vs_bits = {
        p: {c["descricao"]: esperados[p][c["descricao"]] - c["esperado_bits"]
            for c in colunas}
        for p in candidatas
    }
    pior_caso = {p: max(regret[p].values()) for p in candidatas}
    robusta = min(pior_caso, key=lambda p: pior_caso[p])
    return {
        "cenarios": colunas,
        "candidatas": candidatas,
        "esperados": {p: esperados[p] for p in candidatas},
        "regret": regret,
        "regret_vs_bits": regret_vs_bits,
        "pior_caso": pior_caso,
        "pior_caso_vs_bits": {p: max(regret_vs_bits[p].values()) for p in candidatas},
        "robusta": robusta,
    }


# ------------------------------------------------------------------ assinatura


@functools.cache
def assinatura_robustez() -> str:
    """Hash do código que decide os números — mesmo motivo do `assinatura_busca`.

    A grade custa minutos e vai para o disco. Se a definição de M, da família de
    priors ou do arrependimento mudar, o cache tem que morrer junto; reescrever
    um comentário ou reformatar, não. Fica de fora o que tem oráculo em teste
    (`perplexidade` é comparada com o uniforme) e o que já entra pelo
    `assinatura_busca` (a busca do nível 3, importada aqui).
    """
    alvos = (
        prior_familia,
        temperatura_para,
        abertura_nivel2,
        matriz_de_regret,
        MotorNivel3.valor_com_abertura,
    )
    # `_sem_docstring` é do nível 3 e vale a importação atravessada: a regra de
    # "texto não muda o que a conta decide" tem que ser a MESMA nos dois caches,
    # senão um deles cobra recálculo por reescrita de comentário. (Este módulo já
    # tinha esse bug: editar um docstring invalidava os 8 min.)
    partes = [
        ast.unparse(_sem_docstring(ast.parse(textwrap.dedent(inspect.getsource(f)))))
        for f in alvos
    ]
    partes.append(repr((PONTOS_M, M_COLAPSO, CORTES_COLAPSO, BEAM, PROFUNDIDADE)))
    return hashlib.sha256("\n".join(partes).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------- fachada


def calcular(
    lexico: Lexico | None = None, usar_cache: bool = True, verboso: bool = True
) -> dict:
    """A análise inteira: perplexidades, teste de colapso e matriz de regret."""
    lexico = Lexico.carregar(1.0) if lexico is None else lexico
    chave = f"{assinatura_robustez()}:{len(lexico)}"

    if usar_cache and ARQ_CACHE.exists():
        guardado = json.loads(ARQ_CACHE.read_text(encoding="utf-8"))
        if guardado.get("chave") == chave:
            if verboso:
                print(f"  (em cache: {ARQ_CACHE})")
            return guardado

    from .matriz import carregar_matriz

    matriz = carregar_matriz(lexico.palavras)

    escala = [
        {"t": t, "m": perplexidade(prior_familia(lexico.icf, len(lexico), t))}
        for t in (0.25, 0.5, 1.0, 2.0, 4.0, math.inf)
    ]
    if verboso:
        print("\n[1/3] número efetivo de palavras por temperatura")
        for linha in escala:
            t = "∞" if math.isinf(linha["t"]) else f"{linha['t']:g}"
            print(f"  T={t:<5} M={linha['m']:8.1f}")

    if verboso:
        print(f"\n[2/3] teste de colapso: M≈{M_COLAPSO} por três formas diferentes")
    colapso = teste_de_colapso(lexico, matriz)
    if verboso:
        for ponto in colapso["pontos"]:
            t = "∞" if math.isinf(ponto["temperatura"]) else f"{ponto['temperatura']:.3g}"
            print(f"  N={ponto['n']:<5} T={t:<7} M={ponto['m']:7.1f}  "
                  f"-> {ponto['abertura']} ({ponto['entropia']:.3f} bits)")
        print(f"  colapsa: {colapso['colapsa']}  aberturas: {colapso['aberturas']}  "
              f"gap: {colapso['gap_bits']:.4f} bits")

    cenarios = montar_grade(lexico)
    if verboso:
        print(f"\n[3/3] matriz de regret ({len(cenarios)} mundos)", flush=True)
    regret = matriz_de_regret(lexico, matriz, cenarios)

    resultado = {
        "chave": chave,
        "n_lexico": len(lexico),
        "escala_t_m": escala,
        "colapso": colapso,
        "piso_empirico": piso_empirico(lexico),
        **regret,
    }
    ARQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ARQ_CACHE.write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return resultado


def conferir(resultado: dict, lexico: Lexico | None = None) -> list[str]:
    """Os dois sanity checks da tarefa. Devolve a lista de violações.

    1. regret_vs_bits(`tarso`, mundo do T=1) = 0. `tarso` É a abertura do nível 2
       sob T=1, então ela é a referência em bits daquela coluna e o
       arrependimento dela contra ela mesma é zero. Qualquer outro valor é bug de
       pipeline, não achado — pare e investigue.
    2. regret >= 0 em toda célula, contra a referência em TENTATIVAS. Aqui vale
       de verdade, e uma violação seria bug: a referência é o mínimo do conjunto
       avaliado, então nada dentro dele pode ficar abaixo.

    O que NÃO é violação: `regret_vs_bits` negativo. Isso é a distância entre os
    dois objetivos, e vem reportado à parte porque é achado, não erro.
    """
    violacoes = []
    coluna = next(
        (c["descricao"] for c in resultado["cenarios"] if c["rotulo"] == ROTULO_PADRAO),
        None,
    )
    if coluna is None:
        violacoes.append("a coluna do prior padrão sumiu da grade")
    else:
        valor = resultado["regret_vs_bits"].get("tarso", {}).get(coluna)
        if valor is None:
            violacoes.append("`tarso` não está entre as candidatas")
        elif abs(valor) > 1e-9:
            violacoes.append(
                f"regret_vs_bits(tarso, {coluna}) = {valor:.6f}, deveria ser 0 — "
                "`tarso` é a abertura de nível 2 sob T=1 e portanto a referência"
            )
    for palavra, linha in resultado["regret"].items():
        for mundo, valor in linha.items():
            if valor < -1e-9:
                violacoes.append(
                    f"regret({palavra}, {mundo}) = {valor:.6f} < 0 — a referência "
                    "da coluna é o mínimo do conjunto avaliado; nada pode ficar abaixo"
                )
    return violacoes


def diagnostico_bits_vs_tentativas(resultado: dict) -> dict:
    """Quantas vezes a escolha do nível 2 não é a melhor em tentativas.

    Não é sanity check: é o resultado. Cada coluna em que as duas referências
    divergem é um mundo onde maximizar bits deixa tentativas na mesa, e a
    diferença entre elas é quanto.
    """
    divergem = [
        {
            "mundo": c["descricao"],
            "bits": c["referencia_bits"],
            "tentativas": c["referencia_tentativas"],
            "custo": c["esperado_bits"] - c["esperado_tentativas"],
        }
        for c in resultado["cenarios"]
        if c["referencia_bits"] != c["referencia_tentativas"]
    ]
    return {
        "n_mundos": len(resultado["cenarios"]),
        "n_divergem": len(divergem),
        "divergencias": divergem,
        "custo_maximo": max((d["custo"] for d in divergem), default=0.0),
    }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    dados = calcular(usar_cache="--sem-cache" not in sys.argv)
    print(f"\nabertura robusta (minimax de regret): {dados['robusta']}")
    for palavra, valor in sorted(dados["pior_caso"].items(), key=lambda p: p[1]):
        print(f"  {palavra:8s} pior caso {valor:+.4f}")

    diagnostico = diagnostico_bits_vs_tentativas(dados)
    print(f"\nbits x tentativas: as duas referências divergem em "
          f"{diagnostico['n_divergem']}/{diagnostico['n_mundos']} mundos "
          f"(até {diagnostico['custo_maximo']:+.4f} tentativa)")
    for linha in diagnostico["divergencias"]:
        print(f"  {linha['mundo']:32s} bits={linha['bits']:8s} "
              f"tentativas={linha['tentativas']:8s} custo={linha['custo']:+.4f}")

    problemas = conferir(dados)
    print("\nsanity checks:", "todos passam" if not problemas else "")
    for problema in problemas:
        print(f"  ! {problema}")
