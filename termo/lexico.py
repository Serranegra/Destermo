"""Léxico e prior de frequência (seções 2 e 4.2 da especificação v1.1).

Fonte: https://github.com/fserb/pt-br (licença MIT), do criador do Termo.

  lexico  145.744 entradas -> 6.301 palavras de 5 caracteres
  agrupar por forma normalizada -> 6.046 grupos (242 com colisão, 497 palavras)
  manter de cada grupo a variante de menor ICF -> LÉXICO FINAL de 6.046 palavras
  icf     419.486 palavras com Inverse Corpus Frequency (prior e desempate)

O arquivo `conjugações` não entra no léxico: a curadoria oficial do Termo removeu
as formas verbais, então nenhuma delas pode ser a secreta. Mas o Termo as ACEITA
como tentativa, e uma tentativa não precisa poder ser a resposta para ser útil —
só precisa separar bem. Daí a distinção de dois espaços (§2.5):

  candidatas  6.046  o que pode ser a secreta; é sobre elas que se filtra
  sondas      8.629  o que se pode digitar; as 6.046 mais 2.582 conjugações e
                     mais as de `SONDAS_MANUAIS` (hoje só `areio`)

As sondas são um SUPERCONJUNTO com as candidatas no prefixo, na mesma ordem, de
modo que todo índice de candidata continua valendo no espaço ampliado. Sem
`ampliado=True` os dois espaços são o mesmo objeto e nada muda.

Cada palavra tem duas formas (§2.4):

  normalizada ("terco")  todo cálculo — feedback, matriz, entropia, filtragem
  exibição    ("terço")  apresentação ao usuário, puramente cosmética

Contagem é de CARACTERES, não de bytes — acentos em UTF-8 ocupam mais de 1 byte.
"""

from __future__ import annotations

import math
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .feedback import normalizar

RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "data"

ARQ_LEXICO_FINAL = RAIZ / "termo_lexico_5letras.txt"
ARQ_SONDAS_EXTRA = RAIZ / "termo_sondas_extra.txt"
URL_BASE = "https://raw.githubusercontent.com/fserb/pt-br/master/"
FONTES = {
    "lexico.txt": "lexico",
    "icf.txt": "icf",
}
# Baixado sob demanda, só por quem pede o espaço ampliado: são 2,2 MB que não
# servem para nada no modo padrão.
FONTES_SONDAS = {"conjugacoes.txt": "conjugações"}

# Sondas que o term.ooo aceita e as nossas fontes não têm. O critério para entrar
# aqui é estreito: a palavra tem de ser digitável no jogo real E ter uso medido
# entre quem joga — não basta ser uma forma válida do português. Cada entrada vem
# com a justificativa, porque a lista é curadoria manual e sem isso apodrece.
#
#   areio  1ª pessoa de `areiar`. `arear` e `areia` estão no léxico, mas nenhuma
#          conjugação de nenhum dos dois sobreviveu à fonte. É a abertura mais
#          digitada do Termo, e sem ela aqui não dava para medi-la
#          (`benchmark.py --areio`).
SONDAS_MANUAIS = ("areio",)

TAMANHO = 5
# a-z mais o bloco latin-1 acentuado, excluindo o '÷' (U+00F7) que cai no meio da faixa.
_PALAVRA_VALIDA = re.compile(r"[a-zà-öø-ÿ]{%d}" % TAMANHO)
_NORMALIZADA_VALIDA = re.compile(r"[a-z]{%d}" % TAMANHO)

T_PADRAO = 1.0


# ---------------------------------------------------------------- download


def garantir_fontes(
    dir_dados: Path = DIR_DADOS, fontes: dict[str, str] = FONTES
) -> None:
    """Baixa os arquivos do repositório fserb/pt-br que ainda não existem."""
    dir_dados.mkdir(parents=True, exist_ok=True)
    for local, remoto in fontes.items():
        destino = dir_dados / local
        if destino.exists() and destino.stat().st_size > 0:
            continue
        url = URL_BASE + urllib.parse.quote(remoto)
        print(f"baixando {remoto} -> {destino.name} ...")
        with urllib.request.urlopen(url, timeout=120) as resposta:
            destino.write_bytes(resposta.read())


def _ler_linhas(caminho: Path) -> list[str]:
    with caminho.open(encoding="utf-8") as arquivo:
        return [linha.strip() for linha in arquivo]


def _tabela_icf(dir_dados: Path = DIR_DADOS) -> dict[str, float]:
    """Mapa palavra (como aparece na fonte, acentuada) -> score ICF."""
    garantir_fontes(dir_dados)
    tabela: dict[str, float] = {}
    with (dir_dados / "icf.txt").open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            palavra, _, score = linha.strip().partition(",")
            if palavra and palavra not in tabela:
                tabela[palavra] = float(score)
    return tabela


# ---------------------------------------------------------------- construção


def _cinco_caracteres(caminho: Path) -> set[str]:
    """Palavras de exatamente 5 caracteres de um arquivo da fonte."""
    return {
        palavra
        for bruta in _ler_linhas(caminho)
        for palavra in (bruta.lower(),)
        if len(palavra) == TAMANHO and _PALAVRA_VALIDA.fullmatch(palavra)
    }


def construir_lexico(dir_dados: Path = DIR_DADOS, salvar: bool = True) -> list[str]:
    """Reconstrói a lista final a partir da fonte. Devolve as formas de exibição.

    Passo 1 — palavras de exatamente 5 caracteres em `lexico`.
    Passo 2 — agrupar por forma normalizada.
    Passo 3 — de cada grupo, manter a variante de menor ICF (a mais comum).
    """
    garantir_fontes(dir_dados)

    cinco = _cinco_caracteres(dir_dados / "lexico.txt")

    icf = _tabela_icf(dir_dados)
    ausente = max(icf.values()) + 1.0

    grupos: dict[str, list[str]] = defaultdict(list)
    for palavra in cinco:
        grupos[normalizar(palavra)].append(palavra)

    # Sem a dedupe, as variantes de um mesmo grupo seriam indistinguíveis pelo
    # feedback e o solver travaria ao convergir para o grupo (§2.3).
    final = [
        min(sorted(variantes), key=lambda p: icf.get(p, ausente))
        for _, variantes in sorted(grupos.items())
    ]

    colididos = sum(len(v) for v in grupos.values() if len(v) > 1)
    print(
        f"5 caracteres={len(cinco)}  grupos={len(grupos)}  "
        f"colisões={sum(1 for v in grupos.values() if len(v) > 1)} "
        f"({colididos} palavras, {colididos / len(cinco):.1%})  final={len(final)}"
    )
    if salvar:
        ARQ_LEXICO_FINAL.write_text("\n".join(final) + "\n", encoding="utf-8")
    return final


def carregar_exibicao(caminho: Path = ARQ_LEXICO_FINAL) -> list[str]:
    """Lê a lista final (formas acentuadas); reconstrói se ela não existir."""
    if not caminho.exists():
        return construir_lexico()
    palavras = [linha for linha in _ler_linhas(caminho) if linha]
    invalidas = [p for p in palavras if not _PALAVRA_VALIDA.fullmatch(p)]
    if invalidas:
        raise ValueError(f"palavras inválidas no léxico: {invalidas[:5]}")
    normalizadas = [normalizar(p) for p in palavras]
    if len(set(normalizadas)) != len(normalizadas):
        raise ValueError(
            "o léxico tem formas normalizadas repetidas — provavelmente é um "
            "arquivo da v1.0. Apague-o e deixe o construtor gerar de novo."
        )
    return palavras


# ---------------------------------------------------------------- sondas


def construir_sondas(dir_dados: Path = DIR_DADOS, salvar: bool = True) -> list[str]:
    """Palavras de 5 letras que servem de sonda mas nunca de resposta (§2.5).

    Mesma fonte e mesmos filtros do léxico, aplicados ao arquivo `conjugações`,
    mais as de `SONDAS_MANUAIS`. Duas diferenças de propósito em relação a
    `construir_lexico`:

      - o que já existe no léxico é descartado; o resultado é só o ACRÉSCIMO
      - a dedupe por forma normalizada continua (duas variantes acentuadas são a
        mesma sonda), mas o critério de ICF aqui só escolhe qual acentuação
        mostrar — nenhuma delas jamais entra numa conta.
    """
    garantir_fontes(dir_dados, FONTES_SONDAS)
    conhecidas = {normalizar(palavra) for palavra in carregar_exibicao()}

    grupos: dict[str, list[str]] = defaultdict(list)
    fonte = _cinco_caracteres(dir_dados / "conjugacoes.txt") | set(SONDAS_MANUAIS)
    for palavra in fonte:
        chave = normalizar(palavra)
        if chave not in conhecidas:
            grupos[chave].append(palavra)

    icf = _tabela_icf(dir_dados)
    ausente = max(icf.values()) + 1.0
    extras = [
        min(sorted(variantes), key=lambda p: icf.get(p, ausente))
        for _, variantes in sorted(grupos.items())
    ]

    print(
        f"conjugações: {len(conhecidas)} já no léxico  "
        f"+{len(extras)} sondas novas ({len(SONDAS_MANUAIS)} manuais)  "
        f"-> espaço de tentativa {len(conhecidas) + len(extras)}"
    )
    if salvar:
        ARQ_SONDAS_EXTRA.write_text("\n".join(extras) + "\n", encoding="utf-8")
    return extras


def carregar_sondas_extra(caminho: Path = ARQ_SONDAS_EXTRA) -> list[str]:
    """Lê o acréscimo de sondas (formas acentuadas); reconstrói se ausente."""
    if not caminho.exists():
        return construir_sondas()
    extras = [linha for linha in _ler_linhas(caminho) if linha]
    invalidas = [p for p in extras if not _PALAVRA_VALIDA.fullmatch(p)]
    if invalidas:
        raise ValueError(f"sondas inválidas: {invalidas[:5]}")
    return extras


# ---------------------------------------------------------------- ICF / prior


def carregar_icf(
    exibicao: list[str], dir_dados: Path = DIR_DADOS
) -> tuple[np.ndarray, list[str]]:
    """Devolve (scores ICF alinhados a `exibicao`, lista das ausentes).

    ICF = Inverse Corpus Frequency: score BAIXO significa palavra COMUM.
    Palavras ausentes recebem o maior score observado + 1 (tratadas como raras).
    """
    tabela = _tabela_icf(dir_dados)
    ausentes = [p for p in exibicao if p not in tabela]
    padrao = max(tabela.values()) + 1.0 if tabela else 1.0
    icf = np.array([tabela.get(p, padrao) for p in exibicao], dtype=np.float64)
    return icf, ausentes


def calcular_prior(icf: np.ndarray, temperatura: float = T_PADRAO) -> np.ndarray:
    """prior(w) = softmax(-ICF(w) / T), com log-sum-exp para estabilidade.

    T -> 0    corte quase binário nas palavras mais comuns
    T = 1     valor padrão da v1
    T -> inf  distribuição uniforme = entropia pura (Nível 1)
    """
    if not math.isfinite(temperatura):
        return np.full(icf.shape, 1.0 / icf.size, dtype=np.float64)
    if temperatura <= 0:
        raise ValueError("temperatura deve ser > 0 (use math.inf para uniforme)")
    expoentes = -icf / temperatura
    expoentes -= expoentes.max()  # log-sum-exp: evita overflow/underflow
    pesos = np.exp(expoentes)
    return pesos / pesos.sum()


# ---------------------------------------------------------------- fachada


@dataclass
class Lexico:
    """Léxico carregado: formas normalizada e de exibição, ICF e prior.

    Dois espaços de índice convivem aqui (§2.5), e a relação entre eles é o que
    faz o resto do motor não precisar saber da diferença:

      0 .. len(palavras)-1   CANDIDATAS: podem ser a secreta; têm ICF e prior
      0 .. len(sondas)-1     SONDAS: podem ser digitadas; começam pelas candidatas

    Como as candidatas são o PREFIXO das sondas, todo índice de candidata é um
    índice de sonda válido e com o mesmo significado. Os campos de sonda ficam
    com default: quem constrói um `Lexico` sem eles — o benchmark monta mundos
    restritos assim — cai no caso em que os dois espaços coincidem.
    """

    palavras: list[str]       # formas normalizadas — o motor só usa estas
    exibicao: list[str]       # formas acentuadas — só para mostrar ao usuário
    icf: np.ndarray
    ausentes_no_icf: list[str]
    temperatura: float
    prior: np.ndarray
    indice: dict[str, int]    # forma normalizada -> posição (só candidatas)

    sondas: list[str] = field(default_factory=list)           # normalizadas
    sondas_exibicao: list[str] = field(default_factory=list)  # acentuadas
    indice_sonda: dict[str, int] = field(default_factory=dict)
    prior_sondas: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.sondas:
            # Sem espaço ampliado os dois são o mesmo objeto — não uma cópia, para
            # que `carregar_matriz` reconheça a matriz quadrada de sempre.
            self.sondas = self.palavras
            self.sondas_exibicao = self.exibicao
            self.indice_sonda = self.indice
        if self.prior_sondas is None:
            self.prior_sondas = self._espalhar_prior(self.prior)

    def _espalhar_prior(self, prior: np.ndarray) -> np.ndarray:
        """Prior no espaço de sondas: 0 para o que não pode ser a secreta.

        Não é um chute de frequência — é a verdade. Uma conjugação tem
        probabilidade zero de ser a palavra do dia. O valor só entra como
        desempate de última instância em `ordenar_por_entropia`, onde zerar
        significa "entre duas sondas que separam igual, prefira a que ainda pode
        encerrar o jogo".
        """
        extras = len(self.sondas) - len(prior)
        if extras <= 0:
            return prior
        return np.concatenate([prior, np.zeros(extras, dtype=prior.dtype)])

    @classmethod
    def carregar(cls, temperatura: float = T_PADRAO, ampliado: bool = False) -> "Lexico":
        exibicao = carregar_exibicao()
        icf, ausentes = carregar_icf(exibicao)
        palavras = [normalizar(p) for p in exibicao]
        indice = {p: i for i, p in enumerate(palavras)}

        sondas_exibicao: list[str] = []
        sondas: list[str] = []
        indice_sonda: dict[str, int] = {}
        if ampliado:
            sondas_exibicao = exibicao + carregar_sondas_extra()
            sondas = [normalizar(p) for p in sondas_exibicao]
            indice_sonda = {p: i for i, p in enumerate(sondas)}
            if len(indice_sonda) != len(sondas):
                raise ValueError("o acréscimo de sondas colide com o léxico")

        return cls(
            palavras=palavras,
            exibicao=exibicao,
            icf=icf,
            ausentes_no_icf=ausentes,
            temperatura=temperatura,
            prior=calcular_prior(icf, temperatura),
            indice=indice,
            sondas=sondas,
            sondas_exibicao=sondas_exibicao,
            indice_sonda=indice_sonda,
        )

    def com_temperatura(self, temperatura: float) -> "Lexico":
        """Cópia com outro T (recalcula só o prior — o resto é compartilhado)."""
        ampliado = self.sondas is not self.palavras
        return Lexico(
            palavras=self.palavras,
            exibicao=self.exibicao,
            icf=self.icf,
            ausentes_no_icf=self.ausentes_no_icf,
            temperatura=temperatura,
            prior=calcular_prior(self.icf, temperatura),
            indice=self.indice,
            sondas=self.sondas if ampliado else [],
            sondas_exibicao=self.sondas_exibicao if ampliado else [],
            indice_sonda=self.indice_sonda if ampliado else {},
        )

    def __len__(self) -> int:
        return len(self.palavras)

    @property
    def n_sondas(self) -> int:
        """Tamanho do espaço de tentativa — o que a matriz tem de linhas."""
        return len(self.sondas)

    @property
    def ampliado(self) -> bool:
        return self.n_sondas > len(self.palavras)

    def indice_de(self, palavra: str) -> int:
        """Índice no espaço de SONDAS. Aceita a palavra com ou sem acento.

        É o espaço certo para quem quer jogar uma palavra: o de candidatas é um
        prefixo dele, então uma resposta possível devolve o mesmo índice de antes.
        """
        chave = normalizar(palavra)
        try:
            return self.indice_sonda[chave]
        except KeyError:
            raise KeyError(f"{palavra!r} não está no léxico") from None

    def e_candidata(self, indice: int) -> bool:
        return indice < len(self.palavras)

    def mostrar(self, indice: int) -> str:
        """Forma acentuada, que é a que o jogador vê na tela do jogo (§7.2)."""
        return self.sondas_exibicao[indice]

    def mais_comuns(self, n: int) -> np.ndarray:
        """Índices das n palavras de menor ICF (as mais comuns)."""
        return np.argsort(self.icf, kind="stable")[:n]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    lexico = Lexico.carregar()
    print(f"{len(lexico)} palavras")
    print(f"ausentes no ICF: {len(lexico.ausentes_no_icf)}")
    print(f"ICF min={lexico.icf.min():.2f} mediana={np.median(lexico.icf):.2f} "
          f"max={lexico.icf.max():.2f}")
    ordem = np.argsort(lexico.icf)
    print("mais comuns:", [lexico.mostrar(i) for i in ordem[:8]])
    print("mais raras: ", [lexico.mostrar(i) for i in ordem[-4:]])
    for palavra in ("muito", "carro", "banco", "termo", "trave", "xeque", "festa"):
        i = lexico.indice_de(palavra)
        print(f"  {palavra}: exibe={lexico.mostrar(i)!r} ICF={lexico.icf[i]:.2f}  "
              f"prior={lexico.prior[i]:.3e}")
    for grupo in ("terco", "agora", "pique", "manto"):
        i = lexico.indice_de(grupo)
        print(f"  grupo {grupo!r} -> mantida {lexico.mostrar(i)!r} "
              f"(ICF {lexico.icf[i]:.2f})")
