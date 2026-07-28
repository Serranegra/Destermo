"""Cálculo do feedback colorido do Termo (seção 3 da especificação v1.1).

Regra de duas passadas com consumo de estoque de letras. Verdes têm prioridade
absoluta sobre amarelos: se uma letra aparece 2x na tentativa e 1x na secreta,
apenas uma ocorrência fica colorida.

Acentos NÃO são considerados nas cores — o Termo os preenche automaticamente e o
jogador nunca os digita. Tudo aqui opera sobre a forma normalizada: "saúde" e
"saude" são a mesma palavra para efeito de feedback.
"""

import unicodedata
from collections import Counter
from itertools import permutations

TAMANHO = 5

# §3.5 é um ponto em aberto: a ajuda do term.ooo fala em "acentos" genericamente,
# e NFD decompõe "ç" em "c" + cedilha combinante, então a normalização padrão faz
# ç -> c. São 172 palavras do léxico afetadas. Para testar empiricamente: quando a
# resposta do dia tiver "ç", digitar a variante com "c" — se vier verde, está
# correto; se vier cinza, basta pôr False aqui (a matriz em cache se invalida
# sozinha, porque a assinatura é sobre a lista de palavras).
NORMALIZAR_CEDILHA = True

_CEDILHA = "̧"  # combining cedilla

PRETO = "B"
AMARELO = "Y"
VERDE = "G"

_VALOR = {PRETO: 0, AMARELO: 1, VERDE: 2}
_CHAR = (PRETO, AMARELO, VERDE)

N_PADROES = 3**TAMANHO  # 243
PADRAO_VITORIA = 3**TAMANHO - 1  # "GGGGG" == 242


def normalizar(palavra: str) -> str:
    """Remove diacríticos: "saúde" -> "saude", "terço" -> "terco".

    NFD decompõe "á" em "a" + acento combinante; a categoria Unicode 'Mn'
    (Mark, nonspacing) são justamente os diacríticos.
    """
    decomposta = unicodedata.normalize("NFD", palavra.lower())
    manter = () if NORMALIZAR_CEDILHA else (_CEDILHA,)
    return unicodedata.normalize(
        "NFC",
        "".join(
            c for c in decomposta
            if unicodedata.category(c) != "Mn" or c in manter
        ),
    )


def calcular_feedback(tentativa: str, secreta: str) -> str:
    """Devolve o padrão de 5 chars em G/Y/B para `tentativa` contra `secreta`.

    Normaliza defensivamente as duas palavras. O léxico já é guardado normalizado
    (§2.4), então no caminho quente isto é um no-op; o que importa é a fronteira
    de entrada do usuário, que pode vir acentuada.
    """
    tentativa = normalizar(tentativa)
    secreta = normalizar(secreta)

    if len(tentativa) != TAMANHO or len(secreta) != TAMANHO:
        raise ValueError(
            f"palavras devem ter {TAMANHO} caracteres: "
            f"{tentativa!r} ({len(tentativa)}), {secreta!r} ({len(secreta)})"
        )

    resultado = [PRETO] * TAMANHO
    estoque = Counter(secreta)

    # PASSADA 1 — verdes consomem estoque primeiro.
    for i in range(TAMANHO):
        if tentativa[i] == secreta[i]:
            resultado[i] = VERDE
            estoque[tentativa[i]] -= 1

    # PASSADA 2 — amarelos só com o que sobrou.
    for i in range(TAMANHO):
        if resultado[i] == VERDE:
            continue
        letra = tentativa[i]
        if estoque[letra] > 0:
            resultado[i] = AMARELO
            estoque[letra] -= 1

    return "".join(resultado)


def padrao_para_codigo(padrao: str) -> int:
    """"YGBBB" -> 1*3^4 + 2*3^3 = 135. Faixa 0..242."""
    if len(padrao) != TAMANHO:
        raise ValueError(f"padrão deve ter {TAMANHO} chars: {padrao!r}")
    codigo = 0
    for char in padrao:
        try:
            codigo = codigo * 3 + _VALOR[char]
        except KeyError:
            raise ValueError(f"char inválido em padrão: {char!r}") from None
    return codigo


def codigo_para_padrao(codigo: int) -> str:
    """Inverso de `padrao_para_codigo`."""
    if not 0 <= codigo < N_PADROES:
        raise ValueError(f"código fora da faixa 0..{N_PADROES - 1}: {codigo}")
    chars = []
    for _ in range(TAMANHO):
        codigo, resto = divmod(codigo, 3)
        chars.append(_CHAR[resto])
    return "".join(reversed(chars))


def calcular_codigo(tentativa: str, secreta: str) -> int:
    """Feedback já codificado como inteiro."""
    return padrao_para_codigo(calcular_feedback(tentativa, secreta))


def padrao_possivel(tentativa: str, padrao: str) -> bool:
    """Existe alguma palavra secreta que produziria `padrao` para `tentativa`?

    Independe do léxico — é uma checagem puramente lógica, usada para distinguir
    "você digitou o feedback errado" de "a palavra do dia não está na nossa lista"
    (seção 7.3). Dois motivos de impossibilidade:

    1. Ordem dentro de uma letra repetida. A passada 2 pinta de amarelo as
       ocorrências mais à esquerda, então um B nunca pode vir antes de um Y da
       mesma letra. Ex.: "BY" para as duas ocorrências de 'o' em "oovxx".
    2. Falta de casa para os amarelos. Cada amarelo exige uma cópia da letra numa
       posição não-verde, e essa posição não pode ser a da própria letra na
       tentativa. Ex.: "GGGGY" — sobra uma única posição livre, e ela é justamente
       a da letra que precisaria aparecer.
    """
    tentativa = normalizar(tentativa)
    if len(tentativa) != TAMANHO or len(padrao) != TAMANHO:
        return False

    # (1) amarelos antes de pretos, dentro de cada letra
    por_letra: dict[str, list[str]] = {}
    for letra, marca in zip(tentativa, padrao):
        if marca != VERDE:
            por_letra.setdefault(letra, []).append(marca)
    for marcas in por_letra.values():
        if PRETO in marcas and AMARELO in marcas[marcas.index(PRETO) + 1 :]:
            return False

    # (2) cada amarelo precisa de uma posição livre que não seja a sua própria
    livres = [i for i in range(TAMANHO) if padrao[i] != VERDE]
    amarelas = [tentativa[i] for i in range(TAMANHO) if padrao[i] == AMARELO]
    if len(amarelas) > len(livres):
        return False
    return any(
        all(letra != tentativa[pos] for letra, pos in zip(amarelas, alocacao))
        for alocacao in permutations(livres, len(amarelas))
    )


def normalizar_padrao(texto: str) -> str:
    """Aceita entrada do usuário em G/Y/B (case-insensitive) ou emoji/dígitos.

    Levanta ValueError se não for um padrão válido de 5 posições.
    """
    equivalencias = {
        "g": VERDE, "v": VERDE, "2": VERDE, "🟩": VERDE, "🟢": VERDE,
        "y": AMARELO, "a": AMARELO, "1": AMARELO, "🟨": AMARELO, "🟡": AMARELO,
        "b": PRETO, "p": PRETO, "c": PRETO, "0": PRETO, "⬛": PRETO, "⬜": PRETO,
        "-": PRETO, ".": PRETO,
    }
    bruto = "".join(texto.split())
    chars = []
    for char in bruto:
        normalizado = equivalencias.get(char.lower())
        if normalizado is None:
            raise ValueError(f"char inválido no feedback: {char!r}")
        chars.append(normalizado)
    if len(chars) != TAMANHO:
        raise ValueError(f"feedback deve ter {TAMANHO} posições, veio {len(chars)}")
    return "".join(chars)
