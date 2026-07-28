"""Matriz de padrões pré-computada (seção 6 da especificação).

M[i][j] = feedback(palavra_i como tentativa, palavra_j como secreta), codificado
em base 3 (0..242) e guardado em uint8.

    8.996 x 8.996 = ~81M células = ~81 MB em disco.

Em Python puro seriam ~81 milhões de chamadas a `calcular_feedback` (minutos a
horas). Aqui a construção é vetorizada em numpy: as duas passadas do algoritmo
viram 5 operações de indexação por bloco de tentativas.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .feedback import N_PADROES, TAMANHO, calcular_codigo
from .lexico import DIR_DADOS

ARQ_MATRIZ = DIR_DADOS / "matriz_padroes.npy"
ARQ_META = DIR_DADOS / "matriz_padroes.json"

POTENCIAS = np.array([3 ** (TAMANHO - 1 - p) for p in range(TAMANHO)], dtype=np.int16)


def _assinatura(palavras: list[str]) -> str:
    digest = hashlib.sha256("\n".join(palavras).encode("utf-8")).hexdigest()
    return f"{len(palavras)}:{digest[:16]}"


def _codificar_palavras(palavras: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(códigos de letra (N,5) uint8, contagens por letra (N,A) int8)."""
    alfabeto = sorted({letra for palavra in palavras for letra in palavra})
    posicao = {letra: i for i, letra in enumerate(alfabeto)}

    codigos = np.array(
        [[posicao[letra] for letra in palavra] for palavra in palavras], dtype=np.uint8
    )
    contagens = np.zeros((len(palavras), len(alfabeto)), dtype=np.int8)
    linhas = np.arange(len(palavras))[:, None]
    np.add.at(contagens, (linhas, codigos), 1)
    return codigos, contagens


def construir_matriz(
    palavras: list[str], bloco: int = 128, verboso: bool = True
) -> np.ndarray:
    """Constrói a matriz completa de padrões, vetorizada em numpy."""
    n = len(palavras)
    codigos, contagens = _codificar_palavras(palavras)
    n_letras = contagens.shape[1]

    matriz = np.empty((n, n), dtype=np.uint8)
    # Buffers reaproveitados entre blocos (evita realocar ~50 MB por iteração).
    estoque = np.empty((bloco, n, n_letras), dtype=np.int8)
    inicio = time.perf_counter()

    for i0 in range(0, n, bloco):
        i1 = min(i0 + bloco, n)
        b = i1 - i0
        tentativas = codigos[i0:i1]                       # (b, 5)
        verdes = tentativas[:, None, :] == codigos[None, :, :]  # (b, n, 5)

        est = estoque[:b]
        est[...] = contagens                              # (b, n, A)

        bidx = np.arange(b)[:, None]
        jidx = np.arange(n)[None, :]

        # PASSADA 1 — verdes consomem estoque antes de qualquer amarelo.
        for p in range(TAMANHO):
            letra = tentativas[:, p][:, None].astype(np.intp)
            est[bidx, jidx, letra] -= verdes[:, :, p]

        # PASSADA 2 — amarelos com o que sobrou, da esquerda para a direita.
        acumulado = np.zeros((b, n), dtype=np.int16)
        for p in range(TAMANHO):
            letra = tentativas[:, p][:, None].astype(np.intp)
            disponivel = est[bidx, jidx, letra]
            verde_p = verdes[:, :, p]
            amarelo = (~verde_p) & (disponivel > 0)
            est[bidx, jidx, letra] -= amarelo
            valor = np.where(verde_p, np.int16(2), amarelo.astype(np.int16))
            acumulado += valor * POTENCIAS[p]

        matriz[i0:i1] = acumulado.astype(np.uint8)

        if verboso and (i0 // bloco) % 10 == 0:
            feito = i1 / n
            decorrido = time.perf_counter() - inicio
            restante = decorrido / feito - decorrido if feito else 0
            print(
                f"  {i1:5d}/{n} ({feito:5.1%})  "
                f"{decorrido:5.1f}s decorridos, ~{restante:4.1f}s restantes",
                flush=True,
            )

    if verboso:
        print(f"  matriz pronta em {time.perf_counter() - inicio:.1f}s")
    return matriz


def carregar_matriz(
    palavras: list[str], caminho: Path = ARQ_MATRIZ, forcar: bool = False
) -> np.ndarray:
    """Carrega a matriz do disco; reconstrói se ausente ou desatualizada."""
    assinatura = _assinatura(palavras)
    if not forcar and caminho.exists() and ARQ_META.exists():
        meta = json.loads(ARQ_META.read_text(encoding="utf-8"))
        if meta.get("assinatura") == assinatura:
            return np.load(caminho, mmap_mode=None)

    print(f"construindo matriz de padrões {len(palavras)}x{len(palavras)} ...")
    matriz = construir_matriz(palavras)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.save(caminho, matriz)
    ARQ_META.write_text(
        json.dumps({"assinatura": assinatura, "n": len(palavras)}, indent=2),
        encoding="utf-8",
    )
    print(f"  salva em {caminho} ({caminho.stat().st_size / 1e6:.0f} MB)")
    return matriz


def codigos_contra(tentativa: str, secretas: list[str]) -> np.ndarray:
    """Feedback de uma tentativa arbitrária (mesmo fora do léxico) contra uma lista.

    Caminho lento em Python puro — usado só quando a tentativa não tem linha na
    matriz. As listas envolvidas são pequenas no uso real.
    """
    return np.fromiter(
        (calcular_codigo(tentativa, secreta) for secreta in secretas),
        dtype=np.uint8,
        count=len(secretas),
    )


if __name__ == "__main__":
    import sys

    from .lexico import Lexico

    sys.stdout.reconfigure(encoding="utf-8")
    # A matriz é sempre sobre a forma normalizada (§2.4).
    palavras = Lexico.carregar().palavras
    matriz = carregar_matriz(palavras, forcar="--forcar" in sys.argv)
    print(f"forma={matriz.shape} dtype={matriz.dtype} {matriz.nbytes / 1e6:.0f} MB")
    print(f"padrões distintos observados: {len(np.unique(matriz))}/{N_PADROES}")
