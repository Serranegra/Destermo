#!/usr/bin/env python
"""Gráficos e tabelas a partir dos JSONs do benchmark (seção 8, passo 8).

    python benchmark.py --bateria realista
    python benchmark.py --bateria completo
    python benchmark.py --varredura-t
    python analise.py

Gera em `resultados/`:
    distribuicao_<bateria>.png   distribuição de tentativas por estratégia
    curva_temperatura.png        tentativas médias e taxa de vitória vs. T
    tabela.md                    a mesma informação em texto (table view)

Cada gráfico sai em duas versões, clara e escura — a escura usa passos próprios
das mesmas rampas, não uma inversão automática.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"

# Paleta categórica de referência da skill de dataviz, slots 1-4, usada sem
# alteração. Passa os limiares de CVD na lista de pares adjacentes (barras
# agrupadas), que é o caso aqui.
TEMAS = {
    "claro": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "superficie": "#fcfcfb",
        "tinta": "#0b0b0b",
        "tinta2": "#52514e",
        "apagada": "#898781",
        "grade": "#e1e0d9",
        "eixo": "#c3c2b7",
    },
    "escuro": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "superficie": "#1a1a19",
        "tinta": "#ffffff",
        "tinta2": "#c3c2b7",
        "apagada": "#898781",
        "grade": "#2c2c2a",
        "eixo": "#383835",
    },
}

FONTES = ["Segoe UI", "DejaVu Sans", "sans-serif"]
N_MAX = 6


def carregar(nome: str) -> list[dict] | None:
    caminho = DIR_RESULTADOS / nome
    if not caminho.exists():
        print(f"  (pulando {nome}: rode o benchmark correspondente primeiro)")
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def aplicar_tema(figura, eixos_lista, tema: dict) -> None:
    figura.patch.set_facecolor(tema["superficie"])
    for eixos in eixos_lista:
        eixos.set_facecolor(tema["superficie"])
        eixos.tick_params(colors=tema["apagada"], length=0, labelsize=9)
        for lado in ("top", "right"):
            eixos.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eixos.spines[lado].set_color(tema["eixo"])
            eixos.spines[lado].set_linewidth(1.0)
        # Grade em hairline sólido, um tom acima da superfície.
        eixos.yaxis.grid(True, color=tema["grade"], linewidth=1.0, zorder=0)
        eixos.set_axisbelow(True)


def grafico_distribuicao(resultados: list[dict], bateria: str, modo: str) -> Path:
    """Barras agrupadas: % dos jogos resolvidos em 1..6 tentativas, por estratégia."""
    tema = TEMAS[modo]
    plt.rcParams["font.family"] = FONTES

    categorias = [str(n) for n in range(1, N_MAX + 1)] + ["não\nresolveu"]
    figura, eixos = plt.subplots(figsize=(9.5, 5.0), dpi=170)

    n_series = len(resultados)
    # 2px de folga da superfície entre barras vizinhas, não uma borda desenhada.
    largura = 0.78 / n_series
    posicoes = range(len(categorias))

    for s, resultado in enumerate(resultados):
        distribuicao = {int(k): v for k, v in resultado["distribuicao"].items()}
        total = resultado["n_jogos"]
        derrotas = total - sum(distribuicao.values())
        alturas = [
            100 * distribuicao.get(n, 0) / total for n in range(1, N_MAX + 1)
        ] + [100 * derrotas / total]
        deslocamento = (s - (n_series - 1) / 2) * largura
        barras = eixos.bar(
            [p + deslocamento for p in posicoes],
            alturas,
            width=largura * 0.92,
            color=tema["series"][s],
            label=resultado["estrategia"],
            zorder=3,
        )
        # Rótulo direto só na barra modal de cada estratégia — não em todas.
        modal = max(range(len(alturas)), key=lambda i: alturas[i])
        eixos.annotate(
            f"{alturas[modal]:.0f}%",
            (barras[modal].get_x() + barras[modal].get_width() / 2, alturas[modal]),
            textcoords="offset points", xytext=(0, 4), ha="center",
            fontsize=8.5, color=tema["tinta2"], zorder=4,
        )

    eixos.set_xticks(list(posicoes))
    eixos.set_xticklabels(categorias)
    eixos.set_xlabel("tentativas até acertar", color=tema["tinta2"], fontsize=10,
                     labelpad=8)
    eixos.set_ylabel("% dos jogos", color=tema["tinta2"], fontsize=10, labelpad=8)
    eixos.set_title(
        f"Distribuição de tentativas — bateria {bateria} "
        f"({resultados[0]['n_jogos']} palavras)",
        color=tema["tinta"], fontsize=13, pad=14, loc="left", fontweight="medium",
    )
    legenda = eixos.legend(
        frameon=False, loc="upper right", fontsize=9.5, ncol=2, handlelength=1.2
    )
    for texto in legenda.get_texts():
        texto.set_color(tema["tinta2"])

    aplicar_tema(figura, [eixos], tema)
    figura.tight_layout()
    sufixo = "" if modo == "claro" else "_dark"
    caminho = DIR_RESULTADOS / f"distribuicao_{bateria}{sufixo}.png"
    figura.savefig(caminho, facecolor=tema["superficie"])
    plt.close(figura)
    return caminho


def grafico_temperatura(resultados: list[dict], modo: str) -> Path:
    """Duas medidas de escalas diferentes = dois painéis, nunca dois eixos y."""
    tema = TEMAS[modo]
    plt.rcParams["font.family"] = FONTES

    rotulos = [
        "∞" if r["temperatura"] is None or r["temperatura"] > 1e12
        else f"{r['temperatura']:g}"
        for r in resultados
    ]
    posicoes = list(range(len(resultados)))
    medias = [r["media_penalizada"] for r in resultados]
    vitorias = [100 * r["taxa_vitoria"] for r in resultados]

    figura, (cima, baixo) = plt.subplots(2, 1, figsize=(8.0, 6.6), dpi=170, sharex=True)

    for eixos, valores, titulo, unidade in (
        (cima, medias, "Tentativas médias (derrota conta como 7)", ""),
        (baixo, vitorias, "Taxa de vitória em 6 tentativas", "%"),
    ):
        eixos.plot(posicoes, valores, color=tema["series"][0], linewidth=2.0,
                   marker="o", markersize=8, markeredgewidth=2,
                   markeredgecolor=tema["superficie"], zorder=3)
        eixos.set_title(titulo, color=tema["tinta"], fontsize=11, loc="left", pad=10)

        if unidade == "%":
            # Escala ancorada em 100% e com pelo menos 3 pontos percentuais de
            # amplitude: uma diferença de 0,1 pp tem que PARECER 0,1 pp. Esticar o
            # eixo até a variação preencher o painel transformaria ruído em cliff.
            eixos.set_ylim(min(97.0, min(valores) - 0.5), 100.4)
            eixos.set_yticks([97, 98, 99, 100])
            formatar = lambda v: f"{v:.1f}%"  # noqa: E731
            destaque = min(range(len(valores)), key=lambda i: valores[i])
        else:
            margem = (max(valores) - min(valores) or 1) * 0.28
            eixos.set_ylim(min(valores) - margem, max(valores) + margem * 1.5)
            formatar = lambda v: f"{v:.3f}"  # noqa: E731
            destaque = min(range(len(valores)), key=lambda i: valores[i])

        # Rótulos diretos só nos extremos e no melhor ponto — nunca em todos.
        for i in {0, len(valores) - 1, destaque}:
            acima = i != destaque or valores[i] == max(valores)
            eixos.annotate(
                formatar(valores[i]), (posicoes[i], valores[i]),
                textcoords="offset points", xytext=(0, 11 if acima else -18),
                ha="center", fontsize=9, color=tema["tinta2"], zorder=4,
            )

    baixo.set_xticks(posicoes)
    baixo.set_xticklabels(rotulos)
    baixo.set_xlabel(
        "temperatura T do prior   (T→∞ = entropia pura, sem prior de frequência)",
        color=tema["tinta2"], fontsize=10, labelpad=8,
    )
    figura.suptitle(
        f"Efeito do prior de frequência — bateria {resultados[0]['bateria']} "
        f"({resultados[0]['n_jogos']} palavras)",
        color=tema["tinta"], fontsize=13, x=0.02, ha="left", y=0.98,
        fontweight="medium",
    )

    aplicar_tema(figura, [cima, baixo], tema)
    figura.subplots_adjust(top=0.88, bottom=0.11, left=0.10, right=0.97, hspace=0.30)
    sufixo = "" if modo == "claro" else "_dark"
    caminho = DIR_RESULTADOS / f"curva_temperatura{sufixo}.png"
    figura.savefig(caminho, facecolor=tema["superficie"])
    plt.close(figura)
    return caminho


def tabela_markdown(blocos: dict[str, list[dict]]) -> Path:
    """Table view: todo valor dos gráficos também legível como texto."""
    linhas = ["# Resultados do benchmark", ""]
    for titulo, resultados in blocos.items():
        if not resultados:
            continue
        linhas += [
            f"## {titulo}",
            "",
            "| estratégia | T | média | média penal. | vitória | "
            + " | ".join(str(n) for n in range(1, N_MAX + 1))
            + " | não resolveu | s/jogo |",
            "|" + "---|" * (7 + N_MAX),  # 5 fixas + N_MAX + "não resolveu" + s/jogo
        ]
        for r in resultados:
            distribuicao = {int(k): v for k, v in r["distribuicao"].items()}
            derrotas = r["n_jogos"] - sum(distribuicao.values())
            temperatura = (
                "∞" if r["temperatura"] is None or r["temperatura"] > 1e12
                else f"{r['temperatura']:g}"
            )
            linhas.append(
                f"| {r['estrategia']} | {temperatura} | {r['media_tentativas']:.3f} "
                f"| {r['media_penalizada']:.3f} | {r['taxa_vitoria']:.1%} | "
                + " | ".join(str(distribuicao.get(n, 0)) for n in range(1, N_MAX + 1))
                + f" | {derrotas} | {r['segundos_por_jogo']:.4f} |"
            )
        piores = [r for r in resultados if r["derrotas"] or r["piores_casos"]]
        if piores:
            linhas += ["", "Piores casos:", ""]
            for r in piores:
                nao_resolvidas = ", ".join(r["derrotas"][:10]) or "—"
                linhas.append(
                    f"- **{r['estrategia']}** — não resolveu: {nao_resolvidas}"
                )
        linhas.append("")
    caminho = DIR_RESULTADOS / "tabela.md"
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho


def main() -> None:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    blocos: dict[str, list[dict]] = {}
    for bateria in ("realista", "completo"):
        resultados = carregar(f"comparacao_{bateria}.json")
        if resultados:
            blocos[f"Comparação de estratégias — bateria {bateria}"] = resultados
            for modo in TEMAS:
                print(f"  {grafico_distribuicao(resultados, bateria, modo)}")

    for bateria in ("realista", "completo"):
        varredura = carregar(f"varredura_t_{bateria}.json")
        if varredura:
            blocos[f"Varredura de temperatura — bateria {bateria}"] = varredura
            for modo in TEMAS:
                print(f"  {grafico_temperatura(varredura, modo)}")
            break

    if blocos:
        print(f"  {tabela_markdown(blocos)}")
    else:
        print("nada a analisar — rode o benchmark primeiro")


if __name__ == "__main__":
    main()
