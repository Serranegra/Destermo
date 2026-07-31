#!/usr/bin/env python
"""Gráficos e tabelas a partir dos JSONs do benchmark (seção 8, passo 8).

    python benchmark.py --bateria realista
    python benchmark.py --bateria completo
    python benchmark.py --varredura-t
    python benchmark.py --nivel3
    python benchmark.py --serao
    python analise.py

Gera em `resultados/`:
    distribuicao_<bateria>.png        distribuição de tentativas por estratégia
    distribuicao_nivel3_<bateria>.png o mesmo, nível 2 vs nível 3
    curva_temperatura.png             tentativas médias e taxa de vitória vs. T
    grade_n_t.png                     melhor abertura em cada mundo (N, T)
    fronteira_h_icf.png               entropia contra frequência, palavra a palavra
    tabela.md                         a mesma informação em texto (table view)

Cada gráfico sai em duas versões, clara e escura — a escura usa passos próprios
das mesmas rampas, não uma inversão automática.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"

# Paleta da marca (brand/README.md), uma rampa por matiz. Duas decisões que não
# são estéticas:
#
# 1. O verde é exclusivo da série de entropia. No vocabulário do Termo ele
#    significa "acerto", então só a estratégia vencedora o usa — e isso resolve
#    de quebra o problema de encaixar 4 séries categóricas em 3 matizes, que não
#    tem solução com dois verdes (o verde da marca precisa escurecer para ter
#    contraste sobre o osso, e aí os dois verdes colidem).
# 2. O dourado fica no slot 0, não-adjacente ao verde: dourado ao lado de verde
#    é justamente o par que colapsa sob deuteranopia.
#
# Verificado por simulação de CVD (deutan/protan/tritan): todo par de séries,
# adjacente ou não, fica acima de 20 de dE76, e cada série tem ao menos 3:1 de
# contraste contra a superfície. A paleta de referência anterior passava nos
# pares adjacentes mas colapsava em 5,7 no par laranja/dourado.
#
# A ordem dos slots segue a da tabela do benchmark: aleatória, freq. de letras,
# mais provável, entropia. Quem escolhe o slot de cada série é `cores_das_series`,
# pelo NOME da estratégia — desde que o nível 3 entrou, a posição na lista não
# identifica mais a série (no head-to-head são só duas, e o verde é de uma delas).
TEMAS = {
    "claro": {
        # Passos escuros das rampas — o osso é claro demais para os hex nominais.
        "series": ["#96702C", "#8B767D", "#4A3E43", "#2D7D72"],
        "superficie": "#F5EFE9",  # osso
        "tinta": "#221C1E",  # ardósia
        "tinta2": "#5C4A50",
        "apagada": "#6E5C62",  # xisto
        "grade": "#E5DCD3",
        "eixo": "#D2C6BB",
        "destaque": "#2D7D72",
    },
    "escuro": {
        "series": ["#D3AD69", "#CFC2C7", "#96818A", "#3AA394"],
        "superficie": "#221C1E",  # ardósia
        "tinta": "#F5EFE9",  # osso
        "tinta2": "#A89AA0",
        "apagada": "#9A888E",  # xisto clareado: 4,5:1 sobre a ardósia
        "grade": "#35292D",
        "eixo": "#4A3B40",
        "destaque": "#3AA394",  # verde
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


def cores_das_series(resultados: list[dict], tema: dict) -> list[str]:
    """Uma cor por série, escolhida pelo PAPEL da estratégia e não pela posição.

    O verde é exclusivo da série de entropia (brand/README.md), então ele sai do
    nome da estratégia. O nível 3 fica no xisto claro (slot 1) e não no dourado:
    no head-to-head as duas séries são vizinhas, e dourado ao lado de verde é
    justamente o par que colapsa sob deuteranopia. As demais preenchem os slots
    livres na ordem em que aparecem, o que reproduz a paleta original da tabela
    de quatro estratégias.
    """
    preferido = {"entropia": 3, "nível 3": 1}
    slots: list[int | None] = [None] * len(resultados)
    livres = list(range(len(tema["series"])))
    for i, resultado in enumerate(resultados):
        for prefixo, slot in preferido.items():
            if resultado["estrategia"].startswith(prefixo) and slot in livres:
                slots[i] = slot
                livres.remove(slot)
                break
    for i, slot in enumerate(slots):
        if slot is None:
            if livres:
                slots[i] = livres.pop(0)
                continue
            # A paleta da marca tem 4 cores e não se inventa uma quinta sem
            # refazer a verificação de CVD (brand/README.md). Duas séries com a
            # mesma cor é um gráfico ilegível, então isto avisa em vez de sair
            # calado — o previsto é comparar 4 estratégias ou fazer o
            # head-to-head de 2, nunca 5 de uma vez.
            slots[i] = i % len(tema["series"])
            print(f"  ! {len(resultados)} séries e só {len(tema['series'])} cores: "
                  f"'{resultados[i]['estrategia']}' repete a cor de outra série")
    return [tema["series"][slot] for slot in slots]


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


def grafico_distribuicao(
    resultados: list[dict],
    bateria: str,
    modo: str,
    nome: str | None = None,
    subtitulo: str = "",
) -> Path:
    """Barras agrupadas: % dos jogos resolvidos em 1..6 tentativas, por estratégia."""
    tema = TEMAS[modo]
    plt.rcParams["font.family"] = FONTES

    categorias = [str(n) for n in range(1, N_MAX + 1)] + ["não\nresolveu"]
    figura, eixos = plt.subplots(figsize=(9.5, 5.0), dpi=170)

    cores = cores_das_series(resultados, tema)
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
            color=cores[s],
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
        f"({resultados[0]['n_jogos']} palavras){subtitulo}",
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
    caminho = DIR_RESULTADOS / f"{nome or f'distribuicao_{bateria}'}{sufixo}.png"
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
        eixos.plot(posicoes, valores, color=tema["destaque"], linewidth=2.0,
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


def grafico_grade_n_t(experimento: dict, modo: str) -> Path:
    """Mapa (N, T): onde a abertura dos fóruns é de fato a melhor.

    As duas varreduras de uma dimensão só — cortar a lista, baixar o prior — dão
    respostas que parecem contraditórias. Vistas como os dois eixos de um mapa,
    param de ser: são regiões vizinhas, e as duas posições conhecidas (o padrão do
    projeto e a hipótese do fórum) são dois pontos marcados nele.

    O verde não é decorativo aqui: no vocabulário do Termo ele é "acerto", e é
    exatamente o que a célula significa — `serão` é a melhor abertura ali. A rampa
    é de matiz único, então a leitura sobrevive a qualquer daltonismo; e o nome da
    vencedora vai escrito em toda célula, então a cor nunca é o único canal.

    A distância é RELATIVA à entropia da melhor abertura da célula, não em bits
    absolutos. Com T baixo a massa toda cabe em poucas palavras e a melhor jogada
    do mundo vale 0,9 bit: ali as diferenças em bits somem por construção, e o
    número absoluto pintaria de verde ("`serão` é ótimo!") um canto onde na verdade
    nenhuma abertura separa coisa alguma.
    """
    tema = TEMAS[modo]
    plt.rcParams["font.family"] = FONTES
    grade = experimento["grade"]
    alvo = experimento["palavra"]

    cortes = grade["cortes"]
    temperaturas = grade["temperaturas"]
    celula = {(c["n"], c["t"]): c for c in grade["celulas"]}
    distancias = [
        [100 * celula[(n, t)]["distancia"] / max(celula[(n, t)]["entropia_melhor"], 1e-9)
         for t in temperaturas]
        for n in cortes
    ]

    # A rampa termina no tom da grade, não na superfície: uma célula ruim ainda
    # tem que se ler como célula, e não como buraco no gráfico.
    rampa = LinearSegmentedColormap.from_list(
        "destermo", [tema["destaque"], tema["grade"]]
    )
    figura, eixos = plt.subplots(figsize=(10.0, 6.6), dpi=170)
    limite = max(max(linha) for linha in distancias)
    imagem = eixos.imshow(distancias, cmap=rampa, vmin=0.0, vmax=limite, aspect="auto")

    # As duas posições conhecidas, se a grade do benchmark ainda as contiver —
    # `CORTES_GRADE` e `TEMPERATURAS_GRADE` são constantes de lá, não daqui.
    conhecidas = {}
    if 1.0 in temperaturas:
        conhecidas[(len(cortes) - 1, temperaturas.index(1.0))] = "padrão do projeto"
    if 300 in cortes:
        conhecidas[(cortes.index(300), 0)] = "hipótese do fórum"
    for i, n in enumerate(cortes):
        for j, t in enumerate(temperaturas):
            atual = celula[(n, t)]
            # Texto claro sobre as células escuras (perto de 0) e escuro sobre as
            # claras — o limiar segue a rampa, não o valor absoluto.
            escura = distancias[i][j] < limite * 0.45
            rotulo = conhecidas.get((i, j))
            eixos.text(
                j, i - (0.13 if rotulo else 0), atual["melhor"],
                ha="center", va="center", fontsize=8.5,
                color=tema["superficie"] if escura else tema["tinta"],
                fontweight="bold" if atual["distancia"] == 0 else "normal",
            )
            if rotulo:
                # Dentro da célula: fora dela o rótulo cai por cima da vizinha.
                eixos.text(
                    j, i + 0.19, rotulo, ha="center", va="center", fontsize=7.5,
                    style="italic",
                    color=tema["superficie"] if escura else tema["tinta2"],
                )

    eixos.set_xticks(range(len(temperaturas)))
    eixos.set_xticklabels(["∞" if t > 1e12 else f"{t:g}" for t in temperaturas])
    eixos.set_yticks(range(len(cortes)))
    eixos.set_yticklabels([f"{n}" for n in cortes])
    eixos.set_xlabel("temperatura T do prior dentro do corte   (∞ = uniforme)",
                     color=tema["tinta2"], fontsize=10, labelpad=8)
    eixos.set_ylabel("N — quantas palavras podem ser a resposta",
                     color=tema["tinta2"], fontsize=10, labelpad=8)
    eixos.set_title(
        f"Melhor abertura em cada mundo (N, T) — e o quanto '{alvo}' fica atrás",
        color=tema["tinta"], fontsize=13, pad=38, loc="left", fontweight="medium",
    )
    eixos.text(
        0, 1.015,
        "à direita de T=0,5 a massa cabe em poucas palavras (a melhor abertura vale "
        "0,9 bit em T=0,1):\nlá nenhuma abertura separa nada, e o desacordo some por "
        "construção",
        transform=eixos.transAxes, fontsize=8.5, color=tema["apagada"], va="bottom",
    )

    for (i, j) in conhecidas:
        eixos.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False,
            edgecolor=tema["tinta"], linewidth=1.8, zorder=5,
        ))

    barra = figura.colorbar(imagem, ax=eixos, pad=0.02)
    barra.set_label(f"% da entropia da melhor abertura que '{alvo}' perde para ela",
                    color=tema["tinta2"], fontsize=9.5)
    barra.ax.tick_params(colors=tema["apagada"], length=0, labelsize=8.5)
    barra.outline.set_edgecolor(tema["eixo"])

    figura.patch.set_facecolor(tema["superficie"])
    eixos.tick_params(colors=tema["apagada"], length=0, labelsize=9)
    for lado in eixos.spines:
        eixos.spines[lado].set_color(tema["eixo"])
    figura.tight_layout()
    sufixo = "" if modo == "claro" else "_dark"
    caminho = DIR_RESULTADOS / f"grade_n_t{sufixo}.png"
    figura.savefig(caminho, facecolor=tema["superficie"])
    plt.close(figura)
    return caminho


def grafico_fronteira(experimento: dict, modo: str) -> Path:
    """Entropia contra frequência: a troca que produz o desacordo inteiro.

    Uma abertura quer duas coisas incompatíveis — separar muito o conjunto e poder
    ser a resposta. A fronteira de Pareto é o conjunto das palavras em que ganhar
    numa exige perder na outra, e todas as candidatas a "melhor abertura" que
    apareceram nesta investigação estão nela. Não há erro de cálculo em lado
    nenhum: cada um escolheu um ponto diferente da mesma curva.
    """
    tema = TEMAS[modo]
    plt.rcParams["font.family"] = FONTES
    nuvem = experimento["nuvem"]
    alvo = nuvem["alvo"]

    figura, eixos = plt.subplots(figsize=(9.5, 5.8), dpi=170)
    # Recorte no topo: a nuvem desce até ~1,7 bit, e 4.000 palavras ruins não
    # informam nada sobre qual é a melhor abertura — só achatam a fronteira.
    piso = 4.55
    teto = max(nuvem["entropia"]) + 0.30

    corte = nuvem["cortes_icf"][str(300)]
    eixos.axvspan(min(nuvem["icf"]) - 0.4, corte, color=tema["grade"], zorder=0)
    eixos.annotate(
        "as 300 mais comuns — a lista que o fórum supõe",
        (min(nuvem["icf"]) - 0.2, teto - 0.02), ha="left", va="top",
        fontsize=8.5, color=tema["apagada"], zorder=2,
    )

    eixos.scatter(nuvem["icf"], nuvem["entropia"], s=5, color=tema["apagada"],
                  alpha=0.16, linewidths=0, zorder=1, rasterized=True)

    # Top-10 em entropia: todas raras, e é esse o ponto — ficam à direita da faixa.
    # Quem já está na fronteira ganha marcador aqui mas não um segundo rótulo.
    na_fronteira = {linha[0] for linha in nuvem["fronteira"]}
    topo = [linha for linha in nuvem["anotar"] if linha[0] != alvo]
    eixos.scatter([l[1] for l in topo], [l[2] for l in topo], s=26,
                  color=tema["tinta2"], zorder=3)
    # Alterna acima/abaixo pela contagem do que foi DESENHADO, não do laço: as
    # top-10 se amontoam numa faixa estreita de y e a paridade do índice original
    # deixaria vizinhas do mesmo lado.
    desenhadas = 0
    for palavra, icf, h in sorted(topo, key=lambda l: l[1]):
        if palavra in na_fronteira:
            continue
        eixos.annotate(
            palavra, (icf, h), textcoords="offset points",
            xytext=(0, 9 if desenhadas % 2 == 0 else -16), ha="center",
            fontsize=8, color=tema["tinta2"], zorder=4,
        )
        desenhadas += 1

    fronteira = nuvem["fronteira"]
    eixos.step([l[1] for l in fronteira], [l[2] for l in fronteira], where="post",
               color=tema["destaque"], linewidth=1.6, zorder=4)
    eixos.scatter([l[1] for l in fronteira], [l[2] for l in fronteira], s=34,
                  color=tema["destaque"], zorder=5)
    # A fronteira sobe da esquerda para a direita, então rótulo à esquerda-acima
    # não cobre o próximo ponto; os dois últimos saem por baixo para não brigar
    # com o anel do alvo, que fica entre eles.
    for k, (palavra, icf, h) in enumerate(fronteira):
        if palavra == alvo:
            continue
        abaixo = k >= len(fronteira) - 2
        eixos.annotate(
            palavra, (icf, h), textcoords="offset points",
            xytext=(6, -12) if abaixo else (-7, 4),
            ha="left" if abaixo else "right",
            fontsize=8.5, color=tema["destaque"], zorder=6,
        )

    # O alvo se distingue por FORMA, não por cor: um anel de alto contraste. Um
    # segundo matiz aqui seria o dourado ao lado do verde, que é o par que colapsa
    # sob deuteranopia (brand/README.md).
    ponto = next(l for l in nuvem["anotar"] if l[0] == alvo)
    eixos.scatter([ponto[1]], [ponto[2]], s=150, facecolors="none",
                  edgecolors=tema["tinta"], linewidths=2.0, zorder=7)
    eixos.annotate(
        alvo, (ponto[1], ponto[2]), textcoords="offset points", xytext=(0, 14),
        ha="center", fontsize=10, color=tema["tinta"], fontweight="bold", zorder=8,
    )

    eixos.set_xlabel(
        "ICF — inverse corpus frequency  (≈ −log da frequência; à esquerda = comum)",
        color=tema["tinta2"], fontsize=10, labelpad=8,
    )
    eixos.set_ylabel("entropia da abertura (bits)", color=tema["tinta2"],
                     fontsize=10, labelpad=8)
    eixos.set_title(
        "Informar muito e poder ser a resposta são objetivos opostos",
        color=tema["tinta"], fontsize=13, pad=36, loc="left", fontweight="medium",
    )
    eixos.text(
        0, 1.012,
        f"a fronteira de Pareto tem {len(fronteira)} palavras; toda candidata a "
        "melhor abertura desta investigação está nela\n(recorte no topo: a nuvem "
        "continua até ~1,7 bit)",
        transform=eixos.transAxes, fontsize=8.5, color=tema["apagada"], va="bottom",
    )
    eixos.set_ylim(piso, teto)

    aplicar_tema(figura, [eixos], tema)
    figura.tight_layout()
    sufixo = "" if modo == "claro" else "_dark"
    caminho = DIR_RESULTADOS / f"fronteira_h_icf{sufixo}.png"
    figura.savefig(caminho, facecolor=tema["superficie"])
    plt.close(figura)
    return caminho


def tabela_arrependimento(experimento: dict) -> list[str]:
    """A matriz de arrependimento em texto, para a table view de `tabela.md`."""
    dados = experimento["arrependimento"]
    mundos, aberturas = dados["mundos"], dados["aberturas"]
    linhas = [
        "## Arrependimento por mundo — qual abertura é a menos pior",
        "",
        "Média penalizada em cada mundo (respostas cortadas, léxico inteiro "
        "digitável), e entre parênteses o quanto a abertura perde para a melhor "
        "daquele mundo.",
        "",
        "| abertura | " + " | ".join(mundos) + " | pior caso |",
        "|" + "---|" * (len(mundos) + 2),
    ]
    for palavra in aberturas:
        celulas = [
            f"{dados['medias'][palavra][m]:.3f} (+{dados['arrependimento'][palavra][m]:.3f})"
            for m in mundos
        ]
        marca = " **←**" if palavra == dados["vencedora_minimax"] else ""
        linhas.append(
            f"| {palavra} | " + " | ".join(celulas)
            + f" | {dados['pior_caso'][palavra]:.3f}{marca} |"
        )
    linhas += ["", f"Menor arrependimento de pior caso: **{dados['vencedora_minimax']}**.", ""]
    return linhas


def tabela_markdown(blocos: dict[str, list[dict]],
                    extras: list[str] | None = None) -> Path:
    """Table view: todo valor dos gráficos também legível como texto."""
    linhas = ["# Resultados do benchmark", ""]
    linhas += extras or []
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
        resultados = carregar(f"comparacao_nivel3_{bateria}.json")
        if resultados:
            blocos[f"Nível 2 vs nível 3 — bateria {bateria}"] = resultados
            for modo in TEMAS:
                caminho = grafico_distribuicao(
                    resultados, bateria, modo,
                    nome=f"distribuicao_nivel3_{bateria}",
                    subtitulo="\nproxy da entropia vs. objetivo real",
                )
                print(f"  {caminho}")

    for bateria in ("realista", "completo"):
        varredura = carregar(f"varredura_t_{bateria}.json")
        if varredura:
            blocos[f"Varredura de temperatura — bateria {bateria}"] = varredura
            for modo in TEMAS:
                print(f"  {grafico_temperatura(varredura, modo)}")
            break

    extras: list[str] = []
    serao = carregar("serao.json")
    if serao:
        experimento = serao["experimento"]
        extras = tabela_arrependimento(experimento)
        for modo in TEMAS:
            print(f"  {grafico_grade_n_t(experimento, modo)}")
            print(f"  {grafico_fronteira(experimento, modo)}")

    if blocos or extras:
        print(f"  {tabela_markdown(blocos, extras)}")
    else:
        print("nada a analisar — rode o benchmark primeiro")


if __name__ == "__main__":
    main()
