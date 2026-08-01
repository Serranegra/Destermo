#!/usr/bin/env python
"""Interface web do solver — a mesma camada fina que `solver.py` é na CLI.

Aqui não há lógica de solver nenhuma: o motor de `termo/` continua sendo a única
fonte de verdade (§7.4). O que este arquivo faz é

  1. guardar a partida no `session_state` como uma lista de (tentativa, padrão),
  2. reconstruir as candidatas a partir dela a cada rerun, e
  3. desenhar o tabuleiro nas cores de `brand/`.

O passo 2 merece uma palavra. O Streamlit reexecuta o script inteiro a cada
clique, então a tentação é guardar o `np.ndarray` de candidatas no estado e ir
filtrando. Guardar o HISTÓRICO em vez disso custa uma refiltragem por rodada
(microssegundos: é uma linha da matriz e uma comparação) e ganha duas coisas —
desfazer vira `pop()`, e trocar T ou o nível no meio da partida recalcula tudo
sozinho, sem estado obsoleto.

    streamlit run app.py
"""

from __future__ import annotations

import html
import math
import threading
import urllib.parse
from pathlib import Path

import numpy as np
import streamlit as st

from termo.entropia import N_MAX_TENTATIVAS, Motor, Sugestao
from termo.feedback import (
    TAMANHO,
    normalizar,
    padrao_para_codigo,
    padrao_possivel,
)
from termo.lexico import T_PADRAO, Lexico
from termo.matriz import carregar_matriz
from termo.nivel3 import BEAM, PROFUNDIDADE, TETO_INTERATIVO, MotorNivel3

RAIZ = Path(__file__).resolve().parent
MARCA = RAIZ / "brand"

PADRAO_VITORIA = "G" * TAMANHO

# Paleta de brand/README.md. O tabuleiro é o único lugar do app em que as três
# cores de feedback aparecem juntas — e é justamente onde elas já significam o
# que a marca diz que significam (acerto, fora de lugar, ausente).
VERDE, DOURADO, XISTO = "#3AA394", "#D3AD69", "#6E5C62"
ARDOSIA, OSSO = "#221C1E", "#F5EFE9"

CLASSE = {"G": "dt-verde", "Y": "dt-dourado", "B": "dt-xisto"}
FUNDO = {"G": (VERDE, ARDOSIA), "Y": (DOURADO, ARDOSIA), "B": (XISTO, OSSO)}

# Estado de uma casa da linha em edição. "·" é "ainda não mexeram nela", e não é o
# mesmo que preto: um feedback do Termo é quase todo preto, então exigir cinco
# cliques para registrar seria um imposto por rodada. Na hora de aplicar, "·" vira
# preto; até lá ele se desenha vazio, que é o que o jogo mostra numa linha em branco.
VAZIA = "·"
CICLO = {VAZIA: "B", "B": "Y", "Y": "G", "G": "B"}
LIMPA = VAZIA * TAMANHO

# Quantas alternativas o cartão mostra. O corte é na EXIBIÇÃO, não no motor: a
# abertura vem do `data/aberturas*.json` com cinco já gravadas, e pedir menos ao
# motor não encurtaria a que está em cache — só invalidaria o cache das outras.
N_ALTERNATIVAS = 3

# "Nível 2" e "nível 3" são nomes DE DENTRO: numeram a ordem em que os algoritmos
# foram escritos e o quanto cada um corrige o anterior. Isso interessa a quem lê o
# README, não a quem só quer a próxima palavra — para essa pessoa, "3" não é maior
# nem melhor que "2", é só um número sem referência. Aqui os dois aparecem pelo
# que fazem, e o número não vaza para a tela em lugar nenhum.
RESOLVEDOR = {3: "menor número de tentativas", 2: "maior ganho informacional"}

# O motor assina o próprio nome no `motivo` da abertura ("melhor abertura do
# nível 3 (em cache)"). Ele está certo em fazer isso: a CLI mostra o mesmo campo,
# e lá o número é o da opção `--nivel`. Quem tira o número da frase é esta camada
# de exibição, não o motor — inverter isso poria texto de interface dentro de
# `termo/`, que é justamente o que este arquivo existe para não fazer.
SEM_NIVEL = {f" do nível {n}": "" for n in RESOLVEDOR}

MONO = "ui-monospace, 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"


def mascara(nome: str) -> str:
    """SVG de `brand/` como máscara CSS, para ele pegar a cor do texto do tema.

    `st.html` sanitiza a marcação e come qualquer `<svg>` inline, então o caminho
    de embutir o arquivo no DOM está fechado. Um `<img>` funcionaria, mas o
    `currentColor` do lockup morreria com ele — dentro de um `<img>` não há
    contexto de cor, e a marca sairia preta sobre a ardósia. Como máscara, o SVG
    entra só pela silhueta e a cor vem do `background-color`: um `currentColor`
    ali resolve para o texto do tema, que é o comportamento que os ativos da
    marca pedem (ver `destermo-icon-mono.svg` em brand/README.md).
    """
    svg = (MARCA / nome).read_text(encoding="utf-8")
    return "url('data:image/svg+xml;utf8," + urllib.parse.quote(svg) + "')"


CSS = f"""
<style>
/* A coluna do Streamlit tem 704px e o tabuleiro tem 296: sem estreitar, a grade
   fica ilhada no meio de um cartão, um campo e um botão do dobro da largura dela.
   O respiro de fábrica (96px no topo, 160px embaixo) vai junto — é generoso para
   um relatório, folgado demais para uma tela de uma coisa só.
   Mas há um piso no topo: a barra do Streamlit tem 60px e é `position:absolute`
   por cima do conteúdo, não acima dele. Foi por isso que os 96px existiam. Com
   menos de 60 o lockup entra embaixo dela e aparece cortado — 72 deixa folga. */
.stMainBlockContainer {{ max-width:440px !important;
                         padding-top:4.5rem !important;
                         padding-bottom:5rem !important; }}

.dt-lockup {{ height:38px; width:175px; /* viewBox 460x100 */
              background-color:currentColor;
              -webkit-mask:{mascara("destermo-lockup.svg")} no-repeat left center;
              -webkit-mask-size:contain;
              mask:{mascara("destermo-lockup.svg")} no-repeat left center;
              mask-size:contain; margin:.2rem 0 .25rem; }}
/* Mesma máscara, medida da barra lateral (212px úteis descontando o padding). */
.dt-lockup-mini {{ height:28px; width:129px; margin:0 0 .9rem; }}
/* Texto terciário por OPACIDADE, não pelo xisto nominal: #6E5C62 dá 2,4:1 sobre
   a ardósia (e o mesmo problema espelhado sobre o osso). Atenuar a cor do tema
   fica legível nos dois lados, sem uma segunda paleta só para o tema claro. */
.dt-sub {{ color:inherit; opacity:.62; font-size:.9rem; margin:0 0 1.4rem; }}

/* A medida da casa é uma variável porque duas coisas diferentes precisam bater no
   pixel: as linhas já jogadas, que são HTML nosso, e a linha em edição, que são
   cinco botões do Streamlit. */
:root {{ --dt-casa: clamp(38px, 11vw, 54px); --dt-vao: .4rem; }}

.dt-grade {{ display:flex; flex-direction:column; align-items:center;
             gap:var(--dt-vao); margin:0; }}
.dt-linha {{ display:flex; gap:var(--dt-vao); }}
.dt-casa {{ width:var(--dt-casa); aspect-ratio:1; border-radius:6px;
            display:flex; align-items:center; justify-content:center;
            font-family:{MONO}; font-weight:700;
            font-size:clamp(18px, 5.5vw, 26px); text-transform:uppercase;
            user-select:none; }}
.dt-verde   {{ background:{VERDE};   color:{ARDOSIA}; }}
.dt-dourado {{ background:{DOURADO}; color:{ARDOSIA}; }}
.dt-xisto   {{ background:{XISTO};   color:{OSSO}; }}
/* Vazia herda a cor do texto do tema: assim a grade não quebra se o usuário
   trocar para o tema claro pelo menu do Streamlit.
   A borda é mais fraca que a da linha em edição (que usa .6) de propósito: as
   rodadas futuras não têm nada a oferecer, e com o mesmo traço das duas a linha
   clicável não se distinguia das quatro mortas embaixo dela. */
.dt-vazia {{ background:transparent; border:2px solid rgba(110,92,98,.3);
             color:inherit; }}

/* A LINHA EM EDIÇÃO — cinco botões disfarçados de casa do tabuleiro.
   Só entram aqui seletores que o Streamlit promete: a classe `st-key-` + a chave,
   que ele carimba em qualquer elemento com `key`. Nada de hash de emotion, que
   muda a cada release. O `!important` é porque o hover padrão do botão tem
   especificidade maior que a nossa e repintaria a casa ao passar o mouse.

   CUIDADO ao editar QUALQUER comentário deste bloco: um sinal de menor que, mesmo
   dentro de comentário, faz o sanitizador do `st.html` jogar fora a folha INTEIRA,
   sem erro nenhum — a página só aparece sem estilo. */
/* `st-key-tabuleiro` É o bloco vertical, não um ancestral dele — daí não haver
   descendente aqui: o vão de 1rem que ele traz de fábrica partiria a grade. */
.st-key-tabuleiro {{ gap:var(--dt-vao) !important; }}
.st-key-linha_ativa {{ gap:var(--dt-vao) !important; }}
[class*="st-key-casa"] {{ width:var(--dt-casa) !important; flex:0 0 auto !important; }}
[class*="st-key-casa"] button {{
    width:100% !important; height:var(--dt-casa); min-height:0; padding:0;
    border-radius:6px; border:2px solid transparent;
    font-family:{MONO}; font-weight:700; line-height:1;
    font-size:clamp(18px, 5.5vw, 26px); text-transform:uppercase; }}
/* Sem `transition` no fundo da casa, por mais tentador que seja. Declaração de
   transição fica ACIMA de `!important` na cascata, e como cada clique remonta o
   botão a transição nunca terminava: a casa ficava congelada na cor de partida
   (transparente) e nenhuma regra conseguia pintá-la. */
[class*="st-key-casa"] button:hover {{ filter:brightness(1.15); }}
[class*="st-key-casa"] button p {{ font:inherit; }}
/* Foco de teclado na cor da marca: o anel padrão do Streamlit é o único lugar do
   tabuleiro onde entrava uma cor de fora da paleta. */
[class*="st-key-casa"] button:focus-visible {{
    outline:2px solid {VERDE} !important; outline-offset:2px;
    box-shadow:none !important; }}

.dt-cartao {{ background:rgba(58,163,148,.09); border:1px solid rgba(58,163,148,.35);
              border-left:4px solid {VERDE}; border-radius:8px;
              padding:1rem 1.15rem; margin:.2rem 0 1rem; }}
.dt-rotulo {{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
              color:{VERDE}; margin-bottom:.35rem; }}
/* Menor que os 2,3rem de antes: desde que a linha em edição passou a mostrar a
   sugestão em fantasma, a palavra aparece duas vezes na mesma dobra. */
.dt-palavra {{ font-family:{MONO}; font-weight:700; font-size:1.6rem;
               line-height:1.1; text-transform:uppercase; letter-spacing:.06em; }}
.dt-metricas {{ display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 .5rem; }}
.dt-chip {{ font-family:{MONO}; font-size:.78rem; padding:.18rem .5rem;
            border-radius:4px; background:rgba(110,92,98,.28); }}
.dt-chip-cand {{ background:rgba(58,163,148,.22); }}
.dt-motivo {{ font-size:.85rem; opacity:.75; }}
.dt-alt {{ font-family:{MONO}; font-size:.85rem; opacity:.8; margin-top:.55rem; }}
/* Destaque por PESO, não por dourado. Nas casas o dourado quer dizer "letra fora
   de lugar"; aqui ele queria dizer "também pode ser a resposta" — dois sentidos
   sem relação para a mesma cor, na mesma tela. O dourado volta a ter um só. */
.dt-alt b {{ color:inherit; font-weight:700; }}

/* AJUDA. A numeração vem de um contador de CSS, e não do marcador nativo da
   lista, porque o número precisa alinhar com um texto de várias linhas — o
   marcador de fábrica empurra o recuo para o lado. */
.dt-ajuda {{ font-size:.88rem; line-height:1.5;
             border:1px solid rgba(110,92,98,.35); border-radius:8px;
             padding:.55rem .9rem; margin:0 0 1rem; }}
.dt-ajuda[open] {{ padding-bottom:.9rem; }}
.dt-ajuda summary {{ cursor:pointer; font-size:.85rem; letter-spacing:.06em;
                     text-transform:uppercase; opacity:.72;
                     list-style-position:inside; }}
.dt-ajuda summary::marker {{ color:{VERDE}; }}
.dt-ajuda summary:focus-visible {{ outline:2px solid {VERDE}; outline-offset:3px;
                                   border-radius:4px; }}
.dt-ajuda ol {{ list-style:none; margin:.8rem 0 0; padding:0;
                counter-reset:passo; }}
.dt-ajuda li {{ counter-increment:passo; position:relative;
                padding-left:1.6rem; margin-bottom:.55rem; }}
.dt-ajuda li::before {{ content:counter(passo); position:absolute; left:0; top:0;
                        font-family:{MONO}; font-weight:700; color:{VERDE}; }}
.dt-ajuda p {{ margin:.6rem 0 0; opacity:.7; }}
/* Amostra da cor no meio da frase: a legenda diz o que cada cor quer dizer, e
   dizê-lo com a própria cor poupa o usuário de mapear nome para casa. */
.dt-amostra {{ display:inline-block; width:.72em; height:.72em; border-radius:3px;
               margin-right:.2em; vertical-align:baseline; }}
</style>
"""

# Um passo por ação do usuário, na ordem em que a tela as pede. O texto cita os
# rótulos dos botões ao pé da letra — instrução que parafraseia o botão faz o
# usuário procurar na tela um controle que não existe com aquele nome.
#
# NASCE RECOLHIDO, e fica a cargo do usuário abrir. Aberto de saída, o bloco
# empurra o cartão da sugestão para fora da primeira dobra — quem já sabe jogar
# paga o preço todo dia para que quem não sabe leia uma vez. Recolhido, o custo
# é uma linha, e a linha diz o que há atrás dela.
#
# `details` nosso e não `st.expander` porque este arquivo já desenha o próprio
# HTML e o bloco herda o CSS da marca junto com o resto — sem depender de um
# seletor interno do Streamlit para vestir o painel.
COMO_USAR = f"""
<details class="dt-ajuda">
<summary>como usar</summary>
<ol>
<li>Jogue no Termo a palavra sugerida no cartão verde. Se preferir usar outra
palavra, digite-a no campo e aperte <b>Usar esta palavra</b>. O tabuleiro passa a
mostrá-la.</li>
<li>Copie o feedback do Termo: clique em cada casa da linha ativa até ela ficar
da cor certa 
<span class="dt-amostra" style="background:{XISTO}"></span>ausente,
<span class="dt-amostra" style="background:{DOURADO}"></span>fora de lugar,
<span class="dt-amostra" style="background:{VERDE}"></span>no lugar.</li>
<li>Aperte <b>Registrar rodada</b>. A linha sobe para o tabuleiro e sai a
sugestão seguinte. Repita até acertar.</li>
</ol>
<p>Errou uma cor? <b>Voltar</b> desfaz a última rodada. Na barra lateral dá para
trocar o tipo de resolvedor e o tamanho do léxico.</p>
</details>
"""


# ------------------------------------------------------------------- motor


# A parte cara é a MATRIZ, e ela não depende nem do T nem do nível: as linhas são
# palavras e as colunas são palavras. Só o espaço a muda. Separar os dois caches é
# o que permite ser generoso no de baixo — o de cima guarda no máximo os dois
# espaços que existem (36 MB o padrão, 52 MB o ampliado), e é o único que custa.
@st.cache_resource(show_spinner=False, max_entries=2)
def obter_espaco(ampliado: bool) -> tuple[Lexico, np.ndarray]:
    """Léxico e matriz de padrões — o que é caro e o que é comum a toda config."""
    lexico = Lexico.carregar(T_PADRAO, ampliado)
    return lexico, carregar_matriz(lexico.sondas, lexico.palavras)


# Com a matriz vindo pronta de cima, montar um motor é recalcular o prior: um
# softmax sobre 6.046 floats, microssegundos. Antes um motor custava a matriz
# inteira, e por isso o teto aqui era 2 — o que fazia a barra lateral, com suas 20
# combinações, despejar um motor a cada dois cliques. E despejar um motor do
# nível 3 joga fora a memoização da busca junto, que é o que ele tem de caro.
@st.cache_resource(show_spinner=False, max_entries=8)
def obter_motor(
    temperatura: float, nivel: int, ampliado: bool, beam: int, profundidade: int
) -> Motor | MotorNivel3:
    """Motor pronto para esta configuração.

    `cache_resource` e não `cache_data` porque o motor não deve ser serializado a
    cada uso: ele carrega a matriz de 6.046x8.629 e memoiza escolhas por dentro,
    e é justamente essa memoização que queremos preservar entre reruns.
    """
    base, matriz = obter_espaco(ampliado)
    lexico = base if base.temperatura == temperatura else base.com_temperatura(
        temperatura
    )
    motor: Motor | MotorNivel3 = Motor(lexico, matriz)
    if nivel == 3:
        motor = MotorNivel3(motor, beam, profundidade)
    return motor


# Quantas buscas do nível 3 podem estar EM VOO ao mesmo tempo.
#
# Não é sobre vazão. O Streamlit atende cada sessão numa thread do mesmo
# processo, e a busca é Python com chamadas curtas de numpy: com um GIL só, as
# threads já se revezavam no processador. O que elas faziam era se revezar
# SEGURANDO MEMÓRIA cada uma — os temporários de `Motor.entropias` (a tabela de
# baldes e o buffer de log, dezenas de MB para conjuntos de algumas centenas de
# candidatas) ficam vivos enquanto a thread está suspensa no meio da varredura.
#
# Medido neste processo, com N sessões pedindo jogadas de estados DISTINTOS ao
# mesmo tempo (o pior caso: nenhuma delas acerta o cache). RSS de 71 MB com um
# motor carregado; o pico é o do processo inteiro:
#
#     N            8         16         32
#     sem fila   249 MB    309 MB    386 MB     parede 2,65s  4,02s  9,02s
#     4 vagas    202 MB    198 MB    211 MB     parede 2,51s  4,02s  8,65s
#
# Sem fila o pico cresce com a carga; com fila ele PARA de crescer, que é a
# propriedade que interessa numa página aberta ao público — o custo deixa de
# depender de quanta gente chega junto. E a parede é a mesma, dentro do ruído:
# nada foi perdido em vazão porque nada estava rodando em paralelo. O que muda é
# a latência mediana de cada sessão (2,2s -> 3,6s em N=32), que é a fila
# aparecendo: menos gente calculando ao mesmo tempo, cada uma esperando mais.
# Numa tela que já mostra "procurando a melhor jogada..." isso é o câmbio certo.
#
# 4 e não 2 ou 8 porque é onde a curva já está achatada: de 4 para 8 o pico
# quase não cai mais, e abaixo de 4 a fila começa a aparecer na mediana sem ter
# memória nenhuma a economizar em troca.
VAGAS_DE_BUSCA = threading.Semaphore(4)


# `cache_resource` e não `cache_data`: o `cache_data` guarda o valor por pickle, e
# a `Sugestao` que sai daqui vem de um motor guardado em `cache_resource`, que
# sobrevive aos reruns. Quando o Streamlit recarrega os módulos locais (o que a
# nuvem faz), o motor velho continua produzindo instâncias da classe velha, e o
# pickle recusa: "it's not the same object as termo.entropia.Sugestao". Como a
# `Sugestao` é só de leitura na tela, não há o que isolar por cópia — guardar a
# própria instância é correto e mais barato.
#
# `max_entries` era 64, e 64 é pouco: com a abertura fixa, a RODADA 2 sozinha tem
# 131 estados possíveis por configuração, e cada partida em andamento acrescenta
# um por rodada. Abaixo do tamanho do conjunto quente, um LRU não é um cache — é
# um recálculo com passo extra, e o que se recalcula aqui custa segundos. O que
# ele guarda são `Sugestao`, que são algumas centenas de bytes: 1.024 delas não
# chegam a um megabyte, e cobrem um dia inteiro de tráfego sobre a mesma secreta
# (que é o que o Termo é — todo mundo joga a mesma palavra, então os históricos
# de sessões diferentes convergem e o cache é compartilhado de fato).
@st.cache_resource(show_spinner=False, max_entries=1024)
def sugerir(chave: tuple, historico: tuple[tuple[str, str], ...]) -> Sugestao:
    """Jogada sugerida para este histórico. Cacheada pelo par (config, histórico).

    O nível 3 gasta décimos de segundo por jogada, e o Streamlit reexecuta o
    script a cada tecla no campo de texto — sem este cache a busca rodaria de novo
    a cada letra digitada.

    `escolher_com_teto` e não `escolher` porque aqui há alguém esperando. Sem o
    teto, a busca do nível 3 responde em décimos de segundo no caso comum e em
    MINUTOS quando a rodada anterior separou pouco — e o pior caso está a um
    clique: uma palavra fora do léxico marcada toda preta deixa ~5.800 candidatas,
    que é o mesmo tamanho da raiz da abertura. O teto troca 0,007 tentativa por um
    limite superior de resposta; a medição está em `escolher_com_teto`. Os dois
    níveis expõem o método, então esta camada continua sem saber com qual está
    falando.
    """
    motor = obter_motor(*chave)
    if not historico:
        # A mesma palavra que a CLI joga, nos dois resolvedores e em qualquer T
        # (ver `ABERTURA_PADRAO`). As duas telas do projeto não podem discordar da
        # primeira jogada, e ela não muda quando se mexe na barra lateral.
        return motor.abertura_padrao()
    candidatas, _ = reconstruir(chave, historico)
    # A fila (`VAGAS_DE_BUSCA`) fica DENTRO do corpo da função e depois de
    # `reconstruir`, e as duas coisas são de propósito. Dentro do corpo porque
    # este é um `cache_resource`: um acerto de cache não entra aqui, então quem
    # já tem resposta pronta não pega fila. Depois de `reconstruir` porque a
    # refiltragem é barata e cacheada — só a busca precisa de vaga.
    #
    # Segurar a vaga não pode esperar por mais nada: `motor` já veio resolvido
    # acima, e daqui para baixo é conta pura, sem outro cache no caminho. É o que
    # garante que a fila não entre num ciclo com os locks por chave do Streamlit.
    with VAGAS_DE_BUSCA:
        return motor.escolher_com_teto(
            candidatas, len(historico) + 1, teto=TETO_INTERATIVO
        )


# O docstring do módulo diz que refiltrar a cada rerun custa microssegundos, e
# isso vale enquanto as tentativas estiverem no léxico: aí cada rodada é uma linha
# da matriz. Fora dele não há linha, e `Motor.filtrar` cai no caminho lento — 6.046
# chamadas de `calcular_codigo` em Python puro, por rodada. Medido, uma partida com
# seis tentativas de fora custa 292 ms POR RERUN, e um rerun é cada clique numa
# casa. O Termo aceita palavras que não temos e a interface deixa jogá-las, então
# isso não é entrada malformada: é um usuário fazendo o que a tela permite.
#
# O cache é sobre (config, histórico), a mesma chave de `sugerir` — e pela mesma
# razão de lá, `cache_resource` e não `cache_data`. Quem chama trata o resultado
# como só de leitura; `filtrar` e `argsort` devolvem arrays novos, ninguém escreve
# no que veio daqui.
# A chave é a CONFIG, não o motor: um `Motor` não é hasheável para o Streamlit, e
# passá-lo com `_` na frente (a saída que ele oferece) o tiraria da chave — duas
# configurações diferentes passariam a compartilhar a mesma entrada de cache.
@st.cache_resource(show_spinner=False, max_entries=1024)
def reconstruir(
    chave: tuple, historico: tuple[tuple[str, str], ...]
) -> tuple[np.ndarray, list[int]]:
    """Candidatas restantes e quantas sobraram após cada rodada."""
    motor = obter_motor(*chave)
    candidatas = motor.todas_candidatas()
    trilha: list[int] = []
    for tentativa, padrao in historico:
        if padrao == PADRAO_VITORIA:
            # A rodada vencedora não filtra: a partida acabou, e se a secreta
            # estiver fora do léxico o filtro zeraria o conjunto sem motivo.
            break
        candidatas = motor.filtrar(candidatas, tentativa, padrao_para_codigo(padrao))
        trilha.append(len(candidatas))
    return candidatas, trilha


def rotular(motor: Motor | MotorNivel3, tentativa: str) -> str:
    """Forma acentuada, quando a palavra é uma que conhecemos (§7.2)."""
    indice = motor.lexico.indice_sonda.get(tentativa)
    return motor.lexico.mostrar(indice) if indice is not None else tentativa


# ---------------------------------------------------------------- desenho


def linha_html(palavra: str, padrao: str) -> str:
    casas = []
    for i in range(TAMANHO):
        letra = html.escape(palavra[i].upper()) if i < len(palavra) else "&nbsp;"
        classe = CLASSE[padrao[i]] if i < len(padrao) and padrao[i] in CLASSE else "dt-vazia"
        casas.append(f'<div class="dt-casa {classe}">{letra}</div>')
    return f'<div class="dt-linha">{"".join(casas)}</div>'


def grade_html(linhas: list[tuple[str, str]], vazias: int = 0) -> str:
    """Bloco de linhas: as passadas, mais `vazias` em branco."""
    corpo = [linha_html(palavra, padrao) for palavra, padrao in linhas]
    corpo += [linha_html("", "") for _ in range(vazias)]
    return f'<div class="dt-grade">{"".join(corpo)}</div>'


def estilo_casas(cores: str, fantasma: bool) -> str:
    """Pinta as cinco casas da linha em edição, uma regra por casa.

    O CSS é gerado a cada rerun em vez de existirem 3 classes fixas porque não há
    como pôr classe nossa num botão do Streamlit — o que dá para endereçar é a
    `key`, e ela precisa ficar ESTÁVEL entre reruns (uma key que muda a cada
    clique perderia o próprio clique que a mudou). Então a key é fixa e quem varia
    é a regra.
    """
    regras = []
    for i, cor in enumerate(cores):
        alvo = f".st-key-casa{i} button"
        if cor == VAZIA:
            # Casa intocada: contorno, como uma linha em branco do jogo. O
            # `fantasma` é o palpite sugerido ainda não digitado — visível o
            # bastante para se ler, apagado o bastante para não passar por jogada.
            regras.append(
                f"{alvo}{{background:transparent !important;"
                f"border-color:rgba(110,92,98,.6) !important;color:inherit !important;"
                f"opacity:{'.42' if fantasma else '.85'};}}"
            )
        else:
            fundo, tinta = FUNDO[cor]
            regras.append(
                f"{alvo}{{background:{fundo} !important;border-color:transparent "
                f"!important;color:{tinta} !important;opacity:1;}}"
            )
    return "<style>" + "".join(regras) + "</style>"


def sem_nivel(motivo: str) -> str:
    """`motivo` do motor sem a numeração interna dos algoritmos."""
    for de, para in SEM_NIVEL.items():
        motivo = motivo.replace(de, para)
    return motivo


def cartao_html(sugestao: Sugestao, rotulo: str) -> str:
    chips = []
    if sugestao.entropia > 0:
        chips.append(f"{sugestao.entropia:.2f} bits")
    if sugestao.valor_esperado is not None:
        chips.append(f"{sugestao.valor_esperado:.2f} tentativas esperadas")
    partes = [
        f'<div class="dt-rotulo">{html.escape(rotulo)}</div>',
        f'<div class="dt-palavra">{html.escape(sugestao.palavra)}</div>',
        '<div class="dt-metricas">'
        + "".join(f'<span class="dt-chip">{html.escape(c)}</span>' for c in chips)
        + (
            '<span class="dt-chip dt-chip-cand">pode ser a resposta</span>'
            if sugestao.e_candidata
            else '<span class="dt-chip">só sonda</span>'
        )
        + "</div>",
        f'<div class="dt-motivo">{html.escape(sem_nivel(sugestao.motivo))}</div>',
    ]
    if sugestao.alternativas:
        alternativas = ", ".join(
            f"<b>{html.escape(palavra)}</b>" if e_cand else html.escape(palavra)
            for palavra, _, e_cand in sugestao.alternativas[:N_ALTERNATIVAS]
        )
        partes.append(
            f'<div class="dt-alt">alternativas: {alternativas}'
            "<br><span style='opacity:.6'>em destaque, as que também podem ser a "
            "resposta</span></div>"
        )
    return f'<div class="dt-cartao">{"".join(partes)}</div>'


# ------------------------------------------------------------------- app


st.set_page_config(
    page_title="Destermo, o palpite ótimo do Termo",
    page_icon=str(MARCA / "icon-192.png"),
    layout="centered",
)
st.html(CSS)

estado = st.session_state
estado.setdefault("historico", [])  # list[tuple[tentativa normalizada, padrão]]
estado.setdefault("cores", LIMPA)  # a linha em edição, uma letra de "·BYG" por casa
estado.setdefault("geracao", 0)  # muda a key do campo de texto para limpá-lo
estado.setdefault("aviso", "")


def ciclar_cor(i: int) -> None:
    """Avança a cor da i-ésima casa: vazia → ausente → fora de lugar → certa."""
    cores = estado.cores
    estado.cores = cores[:i] + CICLO[cores[i]] + cores[i + 1 :]
    # Mexer numa casa é a resposta a um aviso de feedback impossível; deixá-lo na
    # tela depois disso faria o usuário reler um erro que ele já está corrigindo.
    estado.aviso = ""


def limpar_linha() -> None:
    """Zera a rodada em edição. A geração troca a key do campo, que é a única
    forma sancionada de esvaziar um widget de texto já instanciado."""
    estado.cores = LIMPA
    estado.geracao += 1
    estado.aviso = ""

st.html('<div class="dt-lockup" role="img" aria-label="Destermo"></div>')

with st.sidebar:
    st.html(
        '<div class="dt-lockup dt-lockup-mini" role="img" aria-label="Destermo"></div>'
    )
    st.subheader("Configurações")
    nivel = st.radio(
        "Tipo do resolvedor",
        (3, 2),
        format_func=lambda n: RESOLVEDOR[n].capitalize(),
        help="O primeiro simula as rodadas seguintes antes de decidir e termina a "
        "partida meia tentativa mais cedo, em média, custando décimos de segundo "
        "por jogada. O segundo escolhe a palavra que mais separa as candidatas "
        "entre si e responde na hora.",
    )
    bruto = st.select_slider(
        "Temperatura do prior (T)",
        options=["0.5", "1.0", "2.0", "5.0", "inf"],
        value="1.0",
        help="Palavras comuns caem no Termo com mais frequência que palavras raras, "
        "e o solver usa essa pista ao escolher entre candidatas parecidas. T diz o "
        "quanto ele confia nela: 0.5 aposta pesado nas comuns, 1.0 é o equilíbrio "
        "padrão, e inf desliga a pista e trata toda candidata como igualmente "
        "provável.",
    )
    temperatura = math.inf if bruto == "inf" else float(bruto)
    ampliado = st.toggle(
        "Espaço ampliado",
        help="Deixa sondar com conjugações: 8.629 palavras jogáveis, mas as 6.046 "
        "de sempre como resposta possível.",
    )
    st.divider()
    if st.button("Nova partida", use_container_width=True):
        estado.historico = []
        limpar_linha()
        st.rerun()

chave = (temperatura, nivel, ampliado, BEAM, PROFUNDIDADE)
with st.spinner("carregando léxico e matriz de padrões..."):
    motor = obter_motor(*chave)

espaco = (
    f"{len(motor.lexico)} respostas · {motor.lexico.n_sondas} jogáveis"
    if motor.lexico.ampliado
    else f"{len(motor.lexico)} palavras"
)
st.html(
    f'<p class="dt-sub">{RESOLVEDOR[nivel]} · T = {bruto} · '
    f"{html.escape(espaco)}</p>"
)
# Acima do tabuleiro, e não depois: quem precisa da ajuda precisa ANTES de mexer
# nas casas. Recolhida, cabe na mesma dobra que o cartão da sugestão.
st.html(COMO_USAR)

historico = tuple(estado.historico)
venceu = bool(historico) and historico[-1][1] == PADRAO_VITORIA
acabou = venceu or len(historico) >= N_MAX_TENTATIVAS

# Não há mais o aviso de "~9 min de busca" na primeira jogada: ele existia porque
# a abertura do nível 3 fora das configurações versionadas é uma busca na árvore a
# partir das 6.046 candidatas, inaceitável numa aba de navegador. Como a abertura
# agora é a do nível 2 (uma varredura de entropia, ~1 s), qualquer T responde na
# hora, e a busca só entra da segunda jogada em diante, onde custa décimos.

candidatas, trilha = reconstruir(chave, historico)

if venceu:
    st.success(f"acertou **{rotular(motor, historico[-1][0])}** em "
               f"{len(historico)} tentativa(s).")
elif len(historico) >= N_MAX_TENTATIVAS:
    st.warning("acabaram as 6 tentativas.")
else:
    with st.spinner("procurando a melhor jogada..."):
        sugestao = sugerir(chave, historico)
    st.html(cartao_html(sugestao, "melhor abertura" if not historico else "sugestão"))

jogadas = [(rotular(motor, t), p) for t, p in historico]

if acabou:
    st.html(grade_html(jogadas, vazias=N_MAX_TENTATIVAS - len(jogadas)))
    # O desfazer some junto com a entrada quando a partida acaba, e uma linha
    # marcada errado deixaria a sessão presa numa vitória que não houve.
    esquerda, direita = st.columns(2)
    if esquerda.button("desfazer a última rodada", use_container_width=True):
        estado.historico.pop()
        limpar_linha()
        st.rerun()
    if direita.button("nova partida", type="primary", use_container_width=True):
        estado.historico = []
        limpar_linha()
        st.rerun()
else:
    g = estado.geracao
    entrada = estado.get(f"entrada_{g}", "")
    digitada = normalizar(entrada.strip().lower())
    # Sem nada digitado, a linha em edição já vem com a sugestão — apagada, para
    # se ler como proposta e não como jogada. Registrar sem digitar nada aceita
    # essa proposta, que é o caminho de longe mais comum.
    tentativa = digitada or normalizar(sugestao.palavra)
    # Acentuada nas casas, como nas linhas já jogadas — sem isso a sugestão "sertã"
    # aparecia SERTA na linha em edição e SERTÃ uma linha acima, depois de jogada.
    letras = (rotular(motor, tentativa) + " " * TAMANHO)[:TAMANHO]
    cores = estado.cores

    # As três fatias do tabuleiro num contêiner só: o bloco vertical do Streamlit
    # separa elementos com 1rem, e um rombo desses no meio da grade a partiria em
    # duas. Dentro do contêiner o vão volta a ser o mesmo das casas.
    with st.container(key="tabuleiro"):
        st.html(grade_html(jogadas))
        st.html(estilo_casas(cores, fantasma=not digitada))
        with st.container(
            key="linha_ativa", horizontal=True, horizontal_alignment="center"
        ):
            for i in range(TAMANHO):
                # NBSP de verdade, não a entidade: o rótulo do botão passa por
                # markdown, e casa vazia precisa ocupar espaço para não colapsar.
                rotulo = letras[i].upper() if letras[i] != " " else " "
                # `on_click` e não um `if st.button(...): ...; st.rerun()`. O rerun
                # no meio do script abortaria a execução ANTES do campo de texto
                # abaixo, e o Streamlit descarta o estado de todo widget que não foi
                # instanciado no run — clicar numa casa apagaria a palavra digitada.
                # O callback roda antes do rerun, com o script inteiro intacto.
                st.button(
                    rotulo,
                    key=f"casa{i}",
                    on_click=ciclar_cor,
                    args=(i,),
                    help="clique para trocar a cor: ausente → fora de lugar → certa",
                )
        st.html(grade_html([], vazias=N_MAX_TENTATIVAS - len(jogadas) - 1))

    # "·" é ausência de clique, e o Termo pinta de preto o que não é nada.
    padrao = cores.replace(VAZIA, "B")

    if trilha:
        st.caption(f"{len(candidatas)} candidatas restantes")

    # DUAS AÇÕES SEPARADAS, e a separação é o conserto de um bug de verdade.
    #
    # O Streamlit não entrega o texto digitado enquanto se digita: medido, o
    # servidor só o recebe no Enter, no blur ou no submit de um form. Isso força
    # uma escolha, e as duas primeiras tentativas erraram:
    #
    #   campo solto + botão "registrar"  o clique chega antes do commit do texto,
    #                                    e quem digitava "termo" registrava a
    #                                    sugestão, em silêncio (§bug medido)
    #   campo e "registrar" no MESMO form  o valor chega junto com o clique, mas
    #                                    até lá o tabuleiro mostra a sugestão:
    #                                    marcam-se as cores sobre as letras de
    #                                    TOSAR enquanto a palavra é TERMO
    #
    # A regra que resolve as duas é: O TABULEIRO É A VERDADE. Submeter o form só
    # COMMITA a palavra — as casas passam a mostrá-la na hora —, e "Registrar
    # rodada" aplica exatamente o que está desenhado. Assim nada pode ser
    # registrado sem ter sido visto antes, e o Enter volta a ter o significado
    # natural de "é esta a palavra".
    with st.form(f"palavra_{g}", border=False):
        st.text_input(
            "Jogou outra palavra?",
            key=f"entrada_{g}",
            max_chars=TAMANHO + 3,
            placeholder=f"em branco = {sugestao.palavra}, a sugestão",
            help="Pode digitar sem acento — o Termo os preenche sozinho, e o solver "
            "normaliza dos dois jeitos. Enter também vale.",
        )
        st.form_submit_button("Usar esta palavra", use_container_width=True)

    registrar = st.button(
        "Registrar rodada", type="primary", use_container_width=True
    )

    # Terciário: é saída de emergência, não deve competir em peso com o botão
    # verde de largura inteira logo acima.
    _, canto = st.columns([3, 1])
    if canto.button(
        "Voltar", type="tertiary", use_container_width=True, disabled=not historico
    ):
        estado.historico.pop()
        limpar_linha()
        st.rerun()

    if registrar:
        estado.aviso = ""
        if len(tentativa) != TAMANHO:
            estado.aviso = f"a tentativa precisa ter {TAMANHO} letras."
        elif not padrao_possivel(tentativa, padrao):
            # §7.3: detectar ANTES de zerar o conjunto de candidatas.
            estado.aviso = (
                f"`{padrao}` é logicamente impossível para **{tentativa}**: nenhuma "
                "palavra secreta produziria esse feedback. Confira as cores."
            )
        elif padrao != PADRAO_VITORIA and not len(
            motor.filtrar(candidatas, tentativa, padrao_para_codigo(padrao))
        ):
            # §7.3 de novo: cenário esperado, não erro. A rodada NÃO é aplicada.
            estado.aviso = (
                "nenhuma candidata é compatível com esse feedback. Ou ele foi "
                f"marcado errado, ou a palavra do dia não está nas {len(motor.lexico)} "
                "do nosso léxico — ele é uma aproximação de fonte legítima "
                "(fserb/pt-br), não a lista oficial do Termo. A rodada não foi "
                "aplicada."
            )
        else:
            estado.historico.append((tentativa, padrao))
            limpar_linha()
            st.rerun()

    if estado.aviso:
        st.error(estado.aviso)

    if tentativa and len(tentativa) == TAMANHO and tentativa not in motor.lexico.indice_sonda:
        # §7.3: avisar, mas permitir — o Termo aceita palavras que não temos.
        st.info(f"**{tentativa}** não está no nosso léxico; seguimos mesmo assim.")

if trilha and not venceu:
    # Depois da vitória o conjunto exibido seria o da rodada ANTERIOR (a linha
    # vencedora não filtra) — anunciar "49 candidatas" ali seria mentira.
    with st.expander(f"candidatas restantes ({len(candidatas)})"):
        ordem = candidatas[np.argsort(-motor.lexico.prior[candidatas])]
        nomes = [motor.lexico.mostrar(int(i)) for i in ordem[:200]]
        st.markdown(
            "mais prováveis primeiro — "
            + ", ".join(f"`{n}`" for n in nomes)
            + (" ..." if len(ordem) > 200 else "")
        )
