<p align="center">
  <img src="brand/destermo-icon.svg" width="140" alt="destermo logo">
</p>

<h1 align="center">Destermo</h1>
<p align="center"><i>O palpite ótimo do Termo.</i></p>

Solver de [Termo](https://term.ooo) baseado em entropia de Shannon, com prior de
frequência de uso. Implementa a [especificação v1.1](docs/ESPECIFICACAO_v1.1.md):
uma ferramenta (CLI que sugere a próxima palavra) e uma análise (benchmark
contra heurísticas simples). A [v1.0](docs/ESPECIFICACAO_v1.0.md) fica no
histórico porque o README compara os resultados das duas.

O solver **nunca conhece a palavra secreta**. Ele mantém o conjunto de candidatas
compatíveis com todo o feedback recebido e escolhe a tentativa que maximiza a
informação esperada.

## Requisitos

Python 3.10+ com `numpy`. Para os gráficos, `matplotlib`. Para os testes, `pytest`.

```bash
pip install -r requirements.txt
```

Nada em `data/` é versionado: as fontes são baixadas do
[`fserb/pt-br`](https://github.com/fserb/pt-br) na primeira execução e a matriz
de padrões se reconstrói em ~4 s. Basta clonar e rodar.

## Uso

```bash
python solver.py
```

```
  melhor abertura: tosar   (5.91 bits, E=3.01 tentativas, é candidata)

[1] tentativa > tarso
[?] feedback de 'tarso' > GBGBG

  candidatas restantes: 18
  sugestão: voncê   (2.50 bits, E=2.01 tentativas, não é candidata)
  motivo: menor nº esperado de tentativas (E=2.006) entre os 11 melhores palpites
          por entropia (18 candidatas, 5 rodadas, profundidade 1)
  alternativas: conde, cnute, coiné
```

Digite as tentativas **sem acento** — é o que o jogador faz no term.ooo, que
preenche os acentos sozinho. Entrada acentuada também é aceita e normalizada. As
sugestões saem na forma acentuada, que é a que aparece na tela do jogo.

O feedback aceita `G`/`Y`/`B` (maiúsculo ou minúsculo), `2`/`1`/`0` ou os emojis
🟩🟨⬛. Comandos: `voltar` desfaz a última rodada, `listar` mostra as candidatas,
`?` mostra a ajuda, `sair` encerra.

O padrão é o **nível 3** ([`termo/nivel3.py`](termo/nivel3.py)): ele minimiza o
número **esperado de tentativas** em vez de maximizar bits por jogada, o que vale
0,54 tentativa na bateria realista. A abertura dele vem em cache no repositório e
cada jogada custa décimos de segundo.

Para o nível 2 puro — entropia, milissegundos por jogada:

```bash
python solver.py --nivel 2
```

Outras temperaturas do prior:

```bash
python solver.py --t 2
```

```bash
python solver.py --t inf
```

`--t inf` desliga o prior de frequência e recupera a entropia pura.

**Atenção ao combinar `--t` com o nível 3.** Só a configuração padrão
(`T=1, beam=10, profundidade=1`) vem com a abertura pronta em
`data/aberturas_nivel3.json`; qualquer outra é uma busca a partir das 6.046
candidatas e leva ~9 min, uma vez, antes da primeira sugestão. A CLI avisa e
sugere `--nivel 2` para começar na hora. O tamanho da busca é ajustável por
`--beam` (palpites testados por nó) e `--profundidade` (níveis de busca antes de
cair na política gulosa) — cada combinação tem a sua própria abertura em cache.

Na primeira execução o programa baixa o léxico do repositório
[`fserb/pt-br`](https://github.com/fserb/pt-br) e constrói a matriz de padrões
(~4 s, 37 MB em `data/`). Depois disso o arranque é instantâneo.

### Benchmark e gráficos

```bash
python benchmark.py --bateria realista
```

```bash
python benchmark.py --bateria completo
```

```bash
python benchmark.py --varredura-t
```

```bash
python benchmark.py --nivel3
```

```bash
python analise.py
```

### Testes

```bash
python -m pytest tests -q
```

## Estrutura

O motor é independente da interface: trocar a CLI por um bot, uma API ou uma
página web não exige mexer em nada dentro de `termo/`.

| Arquivo | Papel |
|---|---|
| `termo/feedback.py` | Normalização de acentos, regra das duas passadas, codificação base-3, detector de padrão impossível |
| `termo/lexico.py` | Download, filtro de 5 caracteres, agrupamento por forma normalizada, dedupe por ICF, prior |
| `termo/matriz.py` | Matriz 6.046 × 6.046 de padrões pré-computados (numpy vetorizado) |
| `termo/entropia.py` | Cálculo de entropia, regra de endgame, cache da abertura (nível 2) |
| `termo/nivel3.py` | Busca na árvore de decisão pelo menor nº esperado de tentativas (nível 3) |
| `termo/estrategias.py` | As estratégias do benchmark |
| `solver.py` | CLI interativa (só I/O) |
| `benchmark.py` | Simulação de partidas e métricas |
| `analise.py` | Gráficos e tabela de resultados |

Cada palavra tem duas formas (§2.4): a **normalizada** (`terco`), usada em todo
cálculo, e a de **exibição** (`terço`), puramente cosmética. O léxico final fica
em `termo_lexico_5letras.txt` na forma acentuada; `data/` guarda as fontes
baixadas e os artefatos gerados.

## Resultados

### Léxico

Todos os números da §2.2 e §2.3 foram reproduzidos exatamente a partir da fonte:

| Passo | Resultado |
|---|---|
| 5 caracteres em `lexico` | 6.301 |
| Grupos por forma normalizada | 6.046 |
| Grupos com colisão | 242 (497 palavras, 7,9%) |
| **Léxico final** | **6.046** |
| Cobertura do ICF | 100% |
| Palavras com "ç" | 172 |

A dedupe escolhe variantes plausíveis: `terço` (não `terco`, `terçó` ou `terçô`),
`agora`, `pique`, `manto`. `festa` — a palavra de 5 jan 2022 (§2.7) — está no
léxico. O sinal de validação da §2.3 se confirma: as 6.046 ficam a 20 palavras das
6.026 que o Gabriel Yshay obteve por um caminho independente.

### Abertura

Melhor primeira jogada com T=1: **`tarso`**, 5,97 bits. Seguem `tirão` (5,97),
`tória` (5,95), `sertã` (5,94), `teira` (5,92).

### Comparação de estratégias

Bateria realista — as 1.500 palavras de menor ICF, que é o que o jogo de fato
sorteia. `média` conta só as partidas vencidas; `penal.` conta derrota como 7.

| estratégia | média | penal. | vitória | 1 | 2 | 3 | 4 | 5 | 6 | s/jogo |
|---|---|---|---|---|---|---|---|---|---|---|
| aleatória | 4.269 | 4.505 | 91.3% | 0 | 54 | 299 | 438 | 383 | 196 | 0.0002 |
| freq. de letras | 4.171 | 4.371 | 92.9% | 0 | 44 | 333 | 523 | 329 | 165 | 0.0009 |
| mais provável | 3.707 | 3.740 | 99.0% | 1 | 108 | 555 | 538 | 227 | 56 | 0.0001 |
| **entropia (T=1)** | **3.581** | **3.581** | **100.0%** | 1 | 24 | 664 | 725 | 86 | 0 | 0.0032 |

Stress test — léxico completo (6.046 palavras, incluindo obscuridades que o Termo
nunca sortearia):

| estratégia | média | penal. | vitória | 1 | 2 | 3 | 4 | 5 | 6 | s/jogo |
|---|---|---|---|---|---|---|---|---|---|---|
| aleatória | 4.328 | 4.564 | 91.2% | 1 | 154 | 1078 | 1900 | 1561 | 818 | 0.0002 |
| freq. de letras | 4.144 | 4.363 | 92.3% | 1 | 162 | 1400 | 2104 | 1301 | 615 | 0.0009 |
| mais provável | 4.268 | 4.459 | 93.0% | 1 | 145 | 1201 | 2027 | 1496 | 752 | 0.0001 |
| **entropia (T=1)** | **3.946** | **3.957** | **99.7%** | 1 | 56 | 1560 | 3150 | 1164 | 94 | 0.0009 |

![Distribuição de tentativas](resultados/distribuicao_realista.png)

**Os três palpites da §5.6 se confirmaram**, incluindo o mais específico:

- Entropia ficou em 3,58 na bateria realista — dentro da faixa 3,5–4 prevista.
- "Mais provável" chegou perto na média: 3,707 contra 3,581.
- **A diferença real aparece na taxa de vitória e no pior caso.** Entropia
  resolveu 100% das 1.500 palavras e **nenhuma** partida chegou à 6ª tentativa.
  "Mais provável" perdeu 15 partidas e chegou à 6ª em 56.

O stress test mostra por que a régua importa: "mais provável" depende do prior
estar certo. Quando as secretas passam a incluir o léxico inteiro, ela cai de
3,71 para 4,27 e fica atrás da heurística de frequência de letras na média. A
entropia degrada de 3,58 para 3,95 e mantém 99,7%.

Custo: a entropia é ~30× mais lenta por jogo que as heurísticas baratas — mas
"lenta" aqui são 3,2 ms. Para uso interativo é irrelevante. (Eram 11 ms até o
caminho rápido de entropia descrito no Nível 3, que a acelerou 3,4×.)

### Varredura de temperatura

A pergunta da §5.5: quanto o prior de frequência ajuda? Como T→∞ recupera a
entropia pura, esta curva compara o Nível 1 e o Nível 2 do algoritmo de forma
contínua.

| T | 0.5 | 1 | 2 | 5 | 10 | ∞ |
|---|---|---|---|---|---|---|
| média (penal.) | 3.599 | **3.581** | 3.627 | 3.631 | 3.648 | 3.887 |
| vitória | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% |

![Curva de temperatura](resultados/curva_temperatura.png)

**O prior de frequência vale cerca de 0,31 tentativa** — de 3,89 (entropia pura)
para 3,58 (T=1). É a resposta empírica para a camada que faltava no artigo
brasileiro citado na especificação.

O mínimo cai exatamente em **T = 1**, o valor padrão da v1. A curva é rasa entre
0,5 e 10 (amplitude de 0,07 tentativa, dentro do ruído de 1.500 partidas) e só
piora de verdade em T→∞. Ou seja: **ter um prior importa; calibrá-lo com precisão,
nem tanto.**

### Nível 3: o proxy contra o objetivo real

A especificação pôs o Nível 3 fora de escopo por custo computacional (§4.1). Ele
está implementado em [`termo/nivel3.py`](termo/nivel3.py) e o custo agora tem
número. A conta é a do 3B1B, com o limite de rodadas dentro dela:

```
V(S, r) = min_g [ 1 + Σ_{padrão ≠ GGGGG} P(padrão | S) · V(S_padrão, r-1) ]
V(S, 0) = 7        acabaram as rodadas sem acertar
```

Três coisas tornam isso viável em 6.046 palavras: **beam** (só os K palpites de
maior entropia entram na busca — o Nível 2 é o *move ordering*), **profundidade**
(abaixo de P níveis a recursão cai na política gulosa do Nível 2, o que dá um
limite superior de V e torna a busca *anytime*) e **memoização** com
branch-and-bound.

#### A abertura muda

| | abertura | entropia | posição no ranking de entropia |
|---|---|---|---|
| Nível 2 | `tarso` | 5,97 bits | 1ª |
| Nível 3 | `tosar` | 5,91 bits | **6ª** |

As duas usam exatamente as mesmas cinco letras. `tosar` é a sexta do léxico em
entropia — o Nível 2 a lista por último entre as alternativas — e mesmo assim é a
que minimiza o número esperado de tentativas (E = 3,01). É a correção do 3B1B em
uma linha: **maximizar bits não é minimizar tentativas.**

#### No meio do jogo a diferença é grande

Depois de `tarso` com o `r` verde sobram 51 candidatas:

| | jogada | entropia | E[tentativas] |
|---|---|---|---|
| Nível 2 (entropia) | `miúde` | 2,01 bits | 2,05 |
| Nível 3 | `verde` | 1,83 bits | **1,43** |

O Nível 3 abre mão de 0,18 bit para jogar uma **candidata**, que pode encerrar o
jogo ali. Com T=1 isso não é aposta: `verde` sozinha carrega 65% da massa de
probabilidade das 51. Subindo T o prior achata e a escolha converge para a do
Nível 2 — em T=5 e T→∞ os dois jogam `miúde`. O dial da §4.2 governa os dois
níveis.

#### Head-to-head

As 300 palavras de menor ICF, mesmas secretas para os dois (`--nivel3` usa uma
amostra menor que a tabela principal, então a linha da entropia aqui não é
comparável com os 3,581 de 1.500 palavras):

| estratégia | média | penal. | vitória | 1 | 2 | 3 | 4 | 5 | 6 | s/jogo |
|---|---|---|---|---|---|---|---|---|---|---|
| entropia (T=1) | 3.397 | 3.397 | 100.0% | 0 | 5 | 174 | 118 | 3 | 0 | 0.0088 |
| **nível 3 (K=10, P=1)** | **2.853** | **2.853** | **100.0%** | 0 | 77 | 191 | 31 | 1 | 0 | 0.3343 |

![Nível 2 vs nível 3](resultados/distribuicao_nivel3_realista.png)

**O objetivo real vale 0,54 tentativa** — 3,397 para 2,853, com as duas
estratégias resolvendo 100%. O ganho não vem de evitar derrotas: vem de **acabar o
jogo na 2ª tentativa**, que a entropia consegue em 5 partidas e o Nível 3 em 77.
Maximizar bits é exatamente a política que se recusa a tentar ganhar cedo.

Custo: **38× mais CPU** (0,33 s por jogo contra 8,8 ms), mais 8,8 min de abertura.
Para uso interativo é tranquilo; para varrer 1.500 palavras em seis temperaturas,
não. A decisão da especificação continua defensável — o que mudou é que agora ela
é uma escolha informada, não uma suposição.

#### O gargalo não era a busca

Perfilando a busca num estado de 223 candidatas, **95% do tempo estava em escolher
o beam** — a varredura de entropia do léxico — e só ~2% na recursão que minimiza
tentativas. Pior: 87% das varreduras eram em conjuntos de **até 8 candidatas**, ou
seja, percorrer 6.046 palavras para ranquear palpites contra quatro.

O caminho geral de `entropias` monta uma tabela de 6.046 × 243 baldes; com poucas
candidatas isso é 11 MB alocados e zerados para preencher meia dúzia de colunas
por linha. Reescrevendo a mesma soma por agrupamento par a par
(`Motor._entropias_poucas`, `H = -Σ_c w_c·log₂(massa do grupo de c)`, custo n·m² em
vez de n·243), mais um cache do beam por conjunto:

| | antes | depois | |
|---|---|---|---|
| jogada com 223 candidatas | 6,96 s | 2,03 s | 3,4× |
| abertura do nível 3 | 953 s | 529 s | 1,8× |
| **nível 2, bateria realista** | 10,7 ms | 3,2 ms | **3,4×** |

A conta é a mesma, não uma aproximação — e isso foi verificado, não suposto: a
abertura recalculada dá `tosar` com `3.0094161966572304`, dígito por dígito, e
reprocessar as duas baterias principais (7.546 partidas × 4 estratégias) muda
**apenas** os campos de tempo dos JSONs. O nível 2 levou o ganho junto, de graça.

#### Uma ressalva honesta

O Nível 3 otimiza o valor esperado **sob o prior**, e o prior de T=1 é agressivo.
A consequência é que ele **sacrifica palavras raríssimas de propósito**: prefere
chutar a candidata provável a gastar uma jogada separando o grupo. Em `criva`
(ICF 19) ele entra na espiral `prima → urina → brida → crica` e perde uma partida
que o Nível 2 ganha. Na bateria realista — o que o Termo de fato sorteia — isso
nunca acontece, e é justamente onde ele ganha 0,54 tentativa.

O `E = 3,01` da abertura é uma esperança ponderada pelo prior sobre o léxico
inteiro; a média do benchmark é uma contagem simples sobre as mais comuns. Os dois
números medem coisas diferentes e não se comparam.

### Efeito da migração v1.0 → v1.1

O léxico corrigido melhorou tudo, como a §5.6 antecipava — menos candidatas para
separar:

| | v1.0 (8.996 palavras) | v1.1 (6.046 palavras) |
|---|---|---|
| Entropia, bateria realista | 3.685 | **3.581** |
| Entropia, stress test | 4.110 | **3.946** |
| Partidas na 6ª tentativa (realista) | 1 | **0** |
| Ganho do prior (T=1 vs T→∞) | 0.33 | 0.31 |
| Matriz | 81 MB, 20 s | **37 MB, 4 s** |
| Entropia da 1ª jogada | 1.4 s | **1.0 s** |
| T ótimo na varredura | 2 | **1** |

A abertura continua sendo `tarso` nas duas versões.

## Achados e desvios da especificação

### O caso de teste 3 da §3.3 continua com o valor errado

A v1.1 corrigiu os casos 5–7 (acentos), mas o caso 3 passou batido — ele já
estava errado na v1.0 e não tem nada a ver com normalização.

| # | Secreta | Tentativa | Especificação | Correto |
|---|---|---|---|---|
| 3 | `carro` | `rerum` | `BGBYB` | `YBGBB` |
| 8 | `termo` | `metro` | "verificar" | `YGYYG` |

No caso 3, o único verde é o `r` da posição 2 (`rerum[2] == carro[2]`), mas a
especificação colocou o `G` na posição 1. Ambos estão corrigidos em
[test_feedback.py](tests/test_feedback.py), com a derivação manual no docstring.
O que cada caso testa continua o mesmo.

A implementação foi validada de três formas independentes: os 8 casos
obrigatórios, propriedades estruturais (simetria da contagem de letras coloridas,
consistência de G e Y, idempotência da normalização), e a matriz vetorizada
conferida célula a célula contra a implementação de referência em Python puro.

### A cedilha (§3.5 e §11) é um interruptor de uma linha

`NORMALIZAR_CEDILHA` em [feedback.py](termo/feedback.py) controla se `ç` vira `c`.
O padrão é `True`, seguindo a §3.2. Se o teste empírico no term.ooo mostrar que a
cedilha é letra distinta, basta trocar para `False`:

- com `True`: `terço` → `terco`; `calcular_feedback("acude", "açude") == "GGGGG"`
- com `False`: `terço` → `terço`; `calcular_feedback("acude", "açude") == "GBGGG"`

A troca **regenera tudo automaticamente**: o léxico é reagrupado (as 172 palavras
com "ç" deixam de colidir com as variantes em "c"), e a matriz em cache se
invalida sozinha, porque a assinatura é um hash da lista de palavras.

### "Excluir conjugações" é um critério de fonte, não de morfologia

A v1.1 manda não usar o arquivo `conjugações`. Isso remove `fugia`, `curas`,
`zarpa`, `andei` — mas **não** remove `abafa`, que consta do dicionário geral por
si próprio. O critério implementado é o da especificação (a fonte), e o teste
`test_conjugacoes_foram_excluidas` documenta essa distinção para não induzir a
erro depois.

### A regra de endgame da penúltima tentativa é conservadora

A especificação (§4.5) manda chutar uma candidata na penúltima tentativa quando
`C` é pequeno. Isso nem sempre é ótimo: com 2 tentativas restantes e 3 candidatas,
chutar uma candidata dá ~67% de vitória, enquanto uma palavra separadora que
distingue as três com certeza dá 100%.

A regra foi implementada como especificado, mas o limiar é configurável
(`Motor(limiar_endgame=...)`, padrão 3). Com 2 ele vira inócuo, já que `len(C) <= 2`
tem tratamento próprio. Fazer isso direito exige minimizar tentativas esperadas —
o Nível 3, que a v1 pôs fora de escopo e que existe agora em
[`termo/nivel3.py`](termo/nivel3.py) (seção abaixo).

### Detector de feedback logicamente impossível

A especificação (§7.3) pede detectar feedback impossível *antes* de zerar `C`, sem
dizer como. `padrao_possivel` faz uma checagem puramente lógica, independente do
léxico, com dois critérios:

1. **Ordem dentro de uma letra repetida.** A passada 2 pinta de amarelo as
   ocorrências mais à esquerda, então um `B` nunca pode preceder um `Y` da mesma
   letra. `BYBBB` para `oovxx` é impossível.
2. **Falta de casa para os amarelos.** Cada amarelo exige uma cópia da letra numa
   posição não-verde que não seja a dela própria. `GGGGY` é impossível: sobra uma
   única posição livre, e é justamente a da letra que precisaria aparecer.

Isso separa "você digitou o feedback errado" de "a palavra do dia não está na
nossa lista" (a ressalva §2.6), que são situações diferentes para o usuário.

### Salvaguarda contra o arquivo de léxico da v1.0

A armadilha da §0.3 é um cache derivado sobreviver à migração. A matriz e a
abertura já se protegem por assinatura, mas o `termo_lexico_5letras.txt` não
tinha proteção nenhuma. `carregar_exibicao` agora rejeita qualquer lista com
formas normalizadas repetidas — que é exatamente o que um arquivo da v1.0
pareceria — com uma mensagem dizendo o que fazer, em vez de produzir resultados
silenciosamente errados.

## Fora de escopo (v1)

Dueto e Quarteto; curadoria manual adicional do léxico; interface web/bot/API;
separação formal entre lista de respostas e lista de tentativas válidas.

O Nível 3 também estava nesta lista (§12) e saiu dela: está implementado em
[`termo/nivel3.py`](termo/nivel3.py) e **é o padrão da CLI**. O motivo da
especificação (custo computacional) não desapareceu — ele ficou mensurável, e
medido dá décimos de segundo por jogada com a abertura em cache, o que é barato
demais para justificar abrir mão de 0,54 tentativa. No benchmark o padrão
continua sendo o nível 2: lá são 1.500 partidas por estratégia, e aí os 38× de
CPU pesam.

## Marca

Os ativos de marca — símbolo, lockup, favicons, ícones de app e banner Open
Graph — ficam em [`brand/`](brand/). A tabela de qual arquivo usar quando, a
paleta e o tamanho mínimo do símbolo estão em
[brand/README.md](brand/README.md).

A paleta é a mesma em qualquer saída colorida (CLI, gráficos, web):

| | hex | papel |
|---|---|---|
| verde | `#3AA394` | marca, acerto |
| dourado | `#D3AD69` | letra fora de lugar |
| xisto | `#6E5C62` | letra ausente, texto terciário |
| ardósia | `#221C1E` | fundo escuro |
| osso | `#F5EFE9` | texto sobre escuro |

Os gráficos de [analise.py](analise.py) derivam suas séries dessas cinco cores.

Não edite os arquivos de `brand/` à mão: os PNGs e o `.ico` são gerados a partir
dos SVGs por `brand/build.sh` (requer `cairosvg` e `imagemagick`).

Quando existir um front-end, copie para a raiz do diretório público
`favicon.ico`, `favicon.svg`, `apple-touch-icon.png`, `icon-192.png`,
`icon-512.png`, `manifest.webmanifest` e `og-banner.png`, e cole o conteúdo de
[brand/head-snippet.html](brand/head-snippet.html) no `<head>`.

Estes ativos **não** estão sob a licença MIT do repositório.

## Fonte dos dados

O léxico de palavras e as pontuações ICF foram derivados do repositório
[fserb/pt-br](https://github.com/fserb/pt-br), licenciado sob
[MIT](https://github.com/fserb/pt-br/blob/master/LICENSE)
(© 2021 Fernando Serboncini).