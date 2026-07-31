# Resultados do benchmark

## Aberturas em três mundos concretos

Média penalizada em cada mundo (respostas cortadas, léxico inteiro digitável), e entre parênteses o quanto a abertura perde para a melhor DAS SEIS. Isso não é um minimax: a referência sai do próprio conjunto comparado, então quem está no topo dele zera por construção. O veredito de robustez é o da seção seguinte, que usa referência independente.

| abertura | N=300 | N=1500 | N=6046 | pior caso |
|---|---|---|---|---|
| tarso | 2.770 (+0.010) | 3.266 (+0.003) | 3.848 (+0.000) | 0.010 |
| tosar | 2.780 (+0.020) | 3.287 (+0.024) | 3.890 (+0.042) | 0.042 |
| sorte | 2.760 (+0.000) | 3.263 (+0.000) | 3.849 (+0.001) | 0.001 |
| serão | 2.787 (+0.027) | 3.331 (+0.068) | 3.906 (+0.057) | 0.068 |
| tirão | 2.760 (+0.000) | 3.292 (+0.029) | 3.860 (+0.012) | 0.029 |
| cairo | 2.783 (+0.023) | 3.282 (+0.019) | 3.862 (+0.013) | 0.023 |


## Robustez da abertura — minimax de arrependimento

Arrependimento em tentativas contra a melhor abertura de cada mundo, onde um mundo é uma distribuição das secretas com M palavras efetivas (M = 2^H). Última coluna: o pior caso, que é o que o critério minimiza.

| abertura | M=300<br>*lista bem curada* | M=447<br>*prior ICF, padrão do projeto* | M=600 | M=1200<br>*estilo Wordle* | M=2400 | M=4800 | M=6046<br>*não sei nada* | **pior caso** |
|---|---|---|---|---|---|---|---|---|
| **sertã** | +0.0283 | +0.0109 | +0.0000 | +0.0000 | +0.0084 | +0.0198 | +0.0281 | **+0.0283** |
| tarso | +0.0000 | +0.0000 | +0.0101 | +0.0027 | +0.0112 | +0.0290 | +0.0255 | +0.0290 |
| tória | +0.0364 | +0.0425 | +0.0323 | +0.0382 | +0.0528 | +0.0437 | +0.0122 | +0.0528 |
| corta | +0.0617 | +0.0507 | +0.0467 | +0.0192 | +0.0000 | +0.0098 | +0.0000 | +0.0617 |
| tirão | +0.0525 | +0.0553 | +0.0643 | +0.0582 | +0.0608 | +0.0481 | +0.0364 | +0.0643 |
| tosar | +0.0603 | +0.0412 | +0.0384 | +0.0391 | +0.0385 | +0.0490 | +0.0686 | +0.0686 |
| cairo | +0.0702 | +0.0602 | +0.0584 | +0.0546 | +0.0567 | +0.0493 | +0.0392 | +0.0702 |
| carto | +0.0985 | +0.0833 | +0.0671 | +0.0137 | +0.0042 | +0.0000 | +0.0012 | +0.0985 |

Abertura robusta: **sertã** (+0.0283 no pior mundo).

Sensibilidade à grade — removendo uma coluna de cada vez, a vencedora muda em M=4800 (passa a tarso, +0.0255). A margem no topo é da ordem da própria arbitrariedade da grade, então leia o pódio como empate, não como ordem.

A referência de cada coluna, e o que o nível 2 escolheria nela:

| mundo | T | melhor em tentativas | escolha do nível 2 (bits) | custo |
|---|---|---|---|---|
| M=300 — lista bem curada | 0.854 | tarso (3.3480) | tarso (3.3480) | +0.0000 |
| M=447 — prior ICF, padrão do projeto | 1 | tarso (3.3715) | tarso (3.3715) | +0.0000 |
| M=600  | 1.13 | sertã (3.3924) | tirão (3.4566) | +0.0643 |
| M=1200 — estilo Wordle | 1.56 | sertã (3.4743) | tirão (3.5325) | +0.0582 |
| M=2400  | 2.35 | corta (3.5664) | tirão (3.6273) | +0.0608 |
| M=4800  | 5.19 | carto (3.7075) | tória (3.7512) | +0.0437 |
| M=6046 — não sei nada | ∞ | corta (3.8250) | cairo (3.8642) | +0.0392 |

## Comparação de estratégias — bateria realista

| estratégia | T | média | média penal. | vitória | 1 | 2 | 3 | 4 | 5 | 6 | não resolveu | s/jogo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aleatória | 1 | 4.269 | 4.505 | 91.3% | 0 | 54 | 299 | 438 | 383 | 196 | 130 | 0.0002 |
| freq. de letras | 1 | 4.171 | 4.371 | 92.9% | 0 | 44 | 333 | 523 | 329 | 165 | 106 | 0.0011 |
| mais provável | 1 | 3.707 | 3.740 | 99.0% | 1 | 108 | 555 | 538 | 227 | 56 | 15 | 0.0001 |
| entropia (T=1) | 1 | 3.581 | 3.581 | 100.0% | 1 | 24 | 664 | 725 | 86 | 0 | 0 | 0.0039 |

Piores casos:

- **aleatória** — não resolveu: altar, anexo, apelo, arara, arcar, atuar, banal, barra, bazar, borda
- **freq. de letras** — não resolveu: agito, ainda, apito, arara, aveia, barra, barro, bazar, caçar, caixa
- **mais provável** — não resolveu: basto, besta, bingo, fardo, lente, lento, nadar, pasto, perna, persa

## Comparação de estratégias — bateria completo

| estratégia | T | média | média penal. | vitória | 1 | 2 | 3 | 4 | 5 | 6 | não resolveu | s/jogo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aleatória | 1 | 4.328 | 4.564 | 91.2% | 1 | 154 | 1078 | 1900 | 1561 | 818 | 534 | 0.0002 |
| freq. de letras | 1 | 4.144 | 4.363 | 92.3% | 1 | 162 | 1400 | 2104 | 1301 | 615 | 463 | 0.0013 |
| mais provável | 1 | 4.268 | 4.459 | 93.0% | 1 | 145 | 1201 | 2027 | 1496 | 752 | 424 | 0.0001 |
| entropia (T=1) | 1 | 3.946 | 3.957 | 99.7% | 1 | 56 | 1560 | 3150 | 1164 | 94 | 21 | 0.0014 |

Piores casos:

- **aleatória** — não resolveu: abafa, abudo, abuxó, açulo, adufo, agala, ajará, alita, altar, aluar
- **freq. de letras** — não resolveu: aerar, agave, agito, aiaçá, aiará, ainda, álalo, alila, alilo, alujá
- **mais provável** — não resolveu: ababá, ábiga, abobó, acajá, adano, ádito, aerar, afano, afilo, agrar
- **entropia (T=1)** — não resolveu: afilo, ajará, bafar, bimar, binar, boxar, doçal, fanga, fimbo, gaiar

## Nível 2 vs nível 3 — bateria realista

| estratégia | T | média | média penal. | vitória | 1 | 2 | 3 | 4 | 5 | 6 | não resolveu | s/jogo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| entropia (T=1) | 1 | 3.397 | 3.397 | 100.0% | 0 | 5 | 174 | 118 | 3 | 0 | 0 | 0.0103 |
| nível 3 (K=10, P=1) | 1 | 2.853 | 2.853 | 100.0% | 0 | 77 | 191 | 31 | 1 | 0 | 0 | 0.4809 |

## Varredura de temperatura — bateria realista

| estratégia | T | média | média penal. | vitória | 1 | 2 | 3 | 4 | 5 | 6 | não resolveu | s/jogo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| entropia (T=0.5) | 0.5 | 3.599 | 3.599 | 100.0% | 1 | 26 | 637 | 745 | 91 | 0 | 0 | 0.0048 |
| entropia (T=1) | 1 | 3.581 | 3.581 | 100.0% | 1 | 24 | 664 | 725 | 86 | 0 | 0 | 0.0045 |
| entropia (T=2) | 2 | 3.627 | 3.627 | 100.0% | 0 | 18 | 636 | 735 | 109 | 2 | 0 | 0.0046 |
| entropia (T=5) | 5 | 3.631 | 3.631 | 100.0% | 0 | 12 | 646 | 727 | 114 | 1 | 0 | 0.0058 |
| entropia (T=10) | 10 | 3.648 | 3.648 | 100.0% | 0 | 13 | 634 | 724 | 126 | 3 | 0 | 0.0049 |
| entropia (T=inf) | ∞ | 3.881 | 3.887 | 99.8% | 1 | 13 | 434 | 789 | 235 | 25 | 3 | 0.0047 |

Piores casos:

- **entropia (T=2)** — não resolveu: —
- **entropia (T=5)** — não resolveu: —
- **entropia (T=10)** — não resolveu: —
- **entropia (T=inf)** — não resolveu: sanar, vinda, xangô
