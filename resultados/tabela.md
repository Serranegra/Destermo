# Resultados do benchmark

## Arrependimento por mundo — qual abertura é a menos pior

Média penalizada em cada mundo (respostas cortadas, léxico inteiro digitável), e entre parênteses o quanto a abertura perde para a melhor daquele mundo.

| abertura | N=300 | N=1500 | N=6046 | pior caso |
|---|---|---|---|---|
| tarso | 2.770 (+0.010) | 3.266 (+0.003) | 3.848 (+0.000) | 0.010 |
| tosar | 2.780 (+0.020) | 3.287 (+0.024) | 3.890 (+0.042) | 0.042 |
| sorte | 2.760 (+0.000) | 3.263 (+0.000) | 3.849 (+0.001) | 0.001 **←** |
| serão | 2.787 (+0.027) | 3.331 (+0.068) | 3.906 (+0.057) | 0.068 |
| tirão | 2.760 (+0.000) | 3.292 (+0.029) | 3.860 (+0.012) | 0.029 |
| cairo | 2.783 (+0.023) | 3.282 (+0.019) | 3.862 (+0.013) | 0.023 |

Menor arrependimento de pior caso: **sorte**.

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
