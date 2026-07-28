# Destermo — Especificação Técnica (v1.1)

> Solver de Termo (Wordle em português) baseado em entropia de Shannon / teoria da
> informação. Combina uma ferramenta prática (sugestão da melhor próxima palavra a
> cada rodada) com um componente de análise (benchmark comparando a estratégia de
> entropia contra heurísticas mais simples).

---

## ⚠️ 0. CHANGELOG v1.0 → v1.1 — LEIA PRIMEIRO

Esta versão corrige **três erros da v1.0**, descobertos ao verificar a documentação
oficial do term.ooo e da Wikipédia. Se você já implementou a v1.0, esta seção é o
guia de migração.

### 0.1 As três correções

| # | Item | v1.0 (ERRADO) | v1.1 (CORRETO) | Fonte |
|---|---|---|---|---|
| 1 | **Acentos** | Letras distintas ("á" ≠ "a") | **Normalizar** — acentos não afetam as cores | Ajuda do term.ooo: "Os acentos são preenchidos automaticamente, e não são considerados nas dicas." |
| 2 | **Conjugações verbais** | Incluir (8.996 palavras) | **Excluir** (6.301 palavras) | Wikipédia: a lista foi refinada para remover "regionalismos obscuros, palavrões, nomes próprios e **conjugações verbais**" |
| 3 | **Colisões de normalização** | Não previsto | **Deduplicar** → 6.046 palavras | Consequência da correção 1 |

### 0.2 Raio de alcance na base de código

| Arquivo | Ação |
|---|---|
| `feedback.py` | ⚠️ Corrigir — adicionar normalização na entrada (§3.2) |
| Testes de feedback | ⚠️ Corrigir — casos que assumiam a regra antiga (§3.3) |
| `lexico.py` | ⚠️ Corrigir — remover conjugações, normalizar, deduplicar (§2.2) |
| Matriz `.npy` em cache | 🔄 **DELETAR e regerar** — dado derivado, incompatível |
| Primeira palavra em cache | 🔄 **DELETAR e recalcular** |
| `entropia.py` | ✅ Intacto — não depende de acentos |
| `estrategias.py` | ✅ Intacto |
| `solver.py` (CLI) | ✅ Quase intacto — entrada fica mais simples (§7.2) |
| `benchmark.py` | ✅ Código intacto — **mas todos os resultados anteriores são inválidos** |

**O núcleo do algoritmo não muda.** Ele nunca soube o que é um acento.

### 0.3 Armadilha principal da migração

**Deletar o cache da matriz pré-computada.** Ela foi construída com 8.996 índices
e com a regra de feedback antiga. Se sobreviver à migração, o resultado é um
crash (na melhor hipótese) ou **resultados silenciosamente errados** (na pior).

---

## 1. Visão geral

### 1.1 O que é

Uma ferramenta que **assiste** o jogador do Termo. **Não é um clone do jogo.**

```
1. Usuário joga no term.ooo e digita uma palavra
2. O site devolve o feedback colorido (🟩🟨⬛)
3. Usuário informa ao Destermo: qual palavra tentou + qual feedback recebeu
4. Destermo calcula e sugere a próxima melhor palavra
5. Repete até acertar
```

O solver nunca conhece a palavra secreta. Ele mantém o conjunto de candidatas
compatíveis com todo o feedback recebido e escolhe a próxima tentativa que
maximiza a informação esperada.

### 1.2 Duas frentes

| Frente | Descrição |
|---|---|
| **Ferramenta** | CLI interativa para uso real ao jogar |
| **Análise** | Benchmark comparando estratégias, medindo tentativas médias |

Ambas compartilham o mesmo motor.

### 1.3 Escopo da v1

- ✅ Termo clássico (palavra única de 5 letras, 6 tentativas)
- ❌ Dueto / Quarteto — fora de escopo

### 1.4 Stack

- **Linguagem**: Python
- **Numérico**: numpy (essencial — §6)
- **Interface v1**: CLI

---

## 2. Léxico

### 2.1 Fonte

Repositório oficial do criador do Termo (Fernando Serboncini), licença MIT:

```
https://github.com/fserb/pt-br
```

Baixar de `raw.githubusercontent.com/fserb/pt-br/master/`:

| Arquivo | Conteúdo | Uso na v1.1 |
|---|---|---|
| `lexico` | 145.744 entradas | ✅ Palavras de 5 letras |
| `conjugações` | 195.751 formas verbais | ❌ **NÃO usar** (ver §0.1) |
| `icf` | 419.486 palavras com frequência | ✅ Prior de probabilidade |

### 2.2 Construção da lista (CORRIGIDO)

```
PASSO 1 — Filtrar por comprimento
  palavras de exatamente 5 CARACTERES em `lexico`          → 6.301

PASSO 2 — Agrupar por forma normalizada
  normalizar(w) = remover diacríticos (NFD + descartar Mn)
  agrupar as 6.301 palavras por normalizar(w)              → 6.046 grupos
                                                             (242 com colisão)

PASSO 3 — Deduplicar
  de cada grupo, manter a variante de MENOR ICF (mais comum)

LÉXICO FINAL                                               → 6.046 palavras
```

**Atenção**: contar **caracteres**, não bytes. `len()` em `str` Python está
correto; ferramentas de shell (`awk length`) não.

### 2.3 Por que deduplicar (correção 3)

Após normalizar acentos, 242 grupos de palavras tornam-se **indistinguíveis** no
feedback:

```
terco → terco, terço, terçó, terçô
agora → agora, agorá, ágora
pique → pique, piqué, piquê
manto → manto, mantó, mantô
```

São **497 palavras (7,9%)** que o solver nunca conseguiria separar. Se o conjunto
de candidatas convergir para um desses grupos, **não existe jogada que os
distinga** — o solver trava.

Isso é artefato da nossa fonte ser um dicionário bruto. A lista real do jogo só
pode conter **uma** variante de cada forma, já que o Termo preenche acentos
automaticamente (o jogador digita "terco" e o jogo completa).

**Sinal de validação**: Gabriel Yshay chegou a **6.026 palavras** por um caminho
independente (dicionário do IME-USP). Nossas **6.046** ficam a 20 palavras de
distância — forte indício de que esse é o tamanho correto do problema.

### 2.4 Estrutura de dados

Manter **duas formas de cada palavra**:

| Forma | Uso |
|---|---|
| **Normalizada** (`terco`) | Todo cálculo: feedback, matriz, entropia, filtragem |
| **Original** (`terço`) | Exibição ao usuário |

Todo o motor opera exclusivamente sobre a forma normalizada. A forma acentuada é
puramente cosmética.

### 2.5 Decisões (ATUALIZADO)

| Decisão | Escolha v1.1 | Justificativa |
|---|---|---|
| Conjugações verbais | **Excluir** | A curadoria oficial as removeu |
| Curadoria de obscuras | Não aplicar | Ajustar depois se aparecer lixo |
| Respostas vs tentativas | Lista única | O prior (§4) já diferencia probabilidade |
| **Acentos** | **Normalizar** | Não afetam as cores no jogo real |
| **Colisões** | **Deduplicar por menor ICF** | Evita candidatas indistinguíveis |

### 2.6 Ressalva conhecida

A lista oficial do Termo tem ~18.000 palavras e resultou de curadoria manual
**não publicada**. O `fserb/pt-br` é o léxico geral que o autor liberou depois,
não o dump do jogo.

**Consequência**: nossa lista é uma **aproximação de fonte legítima**. É possível
que a palavra do dia não esteja nela. Isso deve ser tratado sem crash (§7.3).

### 2.7 Trivia útil

- Primeira palavra do jogo (5 jan 2022): **"festa"** — serve como caso de teste real
- A palavra "seios" foi excluída da lista a pedido da mãe do autor

---

## 3. Regras de feedback

### 3.1 Representação

String de **5 caracteres**, um por posição:

| Código | Cor | Significado |
|---|---|---|
| `G` | 🟩 Verde | Letra correta na posição correta |
| `Y` | 🟨 Amarelo | Letra existe na palavra, posição errada |
| `B` | ⬛ Cinza | Letra não existe (ou já esgotada — §3.2) |

Exemplo: `"YGBBB"`

### 3.2 Algoritmo (CORRIGIDO — normalização adicionada)

```
função normalizar(palavra):
    # NFD decompõe "á" em "a" + acento combinante
    # categoria 'Mn' = Mark, nonspacing = os diacríticos
    retorna ''.join(c for c in unicodedata.normalize('NFD', palavra)
                    if unicodedata.category(c) != 'Mn')


função calcular_feedback(tentativa, secreta) -> string de 5 chars:

    # === NOVO NA v1.1 ===
    tentativa = normalizar(tentativa)
    secreta   = normalizar(secreta)

    resultado = ['B'] * 5
    estoque = Counter(secreta)

    # PASSADA 1 — Verdes
    para i em 0..4:
        se tentativa[i] == secreta[i]:
            resultado[i] = 'G'
            estoque[tentativa[i]] -= 1

    # PASSADA 2 — Amarelos (só o que sobrou no estoque)
    para i em 0..4:
        se resultado[i] == 'G':
            continue
        se estoque[tentativa[i]] > 0:
            resultado[i] = 'Y'
            estoque[tentativa[i]] -= 1
        # senão permanece 'B'

    retorna resultado
```

**Otimização**: se o léxico já é armazenado normalizado (§2.4), a normalização
dentro desta função vira redundante no caminho quente. Normalizar **uma vez** na
carga do léxico e manter a função pura sobre entrada já normalizada — mas
normalizar defensivamente na fronteira de entrada do usuário (§7.2).

**Por que duas passadas**: verdes têm prioridade absoluta. Em passada única, uma
letra em posição errada pode "roubar" o estoque de uma que deveria ficar verde.

### 3.3 Casos de teste obrigatórios (ATUALIZADO)

| # | Secreta | Tentativa | Esperado | Testa |
|---|---|---|---|---|
| 1 | `arara` | `ratos` | `YYBBB` | Amarelos sem verdes; estoque múltiplo |
| 2 | `arara` | `arara` | `GGGGG` | Acerto total |
| 3 | `carro` | `rerum` | `BGBYB` | Duplicada na tentativa, secreta tem 2 R |
| 4 | `banco` | `barra` | `GGBBB` | Duplicada na tentativa, secreta tem 1 R → só um colorido |
| 5 | `saúde` | `saude` | `GGGGG` ⚠️ | **MUDOU** — antes era `GGBGG` |
| 6 | `saude` | `saúde` | `GGGGG` ⚠️ | **NOVO** — normalização simétrica |
| 7 | `açude` | `acude` | `GGGGG` ⚠️ | **NOVO** — cedilha normaliza (verificar, §3.5) |
| 8 | `termo` | `metro` | verificar | Verdes têm prioridade sobre amarelos |

⚠️ **Ao migrar**: varrer o suite inteiro procurando qualquer caso que assumisse
"acento = letra distinta". O caso 5 é o exemplo conhecido, mas podem existir
outros.

### 3.4 Validação cruzada

Jogar alguns dias no term.ooo, registrar tentativa + feedback recebido, e
conferir que `calcular_feedback` reproduz o mesmo resultado.

Caso real disponível: a palavra de 5 jan 2022 foi **"festa"**.

### 3.5 Ponto em aberto: cedilha

`unicodedata.normalize('NFD', 'ç')` decompõe em `c` + cedilha combinante, então
a normalização padrão transforma **ç → c**.

Isso é *provavelmente* correto (cedilha é diacrítico, e a ajuda do jogo fala em
"acentos" genericamente), mas **não foi confirmado empiricamente**. São 172
palavras do léxico com "ç".

**Como resolver**: quando a resposta do dia contiver "ç", digitar a variante com
"c" e observar a cor. Verde → normalização correta. Cinza → "ç" é letra distinta
e precisa ser preservada na normalização.

---

## 4. Algoritmo — Entropia com prior de frequência

*(Inalterado na v1.1, exceto os totais de palavras.)*

### 4.1 Nível de sofisticação

Referências:

- **Gabriel Yshay** ([artigo](https://gabrielyshay.medium.com/qual-a-melhor-palavra-no-termo-data-science-ao-resgate-1719d6311367)):
  aplica entropia ao Termo, mas trata todas as palavras como equiprováveis e
  resolve **apenas a primeira jogada**.
- **3Blue1Brown** ([lição](https://www.3blue1brown.com/lessons/wordle/)):
  entropia + **ponderação por frequência de uso real**.

**Decisão: Nível 2 (entropia + prior) desde a v1.**

O Nível 3 (3B1B publicou correção: maximizar entropia ≠ minimizar tentativas
esperadas) fica **fora de escopo** por custo computacional.

### 4.2 Prior via softmax com temperatura

O arquivo `icf` traz **Inverse Corpus Frequency**: score **baixo = comum**,
**alto = raro**.

```
Faixa observada:
  6.14  → "muito"  (mais comum)
  8.40  → "carro"
  8.78  → "banco"
  9.49  → "termo"
 12.48  → "trave"
 12.90  → "xeque"
 16.12  → mediana
 20.52  → "ababá"
 23.13  → "obvim", "puirá" (mais raras)
```

Cobertura validada: **100%** das palavras têm ICF.

```
prior(w) = exp(-ICF(w) / T) / Σ_w' exp(-ICF(w') / T)
```

| T | Comportamento |
|---|---|
| T → 0 | Corte quase binário: só as mais comuns têm peso |
| T ≈ 1.44 | Equivale a `2^(-ICF)` normalizado |
| **T = 1** | **Valor inicial da v1** |
| T grande | Distribuição achatada |
| T → ∞ | Pesos iguais → **recupera a entropia pura (Nível 1)** |

**Implementação**: usar log-sum-exp (subtrair o máximo antes de exponenciar) para
estabilidade numérica.

**Consequência**: T é um **dial contínuo** entre Nível 1 e Nível 2 — explorado no
benchmark (§5.5).

**Nota**: o ICF também é o critério de desempate na deduplicação do léxico
(§2.2, passo 3).

### 4.3 Cálculo da entropia

Para cada tentativa candidata `g`, dado o conjunto atual de candidatas `C`:

```
1. Para cada palavra c em C:
       padrão = calcular_feedback(g, c)
       bucket[padrão] += prior(c)

2. Normalizar: p(padrão) = bucket[padrão] / Σ buckets

3. Entropia(g) = -Σ_padrões  p × log₂(p)
```

Escolhe-se o `g` de **maior entropia**.

**Atenção:**
- Ignorar padrões com `p = 0` (`log(0)` explode)
- O espaço de `g` é o **léxico completo** (6.046), não apenas `C` — permite
  "queimar" uma tentativa numa palavra improvável mas informativa
- O prior pesa apenas as **candidatas** `c`; nunca restringe quais `g` testar

### 4.4 Filtragem de candidatas

Após cada rodada, manter em `C` apenas as palavras `c` tais que
`calcular_feedback(tentativa, c) == feedback_recebido`.

Reutilizar **exatamente** a função de §3 — não implementar uma segunda lógica de
filtragem por regras (fonte clássica de inconsistência).

### 4.5 Regra de endgame

**Problema**: com poucas candidatas, a palavra de maior entropia pode não estar
entre elas. Separa melhor em teoria, mas desperdiça uma tentativa.

```
se len(C) == 1:
    chutar a única candidata

se len(C) == 2:
    chutar a candidata de maior prior

se é a última tentativa (6ª) ou penúltima com C pequeno:
    chutar a candidata de maior prior (nunca uma não-candidata)

caso contrário:
    chutar a palavra de maior entropia do léxico completo
```

**Desempate de entropia**: preferir a palavra que **também é candidata**; em novo
empate, a de maior prior.

---

## 5. Benchmark

### 5.1 Pergunta central

> A estratégia de entropia compensa o custo computacional, ou uma heurística
> simples chega perto?

### 5.2 Estratégias competidoras

| # | Estratégia | Descrição |
|---|---|---|
| 1 | **Aleatória** | Chuta qualquer candidata restante. Piso de comparação. |
| 2 | **Frequência de letras** | Score pela frequência das letras. A abordagem que o Yshay mostrou ser furada — e a que a maioria dos solvers existentes usa. |
| 3 | **Mais provável** | Chuta a candidata de maior prior. Simples e competitiva. |
| 4 | **Entropia (nosso)** | §4, com T configurável. |

Bater as estratégias 2 e 3 é a demonstração de valor do projeto.

### 5.3 Conjuntos de teste (ATUALIZADO)

| Bateria | Palavras secretas | Objetivo |
|---|---|---|
| **Realista** | ~800–1.500 de menor ICF | Simula o jogo real |
| **Stress test** | Léxico completo (6.046) | Robustez em pior caso |

> A faixa da bateria realista foi reduzida (era 1.000–2.000) porque o léxico
> encolheu de 8.996 → 6.046.

**Justificativa**: testar contra o léxico inteiro é irrealista — o Termo nunca
sorteia "ababá". Isso puniria todas as estratégias e inflaria a média.

### 5.4 Métricas

| Métrica | Por quê |
|---|---|
| **Tentativas médias** | Métrica principal |
| **Taxa de vitória em 6** | Média 4.2 com 5% de derrota é pior que 4.4 com 0% |
| **Distribuição completa** | Jogos em 1, 2, 3, 4, 5, 6 tentativas — o gráfico do projeto |
| **Pior caso** | Qual palavra quebra cada estratégia |
| **Tempo de execução** | Entropia é ~4x mais lenta; relevante se virar app |

### 5.5 Experimento de temperatura (o diferencial)

Varrer **T** ∈ {0.5, 1, 2, 5, 10, ∞} e plotar **tentativas médias × T**.

Como T→∞ recupera a entropia pura, este gráfico compara Nível 1 vs Nível 2 de
forma **contínua**, respondendo "quanto o prior de frequência ajuda?". É
provavelmente o resultado mais interessante do projeto.

### 5.6 Expectativa de resultado

Sanity check (não é meta):

- Entropia em torno de **3,5–4 tentativas** na bateria realista
- "Mais provável" deve chegar perto na média
- A diferença real aparece na **taxa de vitória** e no **pior caso**

> Com o léxico menor (6.046 vs 8.996), espera-se desempenho **um pouco melhor**
> que na v1.0 — menos candidatas para separar.

Números muito distantes disso indicam bug — mais provavelmente em §3
(feedback) ou §4.4 (filtragem).

---

## 6. Performance

### 6.1 O problema

Entropia da primeira jogada = 6.046 × 6.046 ≈ **36,5 milhões** de chamadas a
`calcular_feedback`. Em Python puro: inviável para uso interativo e
completamente inviável para o benchmark.

> A v1.0 estimava ~81M comparações (8.996²). Com o léxico corrigido caiu para
> ~36,5M — **55% menos trabalho**.

### 6.2 Soluções obrigatórias

**1. Matriz de padrões pré-computada**

`M[i][j] = feedback(palavra_i, palavra_j)`, cada padrão codificado como inteiro
(base 3, 5 dígitos → 0..242, cabe em `uint8`).

```
Dimensão: 6.046 × 6.046 ≈ 36,5M células
Tipo:     uint8
Tamanho:  ~37 MB   (era ~81 MB na v1.0)
```

Serializar em `.npy`. Cada rodada vira operação vetorizada numpy (`bincount`
sobre uma fatia) — milissegundos.

> ⚠️ **MIGRAÇÃO**: deletar qualquer `.npy` gerado pela v1.0. Ver §0.3.

**2. Primeira palavra em cache**

A melhor abertura não depende de feedback. Calcular uma vez e salvar.

> ⚠️ **MIGRAÇÃO**: recalcular. A palavra da v1.0 está errada.

### 6.3 Codificação de padrão

```
Mapear: B → 0, Y → 1, G → 2
"YGBBB" → [1,2,0,0,0] → 1·3⁴ + 2·3³ = 135
Faixa: 0..242  (3⁵ = 243 padrões)
```

Manter funções de conversão string ↔ inteiro para I/O.

---

## 7. Interface (CLI v1)

### 7.1 Fluxo

```
$ destermo

Melhor abertura: ______

> tentativa: ______
> feedback:  YGBBB

  Candidatas restantes: 47
  Próxima sugestão: ______
  Alternativas: ______, ______, ______

> tentativa: ...
```

### 7.2 Entrada (RESOLVIDO na v1.1)

- **Tentativa**: 5 letras, **digitadas SEM acento**.
  O Termo preenche acentos automaticamente, então o jogador nunca os digita.
  Aceitar entrada acentuada também (normalizar defensivamente), mas não exigir.
- **Feedback**: 5 chars em `G`/`Y`/`B`, case-insensitive.

> Isto resolve a questão que ficou em aberto na v1.0 §7.2.

**Saída**: exibir as sugestões na forma **acentuada** (§2.4) — é o que o jogador
verá na tela do jogo.

### 7.3 Tratamento de erros

| Situação | Comportamento |
|---|---|
| Feedback com tamanho ≠ 5 ou char inválido | Pedir novamente |
| Tentativa fora do léxico | Avisar, mas **permitir** (§2.6) |
| `C` fica vazio | Avisar: ou o feedback foi digitado errado, ou a palavra do dia não está no léxico (§2.6). Oferecer desfazer a última rodada. |
| Feedback logicamente impossível | Detectar e avisar **antes** de zerar `C` |

**Nunca crashar** por `C` vazio — é cenário esperado dado §2.6.

### 7.4 Desacoplamento

O motor (léxico, feedback, entropia, estratégias) fica **completamente separado**
da interface. A v1 é CLI, mas se virar app web, bot ou API, apenas a camada de
I/O muda.

> Este desacoplamento é o que tornou a migração v1.0 → v1.1 barata (§0.2).

---

## 8. Ordem de implementação

### 8.1 Do zero

1. **`feedback.py`** + testes de §3.3 — nada mais funciona se isso falhar
2. **`lexico.py`** — download, filtragem, normalização, dedupe, ICF, prior
3. **Matriz pré-computada** (§6.2)
4. **`entropia.py`** — entropia + regra de endgame
5. **`solver.py`** — CLI
6. **`estrategias.py`** — as 3 heurísticas concorrentes
7. **`benchmark.py`** — simulação, métricas, varredura de T
8. **Análise/gráficos**

Passos 1–5 entregam a ferramenta. Passos 6–8 entregam a análise.

### 8.2 Migrando da v1.0

1. **`feedback.py`**: adicionar normalização (§3.2). Rodar testes.
2. **Testes**: corrigir caso 5, adicionar casos 6 e 7 (§3.3). Varrer o suite
   procurando outros casos com a regra antiga.
3. **`lexico.py`**: remover conjugações, normalizar, deduplicar (§2.2).
   Verificar: **6.046 palavras**.
4. **Deletar caches**: matriz `.npy` e primeira palavra (§0.3). ← *não esquecer*
5. **Regerar** matriz e primeira palavra.
6. **`solver.py`**: aceitar entrada sem acento, exibir com acento (§7.2).
7. **`benchmark.py`**: ajustar faixa da bateria realista (§5.3).
8. **Rodar tudo de novo** — resultados anteriores são inválidos.

---

## 9. Checklist de decisões

| Item | Decisão |
|---|---|
| Nome | **Destermo** |
| Escopo v1 | Termo clássico apenas |
| Fonte do léxico | `github.com/fserb/pt-br` (MIT) |
| Composição | Só `lexico`, 5 letras, **sem conjugações** |
| **Acentos** | **Normalizar (não afetam as cores)** |
| **Colisões** | **Deduplicar por menor ICF** |
| **Tamanho final** | **6.046 palavras** |
| Curadoria adicional | Nenhuma na v1 |
| Listas | Lista única |
| Formato de feedback | String `G`/`Y`/`B` |
| Algoritmo de feedback | Duas passadas com consumo de estoque, sobre forma normalizada |
| Nível do algoritmo | Nível 2 — entropia + prior |
| Prior | Softmax sobre ICF com temperatura T |
| T inicial | 1 (hiperparâmetro) |
| Espaço de tentativas | Léxico completo |
| Endgame | Regra explícita (§4.5) |
| Linguagem | Python + numpy |
| Interface v1 | CLI desacoplada |
| Entrada do usuário | Sem acento; saída com acento |
| Estratégias do benchmark | Aleatória, freq. letras, mais provável, entropia |
| Baterias | Realista (~800–1.500) + stress (6.046) |
| Métricas | Média, vitória, distribuição, pior caso, tempo |
| Experimento extra | Varredura de T |

---

## 10. Fora de escopo (v1)

- Dueto e Quarteto
- Nível 3 do algoritmo (minimização direta de tentativas esperadas)
- Curadoria manual adicional do léxico
- Interface web / bot / API
- Separação formal entre lista de respostas e lista de tentativas válidas

---

## 11. Pontos em aberto

| # | Questão | Como resolver |
|---|---|---|
| 1 | "ç" normaliza para "c"? (§3.5) | Teste empírico no term.ooo — 172 palavras afetadas |
| 2 | Léxico é aproximação (~6k vs ~18k oficial) | Aceito na v1; monitorar frequência de `C` vazio no uso real |
