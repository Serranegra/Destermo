# Termo Solver — Especificação Técnica (v1)

> Solver de Termo (Wordle em português) baseado em entropia de Shannon / teoria da
> informação. Combina uma ferramenta prática (sugestão da melhor próxima palavra a
> cada rodada) com um componente de análise (benchmark comparando a estratégia de
> entropia contra heurísticas mais simples).

---

## 1. Visão geral

### 1.1 O que é

Uma ferramenta que **assiste** o jogador do Termo. **Não é um clone do jogo.**

O fluxo de uso é:

```
1. Usuário joga no term.ooo e digita uma palavra
2. O site devolve o feedback colorido (🟩🟨⬛)
3. Usuário informa ao solver: qual palavra tentou + qual feedback recebeu
4. Solver calcula e sugere a próxima melhor palavra
5. Repete até acertar
```

O solver nunca conhece a palavra secreta. Ele apenas mantém o conjunto de
candidatas compatíveis com todo o feedback recebido até então, e escolhe a
próxima tentativa que maximiza a informação esperada.

### 1.2 Duas frentes

| Frente | Descrição |
|---|---|
| **Ferramenta** | CLI interativa para uso real ao jogar |
| **Análise** | Benchmark comparando estratégias, medindo tentativas médias |

Ambas compartilham o mesmo motor de entropia.

### 1.3 Escopo da v1

- ✅ Termo clássico (palavra única de 5 letras, 6 tentativas)
- ❌ Dueto (2 palavras simultâneas) — fora de escopo
- ❌ Quarteto (4 palavras simultâneas) — fora de escopo

O motor é reaproveitável para as variantes no futuro, mas a v1 não as implementa.

### 1.4 Stack

- **Linguagem**: Python
- **Numérico**: numpy (essencial — ver seção 6 sobre performance)
- **Interface v1**: CLI

---

## 2. Léxico

### 2.1 Fonte

Repositório oficial do criador do Termo (Fernando Serboncini), licença MIT:

```
https://github.com/fserb/pt-br
```

Arquivos usados (baixar de `raw.githubusercontent.com/fserb/pt-br/master/`):

| Arquivo | Conteúdo | Uso |
|---|---|---|
| `lexico` | 145.744 entradas — dicionário geral | Palavras de 5 letras |
| `conjugações` | 195.751 formas verbais | Palavras de 5 letras |
| `icf` | 419.486 palavras com score de frequência | Prior de probabilidade |

### 2.2 Construção da lista

```
lexico5      = palavras de exatamente 5 CARACTERES em `lexico`        → 6.301
conjugações5 = palavras de exatamente 5 CARACTERES em `conjugações`   → 3.543
interseção                                                            →   848
LÉXICO FINAL = união dos dois conjuntos                               → 8.996
```

**Atenção**: contar **caracteres**, não bytes. Acentos em UTF-8 ocupam mais de
1 byte; `len()` em `str` Python já faz o correto, mas ferramentas de shell
(`awk length`) não.

### 2.3 Características da lista

- 8.996 palavras
- Todas minúsculas
- ~21,5% contêm acento
- Nenhuma contém espaço, hífen ou caractere fora de `[a-zà-ÿ]`
- Ordenada alfabeticamente, uma palavra por linha, UTF-8

### 2.4 Decisões tomadas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Conjugações verbais | **Incluir** | Mais próximo do jogo real ("fugia", "curas", "zarpa" são válidas) |
| Curadoria de obscuras | **Não aplicar (v1)** | Mais rápido; ajustar depois se aparecer lixo |
| Respostas vs tentativas | **Lista única** | Simplicidade na v1; o prior (seção 4) já diferencia probabilidade |
| Acentos | **Letras distintas** | "á" ≠ "a" para todos os efeitos — fiel ao jogo |

### 2.5 Ressalva conhecida

A Wikipedia menciona que o Termo usa ~18.000 palavras de 5 letras. Nossa lista
tem 8.996. A diferença provavelmente vem de o repositório `fserb/pt-br` ser um
léxico geral publicado depois, e não o dump exato usado em produção no jogo.

**Consequência**: nossa lista é uma **aproximação** de fonte legítima, não uma
cópia da lista oficial. É possível que o solver encontre situações onde a
palavra do dia não está na nossa lista. Isso deve ser tratado sem crash
(ver seção 7.3).

---

## 3. Regras de feedback

### 3.1 Representação

Feedback é uma **string de 5 caracteres**, um por posição:

| Código | Cor | Significado |
|---|---|---|
| `G` | 🟩 Verde | Letra correta na posição correta |
| `Y` | 🟨 Amarelo | Letra existe na palavra, posição errada |
| `B` | ⬛ Cinza/preto | Letra não existe (ou já esgotada — ver 3.2) |

Exemplo: `"YGBBB"`

### 3.2 Algoritmo de cálculo (duas passadas)

Esta é a parte mais sutil da implementação. Deve reproduzir **exatamente** a
regra do Termo/Wordle, baseada em **consumo de estoque de letras**.

```
função calcular_feedback(tentativa, secreta) -> string de 5 chars:

    resultado = ['B', 'B', 'B', 'B', 'B']
    estoque = Counter(secreta)          # contagem de cada letra da secreta

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

**Por que duas passadas**: verdes têm prioridade absoluta sobre amarelos. Se
processado numa passada só, uma letra em posição errada pode "roubar" o estoque
de uma letra que deveria ficar verde mais adiante.

### 3.3 Casos de teste obrigatórios

A implementação **deve** passar nestes casos antes de qualquer outra coisa:

| # | Secreta | Tentativa | Esperado | Testa |
|---|---|---|---|---|
| 1 | `arara` | `ratos` | `YYBBB` | Amarelos sem verdes; estoque múltiplo |
| 2 | `arara` | `arara` | `GGGGG` | Acerto total |
| 3 | `carro` | `rerum` | `BGBYB` | Letra duplicada na tentativa, secreta tem 2 R |
| 4 | `banco` | `barra` | `GGBBB` | Duplicada na tentativa, secreta tem 1 R → só um colorido |
| 5 | `saúde` | `saude` | `GGBGG` | Acento: "ú" ≠ "u" → cinza |
| 6 | `termo` | `metro` | `YYYYG`* | Verdes têm prioridade sobre amarelos |

\* Verificar manualmente antes de fixar — o ponto é que o caso exista no suite.

**Regra de ouro para o caso 4**: se a tentativa tem 2 ocorrências de uma letra e
a secreta tem apenas 1, **apenas uma** fica colorida (G ou Y); a outra fica `B`.
Este é o bug mais comum em implementações ingênuas.

### 3.4 Validação cruzada

Além dos testes unitários, validar contra o jogo real: jogar alguns dias no
term.ooo, registrar tentativa + feedback recebido, e conferir que
`calcular_feedback` reproduz o mesmo resultado.

---

## 4. Algoritmo — Entropia com prior de frequência

### 4.1 Nível de sofisticação

Referências consultadas:

- **Gabriel Yshay** ([artigo](https://gabrielyshay.medium.com/qual-a-melhor-palavra-no-termo-data-science-ao-resgate-1719d6311367)):
  aplica entropia ao Termo, mas trata todas as palavras como equiprováveis e
  resolve **apenas a primeira jogada** — não fecha o loop do jogo.
- **3Blue1Brown** ([lição](https://www.3blue1brown.com/lessons/wordle/)):
  entropia + **ponderação por frequência de uso real** das palavras. É a camada
  que falta no artigo brasileiro.

**Decisão: implementar o Nível 2 (entropia + prior de frequência) desde a v1.**

Um terceiro nível existe (3B1B publicou um vídeo de correção: maximizar entropia
não é *idêntico* a minimizar tentativas esperadas — o ótimo real exige simular o
jogo completo por candidata). **Fora de escopo da v1** por custo computacional.

### 4.2 Prior via softmax com temperatura

O arquivo `icf` traz **Inverse Corpus Frequency**: score **baixo = palavra
comum**, score **alto = palavra rara**.

```
Validação já feita: 100% das 8.996 palavras têm score ICF.

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

Transformação ICF → probabilidade:

```
prior(w) = exp(-ICF(w) / T) / Σ_w' exp(-ICF(w') / T)
```

**T** (temperatura) controla o formato da distribuição:

| T | Comportamento |
|---|---|
| T → 0 | Corte quase binário: só as palavras mais comuns têm peso |
| T ≈ 1.44 | Equivale a `2^(-ICF)` normalizado |
| **T = 1** | **Valor inicial da v1** |
| T grande | Distribuição achatada |
| T → ∞ | Todos os pesos iguais → **recupera exatamente a entropia pura (Nível 1)** |

**Nota de implementação**: usar log-sum-exp para estabilidade numérica
(subtrair o máximo antes de exponenciar), já que ICF/T pode gerar underflow.

**Consequência importante**: como T→∞ recupera o Nível 1, **T é um dial contínuo
entre os dois níveis**. Isso é explorado no benchmark (seção 5.4).

### 4.3 Cálculo da entropia

Para cada palavra candidata a tentativa `g`, dado o conjunto atual de candidatas
`C` (palavras ainda compatíveis com todo o feedback recebido):

```
1. Para cada palavra c em C:
       padrão = calcular_feedback(g, c)
       bucket[padrão] += prior(c)

2. Normalizar: p(padrão) = bucket[padrão] / Σ buckets

3. Entropia(g) = Σ_padrões  p × log₂(1 / p)
                = -Σ_padrões  p × log₂(p)
```

Escolhe-se o `g` com **maior entropia**.

**Pontos de atenção:**
- Ignorar padrões com `p = 0` (não contribuem; `log(0)` explode).
- O espaço de tentativas `g` é o **léxico completo** (8.996 palavras), não apenas
  as candidatas restantes `C`. Isso permite "queimar" uma tentativa numa palavra
  improvável mas altamente informativa.
- O prior entra apenas no peso das **candidatas** `c`, nunca restringe quais `g`
  podem ser testadas.

### 4.4 Filtragem de candidatas

Após cada rodada, `C` é reduzido: mantém-se apenas as palavras `c` tais que
`calcular_feedback(tentativa, c) == feedback_recebido`.

Isto reutiliza exatamente a mesma função da seção 3 — não implementar uma
segunda lógica de filtragem por regras (fonte clássica de inconsistência).

### 4.5 Regra de endgame

**Problema**: quando restam poucas candidatas, a palavra de maior entropia pode
ser uma palavra que **não está** entre as candidatas. Ela separa melhor em teoria,
mas desperdiça uma tentativa que poderia ter chance de acertar.

**Exemplo**: restam 2 candidatas e há 3 tentativas sobrando. Chutar uma das duas
tem 50% de acerto imediato. Chutar uma terceira palavra "ótima" garante
identificar a resposta, mas gasta uma tentativa a mais.

**Regra a implementar:**

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

**Desempate de entropia**: quando duas palavras têm entropia (quase) igual,
preferir a que **também é candidata** — ela tem chance não-nula de ser a resposta,
o que é ganho grátis. Em caso de novo empate, preferir a de maior prior.

---

## 5. Benchmark

### 5.1 Pergunta central

> A estratégia de entropia realmente compensa o custo computacional, ou uma
> heurística simples chega perto?

### 5.2 Estratégias competidoras

| # | Estratégia | Descrição |
|---|---|---|
| 1 | **Aleatória** | Chuta qualquer candidata restante. Piso de comparação. |
| 2 | **Frequência de letras** | Score = soma/média das frequências das letras. É a abordagem que o Yshay mostrou ser furada, e a que a maioria dos solvers existentes usa. |
| 3 | **Mais provável** | Chuta sempre a candidata de maior prior (menor ICF). Simples e surpreendentemente competitiva. |
| 4 | **Entropia (nosso)** | Seção 4, com T configurável. |

As três primeiras são baratas e servem como régua. Bater as estratégias 2 e 3 é
literalmente a demonstração de valor do projeto.

### 5.3 Conjuntos de teste

Rodar **duas baterias**:

| Bateria | Conjunto de palavras secretas | Objetivo |
|---|---|---|
| **Realista** | ~1.000–2.000 palavras de menor ICF (mais comuns) | Simula o comportamento do jogo real |
| **Stress test** | Léxico completo (8.996) | Robustez em pior caso |

**Justificativa**: testar contra o léxico inteiro é irrealista — o Termo nunca vai
sortear "ababá". Isso puniria todas as estratégias e inflaria artificialmente o
número médio de tentativas.

### 5.4 Métricas

Para cada estratégia × bateria:

| Métrica | Por quê |
|---|---|
| **Tentativas médias** | Métrica principal |
| **Taxa de vitória em 6** | Média 4.2 com 5% de derrota é pior que 4.4 com 0% |
| **Distribuição completa** | Quantos jogos em 1, 2, 3, 4, 5, 6 tentativas — o gráfico do projeto |
| **Pior caso** | Qual palavra quebra cada estratégia |
| **Tempo de execução** | Entropia é ~4x mais lenta; relevante se virar app |

### 5.5 Experimento de temperatura (o diferencial)

Varrer **T** ∈ {0.5, 1, 2, 5, 10, ∞} e plotar **tentativas médias em função de T**.

Como T→∞ recupera a entropia pura, este único gráfico compara Nível 1 vs Nível 2
de forma **contínua**, respondendo empiricamente "quanto o prior de frequência
ajuda?". É provavelmente o resultado mais interessante do projeto.

### 5.6 Expectativa de resultado

Palpite para servir de sanity check (não é meta):

- Entropia deve ficar em torno de **3,5–4 tentativas** na bateria realista
- "Mais provável" deve chegar surpreendentemente perto na média
- A diferença real deve aparecer na **taxa de vitória** e no **pior caso**, não na média

Se os números saírem muito distantes disso, provavelmente há bug — mais
provavelmente na função de feedback (seção 3) ou na filtragem (4.4).

---

## 6. Performance

### 6.1 O problema

Calcular a entropia da primeira jogada exige comparar cada uma das 8.996 palavras
contra cada uma das 8.996 candidatas: **~81 milhões de chamadas** a
`calcular_feedback`. Em Python puro isso leva de minutos a horas — inviável para
uso interativo, e **completamente inviável para o benchmark**, que roda milhares
de jogos completos.

### 6.2 Soluções obrigatórias

**1. Matriz de padrões pré-computada**

Calcular uma única vez a matriz `M[i][j] = feedback(palavra_i, palavra_j)`,
codificando cada padrão como inteiro (base 3, 5 dígitos → 0..242, cabe em
`uint8`).

```
Dimensão: 8.996 × 8.996 = ~81M células
Tipo:     uint8
Tamanho:  ~81 MB  → perfeitamente viável em memória e em disco
```

Serializar em disco (`.npy`) para não recalcular a cada execução. Com a matriz
pronta, cada rodada vira operação vetorizada numpy (`bincount` sobre uma fatia),
tipicamente milissegundos.

**2. Primeira palavra em cache**

A melhor abertura é sempre a mesma (não depende de nenhum feedback). Calcular uma
vez e salvar. Isso elimina o cálculo mais pesado do uso diário — na prática o
solver só faz trabalho real a partir da 2ª rodada, quando `C` já encolheu bastante.

### 6.3 Codificação de padrão

```
Mapear: B → 0, Y → 1, G → 2
Padrão "YGBBB" → dígitos [1,2,0,0,0] → 1·3⁴ + 2·3³ + 0 + 0 + 0 = 135
Faixa total: 0 .. 242  (3⁵ = 243 padrões possíveis)
```

Manter funções de conversão string ↔ inteiro para entrada/saída do usuário.

---

## 7. Interface (CLI v1)

### 7.1 Fluxo

```
$ python solver.py

Melhor abertura: ______

> tentativa: ______
> feedback:  YGBBB

  Candidatas restantes: 47
  Próxima sugestão: ______
  Alternativas: ______, ______, ______

> tentativa: ...
```

### 7.2 Entrada

- **Tentativa**: palavra de 5 letras (aceitar com ou sem acento na digitação? —
  decidir; sugestão: exigir com acento, já que acentos são letras distintas)
- **Feedback**: string de 5 chars em `G`/`Y`/`B`, case-insensitive

### 7.3 Tratamento de erros

| Situação | Comportamento |
|---|---|
| Feedback com tamanho ≠ 5 ou char inválido | Pedir novamente |
| Tentativa fora do léxico | Avisar, mas **permitir** (o Termo pode aceitar palavra que não temos) |
| `C` fica vazio (nenhuma candidata compatível) | Avisar que ou o feedback foi digitado errado, ou a palavra do dia não está no nosso léxico (ver 2.5). Oferecer desfazer a última rodada. |
| Feedback logicamente impossível | Detectar e avisar antes de zerar `C` |

**Nunca crashar** por causa de `C` vazio — é um cenário esperado dado a ressalva
2.5.

### 7.4 Desacoplamento

O motor (léxico, feedback, entropia, estratégias) deve ficar **completamente
separado** da camada de interface. A v1 é CLI, mas o formato final está em aberto
— se depois virar app web, bot de Telegram ou API, apenas a camada de I/O muda.

---

## 8. Ordem sugerida de implementação

1. **`feedback.py`** + suite de testes da seção 3.3 — nada mais funciona se isso
   estiver errado
2. **`lexico.py`** — download, filtragem, carga do ICF, cálculo do prior
3. **Matriz pré-computada** (seção 6.2) — destrava tudo que vem depois
4. **`entropia.py`** — cálculo de entropia + regra de endgame
5. **`solver.py`** — CLI interativa
6. **`estrategias.py`** — as 3 heurísticas concorrentes
7. **`benchmark.py`** — simulação, métricas, varredura de T
8. **Análise/gráficos** — distribuição de tentativas, curva de T

Os passos 1–5 entregam a ferramenta utilizável. Os passos 6–8 entregam a análise.

---

## 9. Checklist de decisões fechadas

| Item | Decisão |
|---|---|
| Escopo v1 | Termo clássico apenas |
| Fonte do léxico | `github.com/fserb/pt-br` (MIT) |
| Composição | léxico + conjugações, 5 letras → 8.996 palavras |
| Curadoria | Nenhuma na v1 |
| Listas | Lista única (sem separar respostas/tentativas) |
| Acentos | Letras distintas |
| Formato de feedback | String `G`/`Y`/`B` |
| Algoritmo de feedback | Duas passadas com consumo de estoque |
| Nível do algoritmo | Nível 2 — entropia + prior de frequência |
| Prior | Softmax sobre ICF, com temperatura T |
| T inicial | 1 (hiperparâmetro a explorar) |
| Espaço de tentativas | Léxico completo |
| Endgame | Regra explícita (seção 4.5) |
| Linguagem | Python + numpy |
| Interface v1 | CLI (desacoplada do motor) |
| Estratégias do benchmark | Aleatória, freq. de letras, mais provável, entropia |
| Baterias | Realista (~1-2k comuns) + stress test (completo) |
| Métricas | Média, taxa de vitória, distribuição, pior caso, tempo |
| Experimento extra | Varredura de T |

---

## 10. Fora de escopo (v1)

- Dueto e Quarteto
- Nível 3 do algoritmo (minimização direta de tentativas esperadas via simulação
  completa por candidata)
- Curadoria manual do léxico
- Interface web / bot / API
- Separação formal entre lista de respostas e lista de tentativas válidas
