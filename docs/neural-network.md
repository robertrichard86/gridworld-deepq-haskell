# 🔮 Rede Neural Artificial — Documentação Técnica

> **Documento de Referência Técnica — Implementação Manual de Rede Neural**
> Projeto: gridworld-deepq-haskell | Módulo: `NeuralNetwork.hs`

---

## 📋 Índice

- [O que é uma Rede Neural Artificial](#-o-que-é-uma-rede-neural-artificial)
- [Arquitetura da Rede Neste Projeto](#-arquitetura-da-rede-neste-projeto)
- [Forward Propagation](#-forward-propagation)
- [Função de Ativação: ReLU](#-função-de-ativação-relu)
- [Backward Propagation](#-backward-propagation)
- [Inicialização de Pesos: Xavier/He](#-inicialização-de-pesos-xavierhe)
- [Gradient Descent](#-gradient-descent)
- [Gradient Clipping](#-gradient-clipping)
- [Função de Perda: MSE](#-função-de-perda-mse)
- [Fluxo Matemático Completo](#-fluxo-matemático-completo)
- [Implementação em Haskell](#-implementação-em-haskell)
- [Visualização dos Tensores](#-visualização-dos-tensores)

---

## 🧠 O que é uma Rede Neural Artificial

Uma **Rede Neural Artificial** (RNA) é um modelo computacional inspirado na estrutura do cérebro humano. É composta por **neurônios artificiais** organizados em **camadas**, onde cada neurônio recebe entradas, aplica uma transformação matemática e produz uma saída.

### O Neurônio Artificial (Perceptron)

Um único neurônio computa:

```
z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b = Σᵢ wᵢ·xᵢ + b = w⃗ᵀ·x⃗ + b
a = f(z)
```

Onde:
- **x⃗** — vetor de entradas
- **w⃗** — vetor de pesos (parâmetros aprendíveis)
- **b** — bias (viés)
- **f** — função de ativação
- **z** — soma ponderada (*pre-activation*)
- **a** — ativação (*post-activation*)

```
     x₁ ──→ [w₁] ──┐
     x₂ ──→ [w₂] ──┤
     x₃ ──→ [w₃] ──┼──→ Σ + b ──→ f(z) ──→ a (saída)
     ...            │
     xₙ ──→ [wₙ] ──┘
```

### Camadas

Uma rede neural organiza neurônios em camadas:

| Tipo de Camada | Função | Neste Projeto |
|---|---|---|
| **Camada de Entrada** | Recebe os dados brutos | 12 neurônios (estado do grid) |
| **Camada(s) Oculta(s)** | Extrai features intermediárias | 1 camada × 64 neurônios |
| **Camada de Saída** | Produz o resultado final | 4 neurônios (Q-values por ação) |

---

## 🏗️ Arquitetura da Rede Neste Projeto

A rede neural implementada é do tipo **feedforward** (*Multi-Layer Perceptron*, MLP) com **uma camada oculta**:

```
                    CAMADA DE         CAMADA           CAMADA DE
                    ENTRADA           OCULTA           SAÍDA
                    (12 neurônios)     (64 neurônios)    (4 neurônios)

  agentRow  [0]  ─→ ┐               ┌──────┐         ┌──────┐
  agentCol  [1]  ─→ ┤               │  h₁  │    ──→  │ Q(Up)│
  goalRow   [2]  ─→ ┤    W₁         │  h₂  │    ──→  │Q(Dn) │
  goalCol   [3]  ─→ ┤  (64×12)      │  h₃  │  W₂    │Q(Lt) │
  distRow   [4]  ─→ ┤    +          │  ... │ (4×64) │Q(Rt) │
  distCol   [5]  ─→ ┤   b₁         │  ... │   +    └──────┘
  obs1_dr   [6]  ─→ ┤  (64)        │  ... │  b₂
  obs1_dc   [7]  ─→ ┤               │  ... │  (4)
  obs2_dr   [8]  ─→ ┤    ReLU       │  h₆₂│
  obs2_dc   [9]  ─→ ┤               │  h₆₃│  Linear
  obs3_dr  [10]  ─→ ┤               │  h₆₄│ (sem ativação)
  obs3_dc  [11]  ─→ ┘               └──────┘
```

### Dimensões dos Parâmetros

| Parâmetro | Dimensão | Total de Valores | Descrição |
|:---:|:---:|:---:|---|
| **W₁** | 64 × 12 | **768** | Pesos da entrada para a camada oculta |
| **b₁** | 64 | **64** | Biases da camada oculta |
| **W₂** | 4 × 64 | **256** | Pesos da camada oculta para a saída |
| **b₂** | 4 | **4** | Biases da camada de saída |
| **Total** | — | **1.092** | Parâmetros treináveis da rede |

### Representação em Haskell

```haskell
data Network = Network
  { netWeights1  :: Matrix Double    -- W₁: 64 × 12
  , netBias1     :: Vector Double    -- b₁: 64
  , netWeights2  :: Matrix Double    -- W₂: 4 × 64
  , netBias2     :: Vector Double    -- b₂: 4
  }
```

---

## ➡️ Forward Propagation

O **Forward Propagation** (propagação direta) é o processo de calcular a saída da rede dado um vetor de entrada. Os dados "fluem" da entrada para a saída, camada por camada.

### Fórmulas Matemáticas

**Camada 1 (Entrada → Oculta):**

```
z₁ = W₁ · x + b₁          (soma ponderada)
h  = ReLU(z₁)              (ativação)
```

**Camada 2 (Oculta → Saída):**

```
z₂ = W₂ · h + b₂           (soma ponderada)
ŷ  = z₂                     (saída linear — sem ativação)
```

### Por que a Saída é Linear?

A camada de saída **não** usa função de ativação (é linear). Isso porque os Q-values podem ser qualquer número real (positivo ou negativo), e uma ativação como ReLU limitaria os valores a ≥ 0, ou sigmoid limitaria a [0, 1].

### Exemplo Numérico Simplificado

Para ilustrar, considere uma rede simplificada com entrada=3, oculta=2, saída=2:

```
Entrada: x = [0.5, 0.3, 0.8]

W₁ = [[0.2, -0.1, 0.4],    b₁ = [0.1, -0.1]
       [0.3,  0.5, -0.2]]

Camada 1:
  z₁ = W₁ · x + b₁
     = [0.2×0.5 + (-0.1)×0.3 + 0.4×0.8 + 0.1,
        0.3×0.5 + 0.5×0.3 + (-0.2)×0.8 + (-0.1)]
     = [0.1 - 0.03 + 0.32 + 0.1,
        0.15 + 0.15 - 0.16 - 0.1]
     = [0.49, 0.04]

  h = ReLU(z₁)
    = [max(0, 0.49), max(0, 0.04)]
    = [0.49, 0.04]

W₂ = [[0.6, -0.3],    b₂ = [0.05, -0.05]
       [0.1,  0.7]]

Camada 2:
  z₂ = W₂ · h + b₂
     = [0.6×0.49 + (-0.3)×0.04 + 0.05,
        0.1×0.49 + 0.7×0.04 + (-0.05)]
     = [0.294 - 0.012 + 0.05,
        0.049 + 0.028 - 0.05]
     = [0.332, 0.027]

Saída: ŷ = [0.332, 0.027]
→ Ação 1 (Q=0.332) é melhor que Ação 2 (Q=0.027)
→ argmax = Ação 1
```

### Implementação em Haskell

```haskell
forward :: Network -> Vector Double -> (Vector Double, Vector Double, Vector Double)
forward net input =
  let z1     = (netWeights1 net #> input) `add` netBias1 net    -- z₁ = W₁·x + b₁
      hidden = relu z1                                            -- h  = ReLU(z₁)
      z2     = (netWeights2 net #> hidden) `add` netBias2 net   -- z₂ = W₂·h + b₂
  in (z2, hidden, z1)     -- retorna: (saída, ativações ocultas, pre-ativação)

predict :: Network -> Vector Double -> Vector Double
predict net input =
  let (output, _, _) = forward net input
  in output
```

> [!NOTE]
> A função `forward` retorna não apenas a saída (`z2`), mas também os valores intermediários (`hidden` e `z1`). Esses valores são necessários durante o backward pass para calcular os gradientes.

---

## ⚡ Função de Ativação: ReLU

### O que é ReLU?

A **ReLU** (*Rectified Linear Unit*) é a função de ativação mais utilizada em redes neurais modernas. Ela é definida como:

```
ReLU(x) = max(0, x) = {
    x,  se x > 0
    0,  se x ≤ 0
}
```

### Gráfico da ReLU

```
Saída (a)
    │
  3 │              ╱
    │             ╱
  2 │            ╱
    │           ╱
  1 │          ╱
    │         ╱
  0 │────────╱─────────── Entrada (z)
    │       ╱
 -3  -2  -1   0   1   2   3
```

### Derivada da ReLU

A derivada da ReLU é essencial para o backward pass:

```
ReLU'(x) = {
    1,  se x > 0
    0,  se x ≤ 0
}
```

```
Derivada
    │
  1 │         ┌─────────
    │         │
  0 │─────────┘──────── Entrada (z)
    │
 -3  -2  -1   0   1   2   3
```

### Por que ReLU?

| Vantagem | Explicação |
|---|---|
| **Simplicidade** | Apenas `max(0, x)` — computacionalmente eficiente |
| **Sem vanishing gradient** | Gradiente = 1 para entradas positivas (não "some" como em sigmoid) |
| **Esparsidade** | Neurônios com entrada negativa são "desligados" (saída = 0) |
| **Convergência rápida** | Treinamento empircamente mais rápido que sigmoid/tanh |

### Implementação em Haskell

```haskell
relu :: Vector Double -> Vector Double
relu = cmap (max 0)                    -- aplica max(0, x) a cada elemento

reluDerivative :: Vector Double -> Vector Double
reluDerivative = cmap (\x -> if x > 0 then 1.0 else 0.0)
```

> [!TIP]
> A função `cmap` da biblioteca `hmatrix` aplica uma função escalar a cada elemento do vetor, similar a `map` em listas mas para vetores de álgebra linear.

---

## ⬅️ Backward Propagation

### O que é Backpropagation?

O **Backpropagation** (propagação reversa) é o algoritmo para calcular os **gradientes** da função de perda em relação a cada parâmetro da rede (pesos e biases). Utiliza a **regra da cadeia** do cálculo para propagar o erro da saída de volta para a entrada.

### A Regra da Cadeia

Se temos funções compostas `L(g(f(x)))`, o gradiente é:

```
∂L/∂x = (∂L/∂g) × (∂g/∂f) × (∂f/∂x)
```

Na rede neural, a composição é:

```
L = Loss(ŷ, y_target)
ŷ = W₂ · h + b₂
h = ReLU(W₁ · x + b₁)
```

### Derivação Passo a Passo

#### Passo 1: Gradiente da Saída (Erro de Saída)

```
∂L/∂ŷ = ŷ - y_target = outputError
```

(Para MSE: `∂L/∂ŷ = 2/n × (ŷ - y_target)`, simplificado para `ŷ - y_target`)

#### Passo 2: Gradientes da Camada 2 (Oculta → Saída)

```
∂L/∂W₂ = outputError ⊗ hᵀ       (produto externo)
∂L/∂b₂ = outputError
```

Onde `⊗` denota o **produto externo** (*outer product*): para vetores `u` (dim m) e `v` (dim n), `u ⊗ v` produz uma matriz m×n.

#### Passo 3: Propagação do Erro para a Camada Oculta

```
hiddenError = W₂ᵀ · outputError    (transposta de W₂ × erro da saída)
```

#### Passo 4: Gradiente através da ReLU

```
hiddenDelta = hiddenError ⊙ ReLU'(z₁)    (multiplicação elemento a elemento)
```

Onde `⊙` denota multiplicação *element-wise* (*Hadamard product*).

#### Passo 5: Gradientes da Camada 1 (Entrada → Oculta)

```
∂L/∂W₁ = hiddenDelta ⊗ xᵀ       (produto externo)
∂L/∂b₁ = hiddenDelta
```

### Diagrama do Fluxo de Gradientes

```
              FORWARD →                          ← BACKWARD
    
    x ──→ [W₁·x+b₁] ──→ [ReLU] ──→ [W₂·h+b₂] ──→ ŷ ──→ L
    │          │              │           │           │      │
    │      z₁ salvo       h salvo    z₂ = ŷ         │   ∂L/∂ŷ
    │          │              │           │           │      │
    └──── ∂L/∂W₁ ←── ∂L/∂h ←── ∂L/∂z₁ ←── ∂L/∂W₂ ←──────┘
              │              │           │
           dW₁, dB₁    hiddenDelta    dW₂, dB₂
```

### Implementação em Haskell

```haskell
backward :: Network -> Vector Double -> Vector Double 
         -> (Matrix Double, Vector Double, Matrix Double, Vector Double)
backward net input targetQ =
  let -- Forward pass (recalcular valores intermediários)
      (output, hidden, z1) = forward net input
      
      -- Passo 1: Erro da saída
      outputError = output `add` scale (-1.0) targetQ    -- ŷ - y_target
      
      -- Passo 2: Gradientes da camada 2
      dW2 = outer outputError hidden                      -- ∂L/∂W₂ = err ⊗ hᵀ
      dB2 = outputError                                   -- ∂L/∂b₂ = err
      
      -- Passo 3: Propagação do erro
      hiddenError = tr' (netWeights2 net) #> outputError  -- W₂ᵀ · err
      
      -- Passo 4: Gradiente através da ReLU
      reluGrad    = reluDerivative z1                     -- ReLU'(z₁)
      hiddenDelta = hiddenError * reluGrad                -- err ⊙ ReLU'(z₁)
      
      -- Passo 5: Gradientes da camada 1
      dW1 = outer hiddenDelta input                       -- ∂L/∂W₁ = delta ⊗ xᵀ
      dB1 = hiddenDelta                                   -- ∂L/∂b₁ = delta
      
  in (dW1, dB1, dW2, dB2)
```

### Tabela Resumo dos Gradientes

| Gradiente | Fórmula | Dimensão | Descrição |
|:---:|---|:---:|---|
| **dW₂** | `outer(outputError, hidden)` | 4 × 64 | Gradiente dos pesos da camada 2 |
| **dB₂** | `outputError` | 4 | Gradiente dos biases da camada 2 |
| **dW₁** | `outer(hiddenDelta, input)` | 64 × 12 | Gradiente dos pesos da camada 1 |
| **dB₁** | `hiddenDelta` | 64 | Gradiente dos biases da camada 1 |

---

## 🎯 Inicialização de Pesos: Xavier/He

### O Problema da Inicialização

A inicialização dos pesos é **crucial** para o sucesso do treinamento. Pesos mal inicializados causam:

| Problema | Causa | Efeito |
|---|---|---|
| **Vanishing gradients** | Pesos muito pequenos | Gradientes se aproximam de zero; rede não aprende |
| **Exploding gradients** | Pesos muito grandes | Gradientes explodem; rede diverge |
| **Simetria** | Todos os pesos iguais | Todos os neurônios aprendem a mesma coisa |

### Inicialização Xavier (Glorot)

A **inicialização Xavier** (Glorot & Bengio, 2010) define os pesos iniciais para manter a **variância das ativações** constante entre as camadas:

```
W ~ N(0, σ²)   onde   σ = √(2 / n_entrada)
```

- **n_entrada** = número de neurônios na camada anterior

### Aplicação no Projeto

Para a camada 1 (entrada → oculta):
```
W₁ ~ N(0, √(2/12)) = N(0, 0.408)
```

Para a camada 2 (oculta → saída):
```
W₂ ~ N(0, √(2/64)) = N(0, 0.177)
```

### Implementação em Haskell

```haskell
initNetwork :: StdGen -> Int -> Int -> Int -> (Network, StdGen)
initNetwork gen inputSize hiddenSize outputSize =
  let (gen1, gen2) = split gen
      (gen3, gen4) = split gen2
      
      -- Fator Xavier para cada camada
      xavierHidden = sqrt (2.0 / fromIntegral inputSize)    -- √(2/12) ≈ 0.408
      xavierOutput = sqrt (2.0 / fromIntegral hiddenSize)   -- √(2/64) ≈ 0.177
      
      -- Gerar matrizes aleatórias N(0,1) e escalar por Xavier
      w1 = scale xavierHidden $ fst $ randn hiddenSize inputSize gen1   -- 64×12
      b1 = konst 0.0 hiddenSize                                         -- 64 zeros
      w2 = scale xavierOutput $ fst $ randn outputSize hiddenSize gen3  -- 4×64
      b2 = konst 0.0 outputSize                                         -- 4 zeros
      
  in (Network w1 b1 w2 b2, gen4)
```

> [!NOTE]
> Os biases são inicializados com **zero**. Isso é uma convenção comum — como os pesos já quebram a simetria (são aleatórios), os biases não precisam ser aleatórios também.

---

## 📉 Gradient Descent

### O que é Gradient Descent?

O **Gradient Descent** (descida de gradiente) é o algoritmo de otimização que atualiza os parâmetros da rede para minimizar a função de perda. A ideia é "descer" na superfície da função de perda na direção de maior declive:

```
θ_novo = θ_atual - α × ∇L(θ)
```

Onde:
- **θ** — parâmetros da rede (pesos e biases)
- **α** — taxa de aprendizado (*learning rate*)
- **∇L(θ)** — gradiente da perda em relação a θ

### Intuição

Imagine que a função de perda é uma paisagem montanhosa, e os parâmetros são sua posição. O gradiente aponta para "cima" (direção de maior aumento). Movendo-se na direção **oposta** ao gradiente (negativo), você "desce" em direção ao mínimo:

```
L(θ)
 │
 │  ╲
 │   ╲        ╱
 │    ╲      ╱
 │     ╲    ╱
 │      ╲  ╱
 │       ╲╱  ← mínimo (objetivo)
 │
 └─────────────── θ
       ←── direção de atualização
           (oposta ao gradiente)
```

### Regras de Atualização

Para cada parâmetro da rede:

```
W₁ ← W₁ - α × dW₁        (pesos da camada 1)
b₁ ← b₁ - α × dB₁        (biases da camada 1)
W₂ ← W₂ - α × dW₂        (pesos da camada 2)
b₂ ← b₂ - α × dB₂        (biases da camada 2)
```

### Taxa de Aprendizado (α)

A taxa de aprendizado controla o tamanho do "passo" dado em cada atualização:

| Valor de α | Efeito | Risco |
|:---:|---|---|
| **Muito pequeno** (0.00001) | Passos minúsculos, convergência lentíssima | Pode ficar preso em mínimo local |
| **Adequado** (0.001) ✅ | Passos moderados, convergência estável | — |
| **Muito grande** (0.1) | Passos enormes, instabilidade | Divergência, perda oscila |

Neste projeto, α = 0.001 (valor padrão, adequado para a maioria das aplicações).

### Implementação em Haskell

```haskell
updateWeights :: Double -> Network -> (Matrix Double, Vector Double, Matrix Double, Vector Double) -> Network
updateWeights lr net (dW1, dB1, dW2, dB2) =
  let clipGrad v = cmap (max (-1.0) . min 1.0) v       -- clipping de vetores
      clipGradM m = cmap (max (-1.0) . min 1.0) m       -- clipping de matrizes
  in Network
    { netWeights1 = netWeights1 net `add` scale (-lr) (clipGradM dW1)  -- W₁ -= α·dW₁
    , netBias1    = netBias1 net    `add` scale (-lr) (clipGrad dB1)   -- b₁ -= α·dB₁
    , netWeights2 = netWeights2 net `add` scale (-lr) (clipGradM dW2)  -- W₂ -= α·dW₂
    , netBias2    = netBias2 net    `add` scale (-lr) (clipGrad dB2)   -- b₂ -= α·dB₂
    }
```

---

## ✂️ Gradient Clipping

### O Problema: Exploding Gradients

Em certas situações, os gradientes podem se tornar **extremamente grandes**, causando atualizações desproporcionais nos pesos, o que desestabiliza o treinamento. Esse fenômeno é chamado de **explosão de gradientes** (*exploding gradients*).

### Causas no Contexto DQN

No Deep Q-Learning, explosão de gradientes é particularmente comum porque:
1. As recompensas têm magnitude grande (+100, -100)
2. O target Q-value pode mudar drasticamente entre passos
3. A rede está constantemente perseguindo um "alvo móvel"

### Solução: Clipping

O **gradient clipping** limita cada componente do gradiente a um intervalo fixo:

```
clip(g, -c, c) = max(-c, min(c, g))
```

Neste projeto, `c = 1.0`:

```
Se gradiente = 5.7   → clip(5.7, -1, 1)  = 1.0    (cortado)
Se gradiente = -0.3  → clip(-0.3, -1, 1) = -0.3   (mantido)
Se gradiente = -12.5 → clip(-12.5, -1, 1) = -1.0  (cortado)
```

### Visualização

```
Sem clipping:              Com clipping [-1, 1]:

g  │                       g  │
 5 │    ·                  1  │────────── ·   ·   ·
   │   · ·                   │
   │  ·   ·                  │ ·   ·
 0 │·───────·──            0 │·───────·──
   │         ·               │         ·
-5 │          ·            -1│──────────·──────
   └──────────── t           └──────────── t
```

### Implementação em Haskell

```haskell
clipGrad :: Vector Double -> Vector Double
clipGrad v = cmap (max (-1.0) . min 1.0) v

clipGradM :: Matrix Double -> Matrix Double
clipGradM m = cmap (max (-1.0) . min 1.0) m
```

> [!WARNING]
> O gradient clipping é aplicado **antes** da atualização de pesos. Sem ele, uma única experiência com recompensa alta (+100 ou -100) poderia produzir gradientes enormes que destruiriam os pesos já aprendidos.

---

## 📏 Função de Perda: MSE

### Mean Squared Error

A função de perda **MSE** (*Mean Squared Error* — Erro Quadrático Médio) mede a discrepância entre os valores previstos e os valores alvo:

```
L = MSE(ŷ, y) = (1/n) × Σᵢ (ŷᵢ - yᵢ)²
```

Onde:
- **ŷ** — vetor de valores previstos (Q-values da rede)
- **y** — vetor de valores alvo (Q-targets calculados pela equação de Bellman)
- **n** — número de dimensões (4, uma por ação)

### Exemplo Numérico

```
Previsto (ŷ):  [12.5,  -3.2,   5.8,  -1.0]
Target (y):    [15.0,  -3.2,   5.0,  -1.0]
                 ↑              ↑
            erro=2.5        erro=0.8

Diferença:     [-2.5,   0.0,   0.8,   0.0]
Quadrado:      [ 6.25,  0.0,   0.64,  0.0]
MSE = (6.25 + 0.0 + 0.64 + 0.0) / 4 = 1.7225
```

### Propriedades do MSE

| Propriedade | Descrição |
|---|---|
| **Sempre ≥ 0** | Valores ao quadrado são sempre não-negativos |
| **= 0 quando perfeito** | Se ŷ = y, então MSE = 0 |
| **Penaliza erros grandes** | Erros são elevados ao quadrado → erros grandes são penalizados desproporcionalmente |
| **Diferenciável** | Necessário para backpropagation |

### Gradiente do MSE

```
∂MSE/∂ŷᵢ = (2/n) × (ŷᵢ - yᵢ)
```

Na implementação simplificada do projeto, usamos apenas `(ŷ - y)` como gradiente (sem o fator 2/n), pois ele é absorvido pela taxa de aprendizado.

### Implementação em Haskell

```haskell
networkLoss :: Vector Double -> Vector Double -> Double
networkLoss predicted target =
  let diff    = predicted `add` scale (-1.0) target     -- ŷ - y
      squared = cmap (\x -> x * x) diff                  -- (ŷ - y)²
  in sumElements squared / fromIntegral (size predicted)  -- (1/n) × Σ(ŷ-y)²
```

---

## 🔢 Fluxo Matemático Completo

### Um Ciclo Completo: Forward → Loss → Backward → Update

Dados:
- Estado `s` (vetor 12D)
- Ação executada `a = Down` (índice 1)
- Recompensa recebida `r = -1.0`
- Próximo estado `s'`
- γ = 0.99, α = 0.001

#### 1. Forward Pass (previsão atual)

```
z₁ = W₁ · s + b₁          (vetor 64D)
h  = ReLU(z₁)              (vetor 64D)
z₂ = W₂ · h + b₂           (vetor 4D)
Q_atual = z₂ = [2.5, -1.3, 0.8, 3.1]
                      ↑
                  ação executada (Down, idx=1)
```

#### 2. Forward Pass (próximo estado)

```
Q_próximo = predict(net, s') = [1.2, 4.5, -0.3, 2.8]
max(Q_próximo) = 4.5
```

#### 3. Cálculo do Target (Equação de Bellman)

```
target_Down = r + γ × max(Q_próximo)
            = -1.0 + 0.99 × 4.5
            = -1.0 + 4.455
            = 3.455

Q_target = [2.5, 3.455, 0.8, 3.1]
            ↑     ↑      ↑    ↑
         mantém  altera  mantém mantém
```

#### 4. Backward Pass (calcular gradientes)

```
outputError = Q_atual - Q_target
            = [2.5-2.5, -1.3-3.455, 0.8-0.8, 3.1-3.1]
            = [0.0, -4.755, 0.0, 0.0]

dW₂ = outer(outputError, h)          → matriz 4×64
dB₂ = outputError                     → vetor 4

hiddenError = W₂ᵀ · outputError       → vetor 64
hiddenDelta = hiddenError ⊙ ReLU'(z₁) → vetor 64

dW₁ = outer(hiddenDelta, s)           → matriz 64×12
dB₁ = hiddenDelta                      → vetor 64
```

#### 5. Gradient Clipping

```
dW₁ = clip(dW₁, -1, 1)    → cada elemento limitado a [-1, 1]
dB₁ = clip(dB₁, -1, 1)
dW₂ = clip(dW₂, -1, 1)
dB₂ = clip(dB₂, -1, 1)
```

#### 6. Atualização de Pesos (Gradient Descent)

```
W₁ ← W₁ - 0.001 × dW₁
b₁ ← b₁ - 0.001 × dB₁
W₂ ← W₂ - 0.001 × dW₂
b₂ ← b₂ - 0.001 × dB₂
```

#### 7. Perda (MSE)

```
Q_novo = predict(net_atualizado, s) = [2.5, -0.82, 0.8, 3.1]
L = MSE(Q_novo, Q_target)
  = (1/4) × [(2.5-2.5)² + (-0.82-3.455)² + (0.8-0.8)² + (3.1-3.1)²]
  = (1/4) × [0 + 18.3 + 0 + 0]
  = 4.575
```

---

## 💻 Implementação em Haskell

### Resumo das Operações com `hmatrix`

| Operação Matemática | Notação | `hmatrix` | Exemplo |
|---|:---:|---|---|
| Multiplicação matriz-vetor | W · x | `W #> x` | `netWeights1 net #> input` |
| Soma de vetores | a + b | `` a `add` b `` | `` z1 `add` netBias1 net `` |
| Escalar × vetor | c · v | `scale c v` | `scale (-lr) dB1` |
| Produto externo | u ⊗ v | `outer u v` | `outer outputError hidden` |
| Transposta | Aᵀ | `tr' A` | `tr' (netWeights2 net)` |
| Aplicar função element-wise | f(v) | `cmap f v` | `cmap (max 0) v` |
| Soma de elementos | Σ vᵢ | `sumElements v` | `sumElements squared` |
| Vetor constante | [c, c, ..., c] | `konst c n` | `konst 0.0 hiddenSize` |
| Vetor de lista | fromList | `fromList xs` | `fromList [1.0, 2.0, 3.0]` |

### Código Completo do Módulo

O módulo `NeuralNetwork.hs` contém **82 linhas** de código Haskell puro, implementando:
- Inicialização Xavier
- Forward propagation
- Backward propagation (com regra da cadeia)
- Atualização de pesos (gradient descent + clipping)
- Função de perda MSE

Todo o módulo utiliza apenas operações de álgebra linear da biblioteca `hmatrix` — sem nenhum framework de machine learning externo.

---

## 📐 Visualização dos Tensores

### Dimensões em Cada Etapa

```
┌────────────────────────────────────────────────────────────────┐
│  FORWARD PROPAGATION                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  input    W₁         b₁       z₁       h        W₂       b₂       output │
│  (12)  × (64×12) + (64)  → (64) → ReLU → (64) × (4×64) + (4) → (4) │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  BACKWARD PROPAGATION                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  outputError    dW₂         dB₂      hiddenError    dW₁         dB₁    │
│  (4)         → (4×64)    → (4)    ← (64)        → (64×12)   → (64)  │
│                                                                │
│  Direção: saída ←────────────────────────────────── entrada    │
└────────────────────────────────────────────────────────────────┘
```

### Contagem de Operações

| Operação | Operação Matemática | FLOPs |
|---|---|:---:|
| z₁ = W₁·x + b₁ | Matriz 64×12 × Vetor 12 | 64 × 12 × 2 = 1.536 |
| h = ReLU(z₁) | Comparação element-wise | 64 |
| z₂ = W₂·h + b₂ | Matriz 4×64 × Vetor 64 | 4 × 64 × 2 = 512 |
| **Forward Total** | | **~2.112** |
| outputError | Subtração de vetores | 4 |
| dW₂ = outer(err, h) | Produto externo 4×64 | 256 |
| hiddenError = W₂ᵀ·err | Matriz 64×4 × Vetor 4 | 512 |
| hiddenDelta | Element-wise × | 64 |
| dW₁ = outer(delta, x) | Produto externo 64×12 | 768 |
| **Backward Total** | | **~1.604** |
| **Forward + Backward** | | **~3.716** |

Para um mini-batch de 32 experiências: ~119.000 FLOPs por atualização.

---

<div align="center">

*Documentação técnica da rede neural para o projeto Dungeon AI — Deep Q-Learning em Haskell*

[📐 Arquitetura](architecture.md) | [🧠 DQN](deep-q-learning.md) | 🔮 Rede Neural | [🎓 Apresentação](presentation.md)

</div>
