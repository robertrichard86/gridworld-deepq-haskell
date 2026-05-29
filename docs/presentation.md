# 🎓 Guia de Apresentação Acadêmica — Dungeon AI

> **Roteiro e Diretrizes para Apresentação em Sala de Aula**
> Projeto: gridworld-deepq-haskell | Duração Sugerida: 15 a 20 minutos

---

## 📋 Índice

- [Estrutura do Tempo Sugerida](#-estrutura-do-tempo-sugerida)
- [Roteiro de Slides Sugerido](#-roteiro-de-slides-sugerido)
  - [Slide 1: Capa & Introdução](#slide-1-capa--introdução)
  - [Slide 2: Contextualização e Objetivo](#slide-2-contextualização-e-objetivo)
  - [Slide 3: O Paradigma de Aprendizado por Reforço](#slide-3-o-paradigma-de-aprendizado-por-reforço)
  - [Slide 4: Por que Deep Q-Learning (DQN)?](#slide-4-por-que-deep-q-learning-dqn)
  - [Slide 5: Por que Haskell para Inteligência Artificial?](#slide-5-por-que-haskell-para-inteligência-artificial)
  - [Slide 6: O Ambiente "Dungeon" (Grid World)](#slide-6-o-ambiente-dungeon-grid-world)
  - [Slide 7: Arquitetura da Rede Neural (DQN)](#slide-7-arquitetura-da-rede-neural-dqn)
  - [Slide 8: O Algoritmo de Treinamento](#slide-8-o-algoritmo-de-treinamento)
  - [Slide 9: Arquitetura Modular em Haskell](#slide-9-arquitetura-modular-em-haskell)
  - [Slide 10: Demonstração e Resultados](#slide-10-demonstração-e-resultados)
  - [Slide 11: Conclusões e Trabalhos Futuros](#slide-11-conclusões-e-trabalhos-futuros)
- [Roteiro de Demonstração ao Vivo](#-roteiro-de-demonstração-ao-vivo)
- [Preparação para Perguntas e Respostas (Q&A)](#-preparação-para-perguntas-e-respostas-qa)

---

## ⏱️ Estrutura do Tempo Sugerida

Para uma apresentação padrão de **15 a 20 minutos**, a seguinte distribuição de tempo é altamente recomendada para garantir que todos os pontos importantes sejam cobertos sem pressa:

```
┌────────────────────────────────────────────────────────────────────────┐
│  DISTRIBUIÇÃO DE TEMPO (20 minutos)                                     │
├────────────────────────────────────────────────────────────────────────┤
│  [00:00 - 02:00]  Abertura, Motivação e Objetivos (Slides 1-2)         │
│  [02:00 - 05:00]  Fundamentação Teórica: RL e DQN (Slides 3-4)          │
│  [05:00 - 07:00]  Escolha de Haskell & O Ambiente Dungeon (Slides 5-6) │
│  [07:00 - 11:00]  Arquitetura da Rede e do DQN (Slides 7-8)            │
│  [11:00 - 13:00]  Arquitetura de Software e Código (Slide 9)           │
│  [13:00 - 17:00]  Demonstração ao Vivo e Resultados (Slide 10)         │
│  [17:00 - 20:00]  Conclusões e Perguntas & Respostas (Slide 11 + Q&A)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Roteiro de Slides Sugerido

Abaixo está o detalhamento de cada slide, incluindo sugestão de conteúdo visual, palavras-chave e as **Notas do Orador** (*Speaker Notes*) para guiar a sua fala.

### Slide 1: Capa & Introdução

*   **Título:** 🏰 Dungeon AI: Deep Q-Learning do Zero em Haskell
*   **Subtítulo:** Aprendizado por Reforço Profundo aplicado a um Grid World Textual com Redes Neurais Manuais e Sem Efeitos Colaterais
*   **Apresentadores:** [Seu Nome] & Robert Richard (Co-autor)
*   **Identidade Visual:** Fundo escuro com cores contrastantes (roxo, azul e verde-água).
*   **Elementos Visuais:** Emojis ou ícones de fantasia (🧙, 💎, 🔥) e logo da linguagem Haskell.

> 🎤 **Notas do Orador:**
> *"Olá a todos. Hoje vou apresentar o projeto Dungeon AI, um trabalho acadêmico focado na implementação prática e conceitual de Deep Q-Learning (DQN) utilizando a linguagem funcional puramente tipada Haskell. O diferencial deste projeto é que não utilizamos nenhum framework de Machine Learning como PyTorch ou TensorFlow. Toda a álgebra linear e a retropropagação da rede neural foram implementadas manualmente, combinando a matemática rigorosa de redes neurais com as garantias de corretude e tipagem forte do paradigma funcional."*

---

### Slide 2: Contextualização e Objetivo

*   **Título:** 🎯 O Desafio: Grid World "Dungeon"
*   **Pontos de Tópico:**
    *   Criar um agente autônomo (🧙 Mago) capaz de navegar em um calabouço 5x5.
    *   Objetivo final: Encontrar o tesouro lendário (💎 Diamante) em (4,4).
    *   Obstáculos: Sobreviver a armadilhas de fogo (🔥 Labas) e paredes (limites).
    *   Abordagem acadêmica: Superar as limitações da Q-Learning tabular usando aproximação de função com Rede Neural.
*   **Elementos Visuais:** Tabela comparativa ou representação simples do Grid World textual.

> 🎤 **Notas do Orador:**
> *"Para testar nosso agente, criamos um ambiente Grid World clássico, mas com uma roupagem de fantasia medieval — o 'Dungeon AI'. Nele, o nosso mago começa no canto superior esquerdo e precisa chegar até o diamante no canto inferior direito. O ambiente impõe severas punições para colisões e armadilhas. Embora um ambiente 5x5 pareça simples para abordagens tabulares clássicas, nosso objetivo aqui é demonstrar a viabilidade e a convergência de uma abordagem profunda e genérica, que pavimenta o caminho para a resolução de ambientes com espaços de estados contínuos e complexos."*

---

### Slide 3: O Paradigma de Aprendizado por Reforço

*   **Título:** 🗺️ Aprendizado por Reforço (RL)
*   **Pontos de Tópico:**
    *   Aprendizado baseado em tentativa e erro através da interação com o ambiente.
    *   **Agente:** O tomador de decisões (🧙 Mago).
    *   **Ambiente:** O calabouço, que reage às ações.
    *   **Estado ($s$):** A representação codificada de onde o agente está e o que o cerca.
    *   **Ação ($a$):** Movimentos válidos (Cima, Baixo, Esquerda, Direita).
    *   **Recompensa ($r$):** Feedback imediato (+100 no Tesouro, -100 no Fogo, -1 por passo).
*   **Elementos Visuais:** O diagrama clássico de ciclo Agente-Ambiente (Ação $\to$ Estado, Recompensa).

> 🎤 **Notas do Orador:**
> *"Antes de entrarmos nos detalhes da rede neural, vale a pena lembrar como funciona o Aprendizado por Reforço. Ao contrário do Aprendizado Supervisionado, onde temos rótulos corretos para cada entrada, o agente de RL aprende puramente através de feedback. Ele toma uma ação, o ambiente transiciona para um novo estado e fornece uma recompensa ou punição. O objetivo do agente é maximizar o retorno acumulado ao longo do tempo. No nosso calabouço, ficar dando voltas ou bater nas paredes custa energia, cair em uma armadilha é catastróde, e achar o diamante é a glória divina."*

---

### Slide 4: Por que Deep Q-Learning (DQN)?

*   **Título:** 🧠 De Q-Learning a Deep Q-Learning
*   **Pontos de Tópico:**
    *   **Limitação do Q-Learning Clássico:** A Q-Table mapeia explicitamente $(s, a)$. Escala pessimamente para problemas com muitos estados (Maldição da Dimensionalidade).
    *   **Solução DQN:** Substituir a tabela por um aproximador de função universal — uma **Rede Neural Artificial**.
    *   A rede recebe o vetor de estado $s$ e prevê os $Q$-values para todas as ações possíveis simultaneamente.
    *   **Fórmula da Perda (Bellman):**
        $$L = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
*   **Elementos Visuais:** Um desenho esquemático comparando uma tabela física (Q-Table) com uma rede neural (DQN) que recebe o estado e cospe 4 saídas.

> 🎤 **Notas do Orador:**
> *"Em problemas reais de robótica ou jogos modernos, o número de estados possíveis é infinito ou astronomicamente grande. Guardar um valor de utilidade para cada par de estado e ação em uma tabela torna-se impossível. É aí que entra o Deep Q-Learning, proposto pela DeepMind em 2013. Usamos uma rede neural como aproximador de parâmetros. Em vez de registrar 'o valor exato de estar na célula (3,2)', a rede aprende a generalizar padrões geográficos do grid, de forma que estados nunca antes vistos possam ter ações inteligentemente preditas através das características compartilhadas."*

---

### Slide 5: Por que Haskell para Inteligência Artificial?

*   **Título:** 🧙 Por que Haskell? Quebrando Mitos
*   **Pontos de Tópico:**
    *   **Ausência de Efeitos Colaterais:** O treinamento é uma transformação de dados pura (`trainDQN :: DQNConfig -> StdGen -> (Network, [TrainingStats])`).
    *   **Tipagem Estática Forte:** Tipos como `Action`, `Experience` e `Network` impedem erros comuns em tempo de execução (bugs de dimensões matriciais).
    *   **Concorrência e Paralelismo:** Facilidade nativa para escalar processamento de dados de treinamento.
    *   **Prevenção de Space Leaks:** Controle fino de avaliação estrita com `NFData` e `StrictData` para impedir vazamento de memória com thunks acumulados.
    *   **Performance Matemática:** Utilização da biblioteca `hmatrix` integrada com as bibliotecas C otimizadas LAPACK e BLAS.
*   **Elementos Visuais:** Logotipo do Haskell contrastado com fórmulas de tipos puras.

> 🎤 **Notas do Orador:**
> *"Historicamente, Python domina a Inteligência Artificial devido à sua rica biblioteca. No entanto, usar Haskell para IA nos traz benefícios fantásticos de engenharia de software. Toda a lógica do agente e do calabouço é garantida matematicamente em tempo de compilação. Em Haskell, a aleatoriedade e os pesos da rede não são variáveis globais mutáveis; eles fluem explicitamente como dados de entrada e saída. Para evitar vazamentos de memória (os famosos space leaks causados pela avaliação preguiçosa do Haskell), estruturamos nossos records de forma estrita e implementamos avaliação profunda no Buffer de Experiências."*

---

### Slide 6: O Ambiente "Dungeon" (Grid World)

*   **Título:** 🗺️ Modelagem do Ambiente & Vetor de Estado
*   **Pontos de Tópico:**
    *   **Representação em ASCII:** Tabuleiro 5x5 renderizado com emojis e caracteres de caixa dupla.
    *   **Vetor de Estado (12 Dimensões):**
        *   Posição do Agente Normalizada: $(x_{ag}, y_{ag})$ (2D)
        *   Posição do Objetivo Normalizada: $(x_{obj}, y_{obj})$ (2D)
        *   Vetor de Distância Agente-Objetivo: $(\Delta x, \Delta y)$ (2D)
        *   Distâncias Relativas a 3 Obstáculos Estáticos: $(\Delta x_i, \Delta y_i)$ (6D)
    *   Todas as entradas normalizadas entre $-1.0$ e $1.0$ para estabilizar o aprendizado.
*   **Elementos Visuais:** Tabela mostrando a especificação exata das 12 dimensões do vetor de entrada.

> 🎤 **Notas do Orador:**
> *"Como nosso calabouço é representado para a nossa rede neural? Não podemos passar um grid visual bruto facilmente. Codificamos as informações cruciais em um vetor de 12 dimensões. Ele contém a coordenada normalizada do mago, do diamante, a distância vetorial relativa entre eles, e as coordenadas relativas das três poças de lava espalhadas pelo calabouço. Normalizar esses valores é fundamental: redes neurais operam de forma extremamente instável quando recebem entradas de magnitudes muito discrepantes. Com tudo na escala de menos um a um, a convergência é muito mais suave."*

---

### Slide 7: Arquitetura da Rede Neural (DQN)

*   **Título:** 🔮 Rede Neural Artificial Manual
*   **Pontos de Tópico:**
    *   **Tipo de Rede:** Perceptron Multicamadas (MLP) Feedforward.
    *   **Arquitetura:** 12 neurônios (Entrada) $\to$ 64 neurônios (Camada Oculta) $\to$ 4 neurônios (Saída).
    *   **Ativação Oculta:** **ReLU** ($f(x) = \max(0, x)$).
    *   **Saída:** Ativação linear (valores reais de $Q$ para Cima, Baixo, Esquerda, Direita).
    *   **Inicialização:** Inicialização Xavier/He para evitar sumiço/explosão de gradientes.
    *   **Otimização:** Descida de Gradiente Estocástica (SGD) com **Gradient Clipping** em $[-1.0, 1.0]$.
*   **Elementos Visuais:** Diagrama da arquitetura MLP (12 $\to$ 64 $\to$ 4) detalhando as matrizes $W_1$ ($64 \times 12$) e $W_2$ ($4 \times 64$).

> 🎤 **Notas do Orador:**
> *"A rede neural que aproxima nossa utilidade Q-value é uma MLP enxuta. Seus doze neurônios de entrada passam por uma matriz de pesos de 64 por 12, somados a um bias de 64 dimensões, sofrendo ativação pela função retificadora ReLU. A saída final passa por outra projeção linear gerando quatro valores numéricos de Q. A ação com o maior Q-value predito é a nossa melhor escolha matemática. Para treinar isso manualmente, implementamos a regra da cadeia no cálculo de derivadas da ReLU e no produto externo matricial. O Gradient Clipping foi a chave para o sucesso do treinamento: ele impede que erros catastróficos nas poças de fogo explodam as derivadas e destruam a memória da rede."*

---

### Slide 8: O Algoritmo de Treinamento

*   **Título:** 🔁 Mecanismos de Estabilização do DQN
*   **Pontos de Tópico:**
    *   **Epsilon-Greedy ($\epsilon$-greedy):**
        *   Taxa de Exploração $\epsilon$ começa em $1.0$ (ações 100% aleatórias)
        *   Decai a cada episódio com fator $0.995$ até o mínimo de $0.01$ (99% de aproveitamento da rede)
    *   **Experience Replay Buffer:**
        *   Grava transições $(s, a, r, s', done)$ em um buffer de até $10.000$ itens.
        *   Amostra aleatoriamente mini-batches de tamanho $32$ para realizar as atualizações.
        *   **Objetivo:** Quebrar a correlação temporal entre passos sucessivos e reaproveitar experiências raras.
*   **Elementos Visuais:** Gráfico conceitual ilustrando o decaimento do $\epsilon$ ao longo dos episódios e o funcionamento lógico do Replay Buffer.

> 🎤 **Notas do Orador:**
> *"Treinar redes neurais com dados puramente sequenciais de RL é notoriamente instável porque dados adjacentes são extremamente parecidos e correlacionados temporariamente. Para contornar isso, usamos duas técnicas consolidadas. O Epsilon-greedy garante que no início do treino o mago explore loucamente o calabouço para mapear os perigos. Conforme o tempo passa, ele gradualmente confia mais na rede neural. Além disso, criamos um buffer de replay. Em vez de treinar a rede apenas com o passo atual, salvamos as experiências na memória e periodicamente sorteamos um lote de 32 experiências passadas aleatórias. Isso remove a correlação sequencial e estabiliza o treinamento."*

---

### Slide 9: Arquitetura Modular em Haskell

*   **Título:** 📐 Organização do Código Haskell
*   **Pontos de Tópico:**
    *   **`Types.hs`:** Estruturas de dados estritas com deriving `Generic` e instâncias `NFData`.
    *   **`Environment.hs`:** Definição física do Grid 5x5 e cálculo puro de recompensas.
    *   **`NeuralNetwork.hs`:** Forward, backward pass, Xavier init, gradientes e MSE Loss.
    *   **`Agent.hs`:** Seleção $\epsilon$-greedy e decaimento exponencial de exploração.
    *   **`ReplayBuffer.hs`:** Lógica LIFO com amostragem aleatória pura via gerador de semente (`StdGen`).
    *   **`DQN.hs`:** O loop orquestrador que une todos os componentes via `foldl'` estrito.
    *   **`Render.hs` & `Main.hs`:** Apresentação ASCII elegante e visualização de caminho aprendido.
*   **Elementos Visuais:** Grafo de dependências entre os módulos (conforme diagramado na especificação técnica).

> 🎤 **Notas do Orador:**
> *"Nosso projeto preza pela extrema organização e clareza. Não há uma única linha de comentário nos arquivos Haskell, seguindo uma restrição de design rígida do projeto. Toda a clareza decorre do nome das funções, expressividade dos tipos e pureza conceitual. O coração do sistema reside no arquivo DQN.hs, onde o loop principal de 500 episódios é orquestrado através do foldl' estrito da biblioteca Data.List, garantindo que nenhum thunk lazily acumulado cause lentidão."*

---

### Slide 10: Demonstração e Resultados

*   **Título:** 🚀 Resultados & Demonstração Visual
*   **Pontos de Tópico:**
    *   **Duração do Treino:** 500 episódios (geralmente executados em menos de 10 segundos).
    *   **Evolução da Perda (Loss):** Redução consistente conforme a rede converge para os Q-values ideais.
    *   **Média Móvel de Recompensas:** Transição visível de recompensas muito negativas (episódios iniciais) para uma média positiva próxima a $+90$ (sucesso).
    *   **Caminho Aprendido:** Traçado final livre de armadilhas até o diamante!
*   **Elementos Visuais:** Demonstração em tempo real ou gravação da tela exibindo a barra de progresso ASCII e o mapa final com o caminho ideal verde marcado.

> 🎤 **Notas do Orador:**
> *"Agora vamos à parte mais empolgante: os resultados. Ao rodarmos o programa, vemos uma barra de progresso e estatísticas a cada 25 episódios. Inicialmente, o mago cai no fogo a todo momento, acumulando recompensas de menos cem. Mas, por volta do episódio 150 a 200, a curva de recompensa dá um salto dramático: a rede neural aprendeu a desviar das três lavas e a traçar uma rota curta e limpa até o diamante. No final, o programa exibe o caminho aprendido de forma visual, provando que o agente aprendeu de verdade!"*

---

### Slide 11: Conclusões e Trabalhos Futuros

*   **Título:** 🏁 Conclusões Acadêmicas e Próximos Passos
*   **Pontos de Tópico:**
    *   **Conclusão Principal:** Haskell provou-se extremamente robusto, rápido e seguro para implementar IA profunda do zero sem frameworks.
    *   **Matemática Pura:** A implementação manual do algoritmo DQN reforça conceitos cruciais de retropropagação e Bellman que costumam ficar ocultos em bibliotecas comerciais.
    *   **Trabalhos Futuros:**
        *   Implementação de DQN com Duas Redes (Double DQN) para evitar superestimação de Q-values.
        *   Ambientes maiores e dinâmicos (onde os obstáculos mudam de lugar).
        *   Integração com visualizações gráficas em tempo real usando frameworks como Gloss ou SDL2.
*   **Elementos Visuais:** Link para o repositório GitHub e agradecimento.

> 🎤 **Notas do Orador:**
> *"Em conclusão, este projeto demonstra que a elegância do paradigma funcional em Haskell combina perfeitamente com o rigor matemático de Machine Learning. A criação manual deste algoritmo nos permitiu entender cada engrenagem interna do Deep Q-Learning — desde como a transposta de uma matriz flui o gradiente até o impacto de pequenas mudanças no fator de desconto e na taxa de aprendizado. Para trabalhos futuros, planejamos experimentar o Double DQN para reduzir a clássica superestimação de valores de utilidade e testar o agente em calabouços dinâmicos. Agradecemos a atenção e abrimos espaço para perguntas!"*

---

## 🏃 Roteiro de Demonstração ao Vivo

Para impressionar o público, faça a demonstração do console. Siga este roteiro detalhado:

### 1. Preparação do Console
*   Certifique-se de que o terminal tem suporte a caracteres Unicode UTF-8 (em ambientes Unix/macOS isso é nativo; no Windows Powershell, o executável executa automaticamente `hSetEncoding stdout utf8` para garantir compatibilidade).
*   Use uma fonte de console com bom suporte a Emojis (como Consolas, Segoe UI Symbol ou Fira Code).

### 2. Comandos de Execução
*   **Passo 1 — Explicar o build rápido (se estiver rodando ao vivo):**
    ```bash
    stack build
    ```
    *(Diga que a compilação vincula as bibliotecas blas/lapack de alta performance).*
*   **Passo 2 — Rodar o executável principal:**
    ```bash
    stack exec gridworld-deepq
    ```

### 3. O que apontar durante a execução:
1.  **Cabeçalho e Configuração:** Chame a atenção para a impressão do calabouço inicial contendo o Mago `🧙` no canto superior esquerdo, o Diamante `💎` no canto inferior direito e as Lavas `🔥` distribuídas nas posições estratégicas.
2.  **A Barra de Progresso:** Destaque o indicador visual ASCII conforme os episódios avançam rapidamente de 1 a 500.
3.  **A Transição de Comportamento:**
    *   **Episódios 1-10:** Mostre que as recompensas totais são muito negativas (por exemplo, `-150.0` ou `-200.0`), e a perda (loss) da rede varia. A taxa de acerto no diamante é praticamente nula ($0\%$).
    *   **Episódios 100-200:** Aponte que a perda começa a diminuir de forma consistente e a média móvel de recompensa começa a subir para valores positivos, indicando os primeiros episódios de sucesso.
    *   **Episódios 400-500:** A taxa de sucesso chega a quase $100\%$ e o agente resolve o grid com o menor número possível de passos (geralmente $8$ passos).
4.  **O Caminho Final:** Comemore a exibição do calabouço final onde a trilha de bolinhas verdes `🟢` mostra a rota perfeita desenhada pelo agente, desviando milimetricamente de todas as lavas `🔥`.

---

## ❓ Preparação para Perguntas e Respostas (Q&A)

Aqui estão as perguntas mais difíceis que professores ou colegas exigentes podem fazer, acompanhadas das respostas mais adequadas baseadas na implementação:

### P1: Por que vocês implementaram a rede neural do zero em vez de usar uma biblioteca como `Grenade` ou `TensorFlow` em Haskell?
> **Resposta:**
> *"A decisão foi puramente educacional e científica. Bibliotecas de Machine Learning de alto nível escondem os detalhes do cálculo diferencial. Ao programarmos o forward pass, backward pass, a derivada da ReLU, a inicialização Xavier e a descida de gradiente estocástica manualmente na mão (usando apenas a `hmatrix` para nos dar suporte de tensores e matrizes), fomos obrigados a compreender perfeitamente a fluxo matemático dos gradientes e a mecânica das atualizações de parâmetros. Além disso, isso demonstra o quão expressiva a linguagem Haskell é, contendo toda a lógica neural em meras 80 linhas de código limpo."*

### P2: Como vocês trataram o problem dos Space Leaks (vazamentos de espaço), que são extremamente comuns em Haskell quando lidamos com loops de treinamento repetitivos?
> **Resposta:**
> *"Esse foi um dos nossos maiores desafios técnicos. Loops de treinamento e buffer de experiências armazenam milhares de dados que, sob a avaliação preguiçosa (lazy evaluation) do Haskell, geram 'thunks' na memória (promessas de cálculo não resolvidas). Se não tratássemos isso, o programa sofreria um estouro de memória após algumas centenas de episódios. Resolvemos isso de três formas estruturadas:
> 1. Ativamos a extensão `StrictData` na compilação, tornando todos os campos de records estritos por padrão.
> 2. Implementamos a instância da classe `NFData` (Normal Form Data) para a nossa `Network` e `Experience` usando o pacote `deepseq`. Isso nos permite realizar avaliação profunda estrita em cada passo.
> 3. Utilizamos `foldl'` estrito da biblioteca `Data.List` para a execução dos episódios e das iterações de lotes, forçando o GHC a resolver as contas e liberar a memória instantaneamente."*

### P3: Por que o vetor de estado tem exatamente 12 dimensões? Uma representação matricial 5x5 do grid não seria melhor?
> **Resposta:**
> *"Uma matriz 5x5 resultaria em 25 entradas. Embora seja viável, para uma rede enxuta (hidden size de 64), o excesso de dimensões esparsas (cheias de zeros) tornaria o aprendizado desnecessariamente demorado. Ao criarmos o vetor de 12 dimensões, fizemos engenharia de features (Feature Engineering): extraímos as posições exatas do agente e do objetivo, as distâncias relativas entre eles, e a distância relativa a cada um dos três perigos. Essas características de relevância direta agilizam massivamente o aprendizado de desvio de armadilhas. Além disso, todas as entradas foram normalizadas para a escala aproximada de $[-1.0, 1.0]$, garantindo a estabilidade matemática no forward pass."*

### P4: O Experience Replay realmente fez diferença? O que aconteceria se treinássemos a rede puramente com o passo atual?
> **Resposta:**
> *"Fizemos esse teste empírico durante o desenvolvimento. Sem o Replay Buffer, a rede neural tenta aprender baseando-se apenas no passo que acabou de dar. Como passos adjacentes são extremamente parecidos, a rede se hiper-ajusta (overfitting) à sua vizinhança imediata e entra em oscilações catastróficas, esquecendo o que aprendeu em episódios anteriores. O Experience Replay age como uma 'memória fotográfica coletiva'. Ao gravarmos as transições e sortearmos lotes aleatórios de 32 itens de momentos temporais distintos, quebramos a forte correlação de dados e permitimos que a rede continue aprendendo a desviar do fogo mesmo quando ela está bem longe dele."*

### P5: Como a aleatoriedade (geração de números randômicos) é tratada de forma pura, já que o Haskell não permite variáveis globais de estado aleatório?
> **Resposta:**
> *"Em linguagens imperativas como Python, funções de randomização alteram um estado global oculto. Em Haskell, isso violaria a pureza funcional. Nossa solução foi gerenciar o gerador aleatório de forma puramente funcional: passamos o gerador `StdGen` explicitamente para funções como `selectAction`, `sampleBatch` e `runEpisode`, e essas funções sempre retornam uma nova semente geradora atualizada (`StdGen'`) além de seu resultado padrão. Isso garante que a nossa base de código permaneça 100% livre de efeitos colaterais impuros, permitindo reprodutibilidade perfeita do treino bastando usar a mesma semente inicial."*

---

<div align="center">

*Guia de apresentação do projeto Dungeon AI — Deep Q-Learning em Haskell*

[📐 Arquitetura](architecture.md) | [🧠 DQN](deep-q-learning.md) | [🔮 Rede Neural](neural-network.md) | 🎓 Apresentação

</div>
