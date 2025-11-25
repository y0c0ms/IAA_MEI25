# Introdução à Aprendizagem Automática — 2025/2026
## Aprendizagem por Reforço

Estes exercícios devem ser resolvidos utilizando **Python notebooks** devido à possibilidade de gerar um relatório integrado com o código. É assumido que o estudante é proficiente em programação. Todas as respostas devem ser **justificadas** e os resultados **discutidos** e **comparados** com baselines apropriados.

A pontuação máxima da tarefa é **2 pontos**. Os exercícios opcionais (se existentes) ajudam a alcançar a pontuação máxima, complementando erros ou falhas.

**Deadline:** final da aula da semana de **24 a 28 de novembro de 2025**.

---

## Figura 1 — Ambiente simplificado

O cenário de aprendizagem é uma sala representada por uma grelha **10 × 10**, com os estados numerados sequencialmente da seguinte forma:

- Primeira linha: `1 2 3 ... 10`  
- Segunda linha: `11 12 ... 20`  
- Terceira linha: `21 ... 30`  
- …  
- Última linha: `91 ... 100`

Os **quadrados azuis escuros** da figura representam **paredes**, isto é, posições que não podem ser ocupadas pelo robot. O estado inicial típico está algures no interior da grelha (representado pelo *Smiley*) e o objetivo é o **estado 100**, identificado com um **X**.

> No PDF original, esta configuração é ilustrada graficamente na **Figura 1: Ambiente simplificado (os estados são numerados de 1 a 100). Os quadrados azuis escuros são paredes.**

---

## Contexto Geral

Um robot (*Smiley*) precisa de aprender a sequência de ações que o leva do **estado 1** até ao **estado 100**, onde existe uma tomada elétrica (“X”). O ambiente é a grelha 10×10 descrita na Figura 1, numerada de 1 a 100, com algumas células ocupadas por paredes.

O robot possui quatro ações possíveis:

- **UP (cima)**
- **DOWN (baixo)**
- **LEFT (esquerda)**
- **RIGHT (direita)**

Um movimento leva o robot do centro de um quadrado para o quadrado adjacente correspondente. Se a ação apontar para fora da grelha ou contra uma parede, o robot **permanece no mesmo estado**.

Ao chegar ao **estado 100**, o robot recebe uma **recompensa positiva**.

---

# Exercício 1 — Construção do Ambiente

### a) Função de transição

Criar uma função

```text
s' = f(s, a)
```

que devolve o novo estado após aplicar a ação `a` no estado `s`.

Exemplos:

- `f(1, right) = 2`
- `f(1, down) = 11`
- `f(1, up) = 1`  (porque existe parede/limite para cima)
- A mesma lógica aplica-se a todos os estados e ações.

---

### b) Função de recompensa

Criar uma função de recompensa `r(s)` tal que:

- `r(s) = 0` para todos os estados exceto o objetivo,
- `r(100) = 100`.

---

### c) Função de ação aleatória

Criar uma função que escolhe, de forma equiprovável, uma das quatro ações possíveis:

- `UP`, `DOWN`, `LEFT`, `RIGHT`.

---

### d) Definição de fim de episódio

Um episódio termina quando:

1. O robot atinge o **estado 100** (tomada elétrica, marcada com X):  
   - recebe a recompensa `r(100) = 100`;  
   - é reiniciado para o **estado inicial**.

2. O robot executa **1000 ações sem atingir o objetivo**:  
   - não recebe recompensa adicional;  
   - é reiniciado para o estado inicial.

---

### e) Simulação

Simular o robot a realizar **30 episódios**. Para cada episódio deve ser medido e registado:

- a **recompensa média por passo**;
- o **número de passos até atingir o objetivo**.

No final, para os 30 episódios, calcular:

- **média** e **desvio padrão** do número de passos para atingir o objetivo;
- **média** e **desvio padrão** da recompensa média por passo;
- **tempos médios de execução** e respetivos desvios padrão.

Estes valores constituem os **baselines** do sistema “puro acaso”, que servirão de referência para comparação com métodos de aprendizagem.

---

### f) Representação gráfica — Figura 2

Representar, cada um em separado, num **boxplot com caixas verticais**:

1. Recompensa média por passo;
2. Número médio de passos para atingir o objetivo;
3. Tempo médio de execução.

> No PDF original, isto corresponde à **Figura 2: Boxplots mostrando a recompensa média por passo, o número médio de passos para atingir o objetivo e o tempo médio de execução para os 30 testes (máximo de 1000 passos cada).**

---

## Notas sobre processos estocásticos

Sempre que existir aleatoriedade (ou pseudo-aleatoriedade), deve-se:

1. **Guardar a semente aleatória**, para poder repetir a mesma experiência.
2. **Repetir cada experiência 30 vezes**, com os mesmos parâmetros.
3. Calcular **média** e **desvio padrão** para todas as quantidades medidas.
4. Utilizar representações gráficas adequadas (por exemplo, `boxplot` em Python/Matplotlib).

---

# Exercício 2 — Q-Learning

Criar uma matriz:

```text
Q[s, a]
```

indexada por estado `s` e ação `a`, inicializada com zeros.

Sempre que o robot faz uma transição de `s` para `s'` aplicando a ação `a`, atualizar Q de acordo com:

\[
Q[s,a] = (1 - lpha)Q[s,a] + lpha\Big( r(s') + \gamma \max_{a'} Q[s', a'] \Big)
\]

onde:

- \(\max_{a'} Q[s', a']\) é a melhor utilidade estimada em `s'` para todas as ações possíveis;
- `r(s')` é a recompensa obtida no novo estado.

Valores a usar:

- **α = 0.7**
- **γ = 0.99**

O objetivo é que, no final, se consiga identificar **a melhor ação para qualquer estado** a partir da tabela Q.

---

## 2.a) Treino com Random Walk

Executar o seguinte procedimento para **30 experiências independentes**:

1. Iniciar Q a zeros.
2. Fazer uma **caminhada aleatória (random walk)** de **20000 passos**, em que:
   - a ação escolhida em cada passo é aleatória;
   - depois de cada transição, Q é atualizado com a equação acima.

3. Nos seguintes passos intermédios:

```text
100, 200, 500, 600, 700, 800, 900, 1000,
2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000
```

parar temporariamente o treino e executar um **teste**:

- O teste consiste em correr o sistema durante **1000 passos**,  
  - usando a tabela Q atual,  
  - escolhendo **sempre a melhor ação** em cada estado (`argmax_a Q[s,a]`),  
  - **sem alterar** a tabela Q durante o teste.
- Medir a **recompensa média por passo** nestes 1000 passos.

4. Medir e registar também:
   - o **tempo total de execução** dos 20000 passos de treino;

5. Repetir este procedimento completo **30 vezes** e calcular:
   - média e desvio padrão dos tempos de execução;
   - evolução da recompensa média nos pontos de teste.

### Representações gráficas pedidas

- Traçar um gráfico **“número de passos de treino” (eixo x) vs “recompensa média por passo no teste” (eixo y)**.
- Produzir, se desejado, uma **série de diagramas de caixas (boxplots)** para mostrar a evolução do comportamento do robot.

### Figura 3 — Mapa de calor da utilidade aprendida

No final do treino, representar a **utilidade máxima por estado**, isto é, para cada estado `s`:

```text
U(s) = max_a Q[s,a]
```

num **mapa de calor** organizado na grelha 10×10 de estados.

> No PDF original, esta visualização aparece como **Figura 3: Mapa de calor da utilidade aprendida representando a utilidade máxima (utilidade da melhor ação) por estado (obtido com seed = 10, no ponto de teste 5000).**

---

## 2.b) Treino Greedy com Q

Repetir o **mesmo tipo de experiência** de 2.a), com as seguintes diferenças:

- Durante o **treino**, a ação escolhida em cada passo é:
  - a **melhor ação segundo Q** para o estado atual (`argmax_a Q[s,a]`);
  - em caso de empate entre várias ações com o mesmo valor de Q, o desempate deve ser feito **aleatoriamente**.

- Continuar a efetuar testes nos mesmos passos intermédios e medir:
  - recompensa média por passo,
  - tempos de execução.

Comparar os resultados com os obtidos em 2.a).

---

# Exercício 3 — Estratégia ε-Greedy (greed)

Usar uma mistura entre exploração e exploração, controlada por um parâmetro `greed`:

- Com probabilidade `greed`, escolher **a melhor ação** de acordo com Q;
- Com probabilidade `1 − greed`, escolher **uma ação aleatória**.

Exemplos:

- Se `greed = 0.9`, cerca de 90% das ações serão as melhores segundo Q, e 10% aleatórias.
- Se `greed = 0.2`, cerca de 20% das ações serão as melhores, e 80% aleatórias.

Tarefas:

1. Experimentar **três valores diferentes** de `greed` (por exemplo 0.2, 0.5, 0.9) e comparar:
   - evolução da recompensa média,
   - número de passos até ao objetivo,
   - tabelas Q finais.

2. Implementar uma estratégia com **greed crescente**:
   - começar em 30% (`greed = 0.3`) nos primeiros 30% dos passos de treino;
   - aumentar gradualmente o valor de `greed` até **100%** no final.

3. Comparar os resultados desta estratégia com os valores fixos de `greed`, tanto em termos de desempenho (recompensas, passos) como em termos da tabela Q aprendida.

---

# Exercício 4 — Paredes e Penalização

Modificar o ambiente para incluir paredes adicionais, como na **Figura 4** do enunciado original. Nessas figuras:

- algumas colunas da grelha são totalmente ocupadas por blocos verticais de paredes;
- isto torna o problema de navegação **mais difícil**, pois há menos caminhos úteis para chegar ao objetivo.

Sempre que o robot tenta executar uma ação que o levaria para uma parede:

- o estado não muda (permanece no mesmo `s`);
- o robot recebe uma **recompensa de penalização de -0.1**, pequena em comparação com a recompensa final, mas suficiente para desencorajar bater constantemente em paredes.

> No PDF original, este cenário está ilustrado na **Figura 4: As paredes tornam o problema mais difícil. Recompensas de penalização (pequenas em comparação com a recompensa final) podem manter o agente no caminho certo.**

Tarefa:

- Repetir os mesmos tipos de experiências (random walk, greedy, ε-greedy) no novo ambiente com paredes.
- Comparar os resultados com os casos **sem paredes**:
  - tempo médio até ao objetivo;
  - recompensa média;
  - trajetórias aprendidas e mapa de calor de utilidades.

---

# Exercício 5 (Opcional) — Ambiente Não Determinista

Introduzir **não determinismo** nas transições:

- Na maior parte dos casos (por exemplo, 95% das vezes), a ação escolhida leva ao estado esperado;
- Com uma pequena probabilidade (por exemplo, 5%), a mesma ação pode levar o robot a **qualquer estado vizinho** do estado atual, escolhido aleatoriamente.

Tarefa:

- Analisar como esta aleatoriedade afeta:
  - o processo de aprendizagem;
  - a estabilidade da política aprendida;
  - os tempos médios até ao objetivo e as recompensas.

---

# Exercício 6 (Opcional) — Estados com Perceção Contínua

Considerar agora um cenário mais próximo do mundo real:

- Os estados **já não são numerados explicitamente** para o agente;
- O agente apenas percebe o ambiente através de “ecos” nas paredes, isto é, distâncias às paredes em cada direção.

A perceção do agente é uma **matriz/vetor de valores de ponto flutuante**, por exemplo:

```text
(NA, 0.56, NA, 0.14)
```

que significa:

- **UP**: nenhuma parede encontrada (NA);
- **LEFT**: parede a 0.56 m;
- **DOWN**: nenhuma parede (NA);
- **RIGHT**: parede a 0.14 m.

Perguntas a discutir:

1. Como podem estas perceções contínuas ser **simplificadas** ou mapeadas para um **índice discreto** de estado que possa ser usado na tabela Q?
2. Quais são os **riscos** desta transformação num cenário como o da Figura 4?
   - Por exemplo, estados diferentes mas com perceções muito semelhantes (como em certas colunas) podem ser **colapsados no mesmo índice**, levando o agente a **confundir posições distintas**.

---

FIM DO ENUNCIADO
