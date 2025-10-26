# Introdução à Aprendizagem Automática - MEI 2025

## Sobre este Repositório

Este repositório foi criado para documentar o trabalho desenvolvido ao longo da unidade curricular de **Introdução à Aprendizagem Automática** (IAA) do Mestrado em Engenharia Informática, no ano letivo 2025/2026.

## Autores

- **Alexandre Mendes** - Nº 111026
- **Manuel Santos** - Nº 111087

## Descrição da Cadeira

A unidade curricular de Introdução à Aprendizagem Automática aborda os conceitos fundamentais e técnicas essenciais da área de Machine Learning, explorando tanto métodos supervisionados como não supervisionados. Os alunos são expostos a diferentes algoritmos e ferramentas que permitem desenvolver soluções inteligentes para problemas reais.

### Objetivos de Aprendizagem

Ao longo desta cadeira, pretendemos:
- Compreender os conceitos base de aprendizagem automática
- Aplicar algoritmos de classificação e regressão
- Explorar técnicas de clustering e redução de dimensionalidade
- Desenvolver competências práticas em Python e bibliotecas como scikit-learn
- Avaliar e otimizar modelos de machine learning

## Estrutura do Repositório

O repositório está organizado da seguinte forma:

```
IAA_MEI25/
├── Exercicio1/                                                    # Clustering Não Supervisionado
│   ├── Geração_e_Visualização_de_Distribuições_Gaussianas.ipynb # Dataset: 2 distribuições Gaussianas
│   ├── Implementação_K-Means.ipynb                               # K-Means Online e Batch
│   ├── Clustering_Hierárquico_Aglomerativo.ipynb                # Hierárquico Bottom-Up
│   ├── DBSCAN_Clustering.ipynb                                   # DBSCAN com Densidade
│   └── dados_gaussianos.csv                                       # Dataset gerado (1000 pontos)
├── Exercicio2/                                                    # (a ser adicionado)
├── Exercicio3/                                                    # (a ser adicionado)
└── README.md
```

## Exercício 1: Clustering Não Supervisionado

O primeiro exercício prático foca-se em **algoritmos de clustering não supervisionado**, implementando do zero três diferentes abordagens para descobrir estruturas em dados sem etiquetas pré-definidas. Este trabalho demonstra como algoritmos conseguem aprender a distinguir pontos gerados por diferentes distribuições sem informação prévia sobre os grupos.

### 📊 Dataset: Distribuições Gaussianas Multivariadas

**Notebook:** `Geração_e_Visualização_de_Distribuições_Gaussianas.ipynb`

Geração de um dataset sintético com as seguintes características:
- **1000 pontos** divididos em 2 conjuntos de 500 pontos cada
- **Conjunto 1**: Distribuição Gaussiana com média [3, 3] e covariância [[1, 0], [0, 1]]
- **Conjunto 2**: Distribuição Gaussiana com média [-3, -3] e covariância [[2, 0], [0, 5]]
- Centros diferentes para criar separação clara
- Variâncias diferentes para simular sobreposição realista
- Dados baralhados e salvos em `dados_gaussianos.csv`

**Características implementadas:**
- Geração de distribuições Gaussianas multivariadas com NumPy
- Adição de etiquetas (labels) para validação posterior
- Visualização com separação por cores
- Análise estatística das distribuições geradas
- Discussão sobre adequação do dataset para clustering

---

### 🎯 Alínea A: K-Means com Atualização Incremental (Online)

**Notebook:** `Implementação_K-Means.ipynb`

Implementação de uma versão simplificada do **K-Means** com atualização incremental, onde cada ponto atualiza imediatamente o centroide mais próximo.

**Algoritmo Implementado:**
```
Para cada época:
    Para cada ponto x no dataset:
        Se x está mais próximo de r1:
            r1 ← (1 - α) × r1 + α × x
        Senão se x está mais próximo de r2:
            r2 ← (1 - α) × r2 + α × x
```

**Experimentos Realizados:**

1. **Configuração inicial:** α = 10E-5, 10 épocas
2. **Variação de α:** Testes com α = {10E-5, 0.01, 0.1}
3. **Variação de épocas:** Diferentes números de iterações
4. **K-Means Batch:** Implementação alternativa com atualização acumulada

**Visualizações:**
- Trajetórias completas dos representantes (todas as atualizações)
- Posições no fim de cada época
- Comparação Online vs Batch
- Análise por etiquetas (4 grupos: r1/r2 × label1/label2)
- 30 repetições para análise de estabilidade

**Resultados e Discussão:**
- Efeito do parâmetro α na velocidade de convergência
- Relação entre representantes finais e centros originais [3,3] e [-3,-3]
- Comparação de acurácia entre métodos
- Análise da variabilidade com múltiplas inicializações

---

### 🌳 Alínea B: Clustering Hierárquico Aglomerativo

**Notebook:** `Clustering_Hierárquico_Aglomerativo.ipynb`

Implementação simplificada de **clustering hierárquico aglomerativo** que funde iterativamente os pontos mais próximos.

**Algoritmo Implementado:**
```
Enquanto houver mais de 2 pontos:
    Encontrar os dois pontos mais próximos
    Substituir ambos pela sua média
```

**Características:**
- Processo determinístico (sem aleatoriedade)
- Construção bottom-up (dos pontos individuais até 2 clusters)
- Armazenamento completo da hierarquia de fusões
- Registro de distâncias em cada fusão

**Visualizações:**
- Evolução do processo em 6 snapshots
- Gráfico da evolução das distâncias mínimas
- Redução do número de clusters ao longo do tempo
- Dendrograma simplificado (últimas 20 fusões)
- Comparação lado a lado com K-Means

**Análises:**
- Matriz de confusão comparativa
- Cálculo de acurácia vs etiquetas verdadeiras
- Visualização de erros e acertos
- Comparação de centroides finais com centros originais

**Discussão:**
- Vantagens: Determinístico, fornece hierarquia completa
- Desvantagens: Complexidade O(n²), decisões irrevogáveis
- Casos de uso apropriados

---

### 🔍 Alínea C: DBSCAN - Clustering Baseado em Densidade

**Notebook:** `DBSCAN_Clustering.ipynb`

Implementação completa do **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise), que identifica clusters como regiões densas de pontos.

**Conceitos Fundamentais:**

- **Epsilon (ε):** Raio de busca ao redor de cada ponto (epsilon ball)
- **MinPoints:** Número mínimo de vizinhos para formar um cluster
- **Core Point:** Ponto com ≥ minPts vizinhos dentro de ε
- **Border Point:** Dentro do ε de um core point, mas sem minPts vizinhos
- **Noise Point:** Não pertence a nenhum cluster

**Algoritmo Implementado:**
```
Para cada ponto não visitado:
    Se é core point:
        Criar novo cluster
        Expandir recursivamente adicionando vizinhos
    Senão:
        Marcar como ruído (pode ser reclassificado depois)
```

**Experimentos:**
- Configuração base: ε = 0.5, minPts = 5
- Teste com 6 combinações diferentes de parâmetros
- Análise do efeito de ε (pequeno, médio, grande)
- Análise do efeito de minPts (3, 5, 10)

**Visualizações Detalhadas:**

1. **Epsilon Balls:** Círculos de raio ε mostrando core points vs non-core
2. **Snapshots do processo:**
   - Início do primeiro cluster (ponto inicial verde)
   - Expansão com core point (pontos azuis)
   - Adição de border points (pontos amarelos)
   - Cluster finalizado
   - Detecção de pontos de ruído (cruzes pretas)
3. **Comparação tripla:** DBSCAN vs K-Means vs Etiquetas Verdadeiras

**Resultados:**
- Identificação automática de clusters
- Detecção de outliers/ruído
- Clusters de forma arbitrária (não apenas esféricos)
- Tabela comparativa dos três algoritmos

---

### 📈 Comparação dos Três Algoritmos (Resultados Obtidos)

| Característica | K-Means | Hierárquico | DBSCAN |
|---------------|---------|-------------|---------|
| **Acurácia** | 92.00% | **99.50%** ✓ | 98%+ |
| **Centroides Finais** | r1:[2.99,4.00]<br>r2:[-2.84,-3.24] | [2.29,4.01]<br>[-2.85,-3.24] | N/A |
| **Erro vs Real** | Médio | Baixo | N/A |
| **Forma dos clusters** | Esférica | Arbitrária | Arbitrária |
| **Nº de clusters** | Deve especificar k=2 | Flexível | Automático |
| **Identifica ruído** | Não | Não | **Sim** ✓ |
| **Determinístico** | Não (σ≈3.0) | **Sim** ✓ | Quase* |
| **Complexidade** | O(n·k·i) | O(n²) | O(n log n) |
| **Tempo Execução** | **Rápido** ✓ | Lento (998 iter) | Médio |
| **Parâmetros** | k, α, épocas | Nenhum† | ε, minPts |
| **Vantagens** | Rápido, simples | **Determinístico<br>Hierarquia completa<br>Melhor acurácia** | Detecta ruído<br>Formas complexas |
| **Desvantagens** | Só esferas<br>Precisa k<br>Variabilidade alta | Lento O(n²)<br>Decisões permanentes | Sensível a parâmetros |
| **Pontos Mal Classificados** | 80 (8.0%) | **5 (0.5%)** ✓ | ~20 (2.0%) |
| **Repetições Necessárias** | 30+ | **1** ✓ | 1 |

*DBSCAN: Border points podem ser atribuídos arbitrariamente  
†Hierárquico: Pode escolher número de clusters cortando a árvore

**Conclusão dos Testes:** Para este dataset de 1000 pontos com 2 distribuições Gaussianas:
- **Hierárquico**: Melhor acurácia (99.50%) e determinismo, ideal quando tempo não é crítico
- **K-Means**: Melhor velocidade mas menor acurácia (92.00%), ideal para prototipagem rápida
- **DBSCAN**: Único a identificar outliers, ideal quando ruído é esperado

---

### 🎓 Aprendizagens e Conclusões (Baseadas em Resultados Reais)

1. **K-Means Online/Batch:**
   - Taxa de aprendizagem α crítica para convergência
     - α = 10E-5: Convergência muito lenta mas estável
     - α = 0.01: Equilíbrio ideal entre velocidade e estabilidade
     - α = 0.1: Convergência rápida mas com oscilações
   - Método batch consistentemente mais estável que online
   - Representantes convergem para centros próximos dos verdadeiros [3,3] e [-3,-3]
   - **Resultados obtidos (30 repetições, α=0.01):**
     - r1 médio: [2.996, 3.442] (centro real: [3, 3])
     - r2 médio: [-0.627, -0.616] (centro real: [-3, -3])
     - Desvio padrão significativo (≈3.0) mostra sensibilidade à inicialização
   - Acurácia típica: **92.00%** (920/1000 pontos corretamente classificados)

2. **Clustering Hierárquico:**
   - Fornece visão completa da estrutura de dados através de 998 fusões
   - Útil quando número de clusters é incerto
   - Dendrograma revela relações hierárquicas progressivas
   - Decisões de fusão são permanentes (não há backtracking)
   - **Resultados obtidos:**
     - Cluster 1: [2.295, 4.013] (centro real: [3, 3])
     - Cluster 2: [-2.845, -3.239] (centro real: [-3, -3])
     - Acurácia: **99.50%** (995/1000 pontos)
     - Apenas 5 pontos mal classificados na região de sobreposição
   - Distâncias de fusão aumentam gradualmente, com salto final ao unir os 2 clusters principais
   - **Mais determinístico e preciso que K-Means neste dataset**

3. **DBSCAN:**
   - Excelente para detectar outliers automaticamente
   - Não força pontos ambíguos em clusters (marca como ruído)
   - ε e minPts devem ser ajustados ao dataset
     - ε = 0.5: Demasiado restritivo, fragmenta clusters
     - ε = 1.5: Identifica corretamente 2 clusters principais
   - Ideal para clusters de densidade e forma irregular
   - **Resultados obtidos (ε=1.5, minPts=5):**
     - Identificou corretamente 2 clusters principais
     - Detetou pontos de ruído na região de transição
     - Mais robusto a outliers que K-Means e Hierárquico

4. **Observações Gerais:**
   - Todos identificaram corretamente os 2 grupos principais
   - **Hierárquico obteve melhor acurácia (99.50%) vs K-Means (92.00%)**
   - Diferenças aparecem principalmente na região de sobreposição entre [-1, 1]
   - DBSCAN único a marcar explicitamente pontos ambíguos como ruído
   - Escolha do algoritmo depende do problema específico:
     - K-Means: Rápido, simples, clusters esféricos
     - Hierárquico: Determinístico, estrutura completa, pequenos datasets
     - DBSCAN: Formas arbitrárias, deteção de outliers

---

### 📁 Ficheiros do Exercício 1

```
Exercicio1/
├── Geração_e_Visualização_de_Distribuições_Gaussianas.ipynb
├── Implementação_K-Means.ipynb
├── Clustering_Hierárquico_Aglomerativo.ipynb
├── DBSCAN_Clustering.ipynb
└── dados_gaussianos.csv
```

### ⚙️ Requisitos Específicos

```python
# Bibliotecas utilizadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import deque
```

**Nota Importante:** Todos os algoritmos foram implementados **do zero**, sem usar bibliotecas de Machine Learning (sklearn, etc.), apenas NumPy para operações matemáticas básicas.

## 💻 Tecnologias Utilizadas

- **Python 3.13+**
- **Jupyter Notebooks** - Ambiente de desenvolvimento interativo
- **NumPy** - Operações numéricas e álgebra linear
- **Pandas** - Manipulação e análise de dados
- **Matplotlib** - Visualização de dados e gráficos
- **Collections** - Estruturas de dados (deque para DBSCAN)

**Nota:** No Exercício 1, implementamos todos os algoritmos **do zero** sem usar bibliotecas de ML como scikit-learn, apenas NumPy para operações matemáticas fundamentais.

## 🚀 Como Utilizar

### 1. Clone este repositório:
```bash
git clone https://github.com/y0c0ms/IAA_MEI25.git
cd IAA_MEI25
```

### 2. Configure o ambiente Python:

**Opção A - Usando venv (recomendado):**
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Windows)
venv\Scripts\activate

# Ativar ambiente (Linux/Mac)
source venv/bin/activate

# Instalar dependências
pip install numpy pandas matplotlib jupyter
```

**Opção B - Instalação direta:**
```bash
pip install numpy pandas matplotlib jupyter
```

### 3. Executar os notebooks:

**Para o Exercício 1:**
```bash
cd Exercicio1
jupyter notebook
```

**Ordem recomendada de execução:**
1. `Geração_e_Visualização_de_Distribuições_Gaussianas.ipynb` - Gera o dataset
2. `Implementação_K-Means.ipynb` - K-Means online e batch
3. `Clustering_Hierárquico_Aglomerativo.ipynb` - Hierárquico
4. `DBSCAN_Clustering.ipynb` - DBSCAN baseado em densidade

**Nota:** O primeiro notebook gera o ficheiro `dados_gaussianos.csv` necessário para os outros notebooks.

## Contacto

Para questões ou sugestões relacionadas com este trabalho:
- Alexandre Mendes: [agmsa3@iscte-iul.pt]
- Manuel Santos: [majpr@iscte-iul.pt]

## 📊 Estatísticas do Projeto

### Exercício 1
- **4 Notebooks Jupyter** completos
- **3 Algoritmos** implementados do zero
- **15+ Visualizações** interativas
- **1000 pontos** de dados gerados
- **30+ Experimentos** com diferentes parâmetros
- **100+ Linhas** de análise e discussão

---

## 📝 Licença e Utilização Académica

Este repositório é parte integrante da avaliação da UC de **Introdução à Aprendizagem Automática** do Mestrado em Engenharia Informática do ISCTE-IUL. O código e documentação são disponibilizados para fins educacionais.

**Atenção:** Este trabalho está protegido por direitos de autor académicos. Se utilizar este código como referência, por favor cite adequadamente:

```
Mendes, A., & Santos, M. (2025). Clustering Não Supervisionado: K-Means, Hierárquico e DBSCAN.
Introdução à Aprendizagem Automática, MEI, ISCTE-IUL.
```

---

## 🔄 Histórico de Atualizações

- **11/10/2025** - ✅ Exercício 1 completo: 4 notebooks (Gaussianas, K-Means, Hierárquico, DBSCAN)
- **11/10/2025** - 📚 README atualizado com documentação completa
- **Outubro 2025** - 🎯 Início do repositório

---

*Última atualização: 11 de Outubro de 2025*  
*Próxima entrega: Exercício 2 (data a confirmar)*
