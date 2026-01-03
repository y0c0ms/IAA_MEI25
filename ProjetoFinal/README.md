# 🔌 D4Maia - Análise e Previsão de Consumo Energético

## Projeto Final | Introdução à Aprendizagem Automática - MEI 2025/2026

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io)
[![Metodologia](https://img.shields.io/badge/Metodologia-CRISP--DM-purple.svg)]()

---

## 📋 Índice

1. [Descrição do Projeto](#-descrição-do-projeto)
2. [Objetivos](#-objetivos)
3. [Metodologia CRISP-DM](#-metodologia-crisp-dm)
4. [Principais Resultados](#-principais-resultados)
5. [Estrutura do Projeto](#-estrutura-do-projeto)
6. [Experiências Realizadas](#-experiências-realizadas)
7. [Visualizações e Gráficos](#-visualizações-e-gráficos)
8. [Instalação e Execução](#-instalação-e-execução)
9. [Requisitos Python](#-requisitos-python)
10. [Datasets Gerados](#-datasets-gerados)
11. [Conclusões e Aplicações Práticas](#-conclusões-e-aplicações-práticas)
12. [Referências](#-referências)

---

## 📖 Descrição do Projeto

Este projeto aplica a metodologia **CRISP-DM** para analisar e prever o consumo energético de edifícios municipais do **Município da Maia**, utilizando o dataset D4Maia.

### Características do Dataset

| Atributo | Descrição |
|----------|-----------|
| **Registos** | ~6 milhões de leituras |
| **Intervalo** | 15 minutos (96 registos/dia) |
| **CPEs** | 89 Códigos de Ponto de Entrega |
| **Período** | 2022-2025 |
| **Variáveis** | `id`, `CPE`, `hora`, `DadosDeConsumo`, `PotActiva`, `PotReactIndut`, `PotReactCapac` |

### Variáveis Principais

- **`CPE`** - Código do Ponto de Entrega (identificador único da instalação)
- **`hora`** (tstamp) - Timestamp da leitura com precisão de 15 minutos
- **`PotActiva`** - Potência ativa registada (kWh) - **variável alvo principal**
- **`DadosDeConsumo`** - Energia consumida para faturação (kWh)
- **`PotReactIndut`** - Potência reativa indutiva (VAR)
- **`PotReactCapac`** - Potência reativa capacitiva (VAR)

---

## 🎯 Objetivos

### Objetivos de Negócio

1. **Caracterização de Consumidores** - Identificar perfis distintos de consumo energético
2. **Deteção de Anomalias** - Identificar instalações com comportamentos atípicos ou desperdícios
3. **Previsão de Consumo** - Prever consumo com antecedência de 1 semana para planeamento

### Objetivos Técnicos

| Objetivo | Técnicas Utilizadas | Métricas |
|----------|---------------------|----------|
| Clustering | K-Means, DBSCAN | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Previsão Séries Temporais | ARIMA, LSTM | MAE, RMSE, MAPE |
| Previsão com Features | RF, XGBoost, MLP | MAE, RMSE |
| Análise de Normalização | StandardScaler | Comparação de métricas com/sem normalização |

---

## 🔬 Metodologia CRISP-DM

O projeto segue rigorosamente a metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

| Fase | Notebook | Descrição |
|------|----------|-----------|
| **1. Business Understanding** | `01_business_data_understanding.ipynb` | Contexto do Município da Maia, objetivos de negócio, stakeholders, métricas de sucesso |
| **2. Data Understanding** | `01_business_data_understanding.ipynb` | EDA completo: distribuições, correlações, pairplot, padrões temporais, análise por CPE |
| **3. Data Preparation** | `02_data_preparation_feature_engineering.ipynb` | Limpeza de dados, tratamento de outliers, feature engineering (38 features) |
| **4. Modeling - Clustering** | `03_clustering_kmeans_dbscan.ipynb` | K-Means (8 clusters), DBSCAN (14.9% ruído), caracterização de perfis |
| **4. Modeling - Time Series** | `04_timeseries_ARIMA_LSTM.ipynb` | ARIMA, LSTM vs Baseline, split 70/30 temporal |
| **4. Modeling - Supervised** | `05_supervised_features_RF_XGB_MLP.ipynb` | RF, XGBoost, MLP com clusters como features |
| **5. Evaluation** | `06_normalization_and_comparisons.ipynb` | Comparação final, análise de normalização, conclusões CRISP-DM |

> **Nota:** A fase de Deployment não foi implementada conforme indicado no enunciado.

---

## 🏆 Principais Resultados

### Clustering (Experiência 1)

| Algoritmo | Clusters | Silhouette | Observações |
|-----------|----------|------------|-------------|
| **K-Means** | 8 | 0.263 | Distribuição equilibrada (1.4%-24.3% por cluster) |
| **DBSCAN** | 3 | 0.317 | 14.9% ruído (outliers identificados) |

**Perfis Identificados:**
- Consumidores diurnos (pico 8h-18h)
- Consumidores 24/7 (consumo constante)
- Outliers com padrões atípicos

### Previsão de Séries Temporais (Experiência 2a)

| Modelo | MAE | RMSE | vs Baseline |
|--------|-----|------|-------------|
| **Baseline** (semana anterior) | 1.225 | 2.003 | - |
| **ARIMA** | **0.804** | **1.176** | **✓ -34.4%** |
| **LSTM** (normalizado) | 1.047 | 1.404 | **✓ +14.6%** |

> **Melhor modelo:** ARIMA superou consistentemente a baseline em 34.4%

### Previsão com Features (Experiência 2b)

| Modelo | Normalização | MAE | RMSE |
|--------|--------------|-----|------|
| Random Forest | Não | 1.629 | 2.168 |
| Random Forest | Sim | 1.630 | 2.170 |
| XGBoost | Não | 1.859 | 2.421 |
| XGBoost | Sim | 1.859 | 2.421 |
| MLP | Não | 1.552 | 2.143 |
| MLP | Sim | 1.625 | 2.203 |

### Impacto da Normalização (Experiência 3)

| Algoritmo | Impacto | Recomendação |
|-----------|---------|--------------|
| K-Means/DBSCAN | **CRÍTICO** | Sempre normalizar |
| LSTM | **ESSENCIAL** | Sempre normalizar |
| MLP | Variável | Testar ambas configurações |
| RF/XGBoost | Sem impacto | Normalização opcional |

---

## 🗂️ Estrutura do Projeto

```
ProjetoFinal/
│
├── 📓 NOTEBOOKS (executar em ordem numérica)
│   ├── 01_business_data_understanding.ipynb   # Business & Data Understanding
│   ├── 02_data_preparation_feature_engineering.ipynb   # Data Preparation
│   ├── 03_clustering_kmeans_dbscan.ipynb      # Clustering (K-Means, DBSCAN)
│   ├── 04_timeseries_ARIMA_LSTM.ipynb         # Séries Temporais (ARIMA, LSTM)
│   ├── 05_supervised_features_RF_XGB_MLP.ipynb # Modelos Supervisionados
│   └── 06_normalization_and_comparisons.ipynb  # Avaliação Final
│
├── 📁 data/
│   └── intermediate/                          # Dados intermédios gerados
│       ├── d4maia_series_per_cpe.csv          # Séries temporais por CPE
│       ├── d4maia_cpe_features.csv            # 38 features agregadas
│       ├── d4maia_ts_train_test_index.csv     # Índices de split temporal
│       ├── d4maia_cpe_clusters.csv            # Labels de clusters
│       ├── d4maia_ts_results.csv              # Resultados ARIMA/LSTM
│       ├── d4maia_feature_models_results.csv  # Resultados RF/XGB/MLP
│       └── d4maia_final_summary.csv           # Resumo comparativo
│
├── 📁 requisitos/                             # Scripts de instalação
│   ├── requirements.txt                       # Dependências Python
│   ├── install_requirements.py
│   ├── setup_windows.bat
│   └── setup_linux_mac.sh
│
├── 📄 consumo15m_11_2025.csv                  # Dataset original D4Maia
├── 📄 RELATORIO_FINAL.md                      # Relatório técnico completo
├── 📄 IAA_Project_2025_2026_v1.pdf            # Enunciado do projeto
└── 📄 README.md                               # Este ficheiro
```

---

## 🔬 Experiências Realizadas

### Experiência 1: Clustering (Aprendizagem Não Supervisionada)

**Objetivo:** Caracterizar diferentes perfis de consumidores e detetar outliers.

**Algoritmos:**
- **K-Means** com método do cotovelo e Silhouette Score
- **DBSCAN** com gráfico K-distance para escolha de eps

**Features Utilizadas (38 no total):**
- Estatísticas de consumo: `consumo_mean`, `consumo_std`, `consumo_cv`, `consumo_min`, `consumo_max`
- Padrões horários: `hora_pico`, `hora_vale`, `consumo_medio_por_hora`
- Perfis temporais: `racio_dia_noite`, `racio_weekend`, `tendencia_semanal`
- Features do enunciado: `avg_afternoon_peak_value`, `avg_daily_peak_time`, `avg_time_below_50_consumption`

**Visualizações Geradas:**
- Método do cotovelo (inércia vs k)
- Silhouette Score por número de clusters
- K-distance plot para DBSCAN
- PCA 2D com clusters coloridos
- Heatmap de características por cluster
- Distribuição de CPEs por cluster

### Experiência 2(a): Previsão com Séries Temporais

**Objetivo:** Prever consumo para a semana seguinte usando dados históricos.

**Divisão Temporal:**
- **Treino:** 70% (dados mais antigos)
- **Teste:** 30% (dados mais recentes)
- **Lag obrigatório:** Features calculadas com dados ≥1 semana antes da previsão

**Modelos:**
- **Baseline:** Consumo da mesma hora, 1 semana antes
- **ARIMA:** Auto-seleção de ordem (p,d,q)
- **LSTM:** Rede recorrente com normalização por CPE

**Métricas:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Comparação percentual vs Baseline

### Experiência 2(b): Previsão com Features Agregadas

**Objetivo:** Usar features agregadas para prever consumo médio semanal.

**Modelos:**
- **Random Forest** (n_estimators=100)
- **XGBoost** (n_estimators=100, max_depth=6)
- **MLP** (hidden_layers=[64, 32], epochs=100)

**Features Adicionais:**
- `cluster_kmeans` e `cluster_dbscan` (resultados do clustering como features)

**Testes de Normalização:**
- Cada modelo testado COM e SEM StandardScaler

### Experiência 3: Análise de Normalização

**Objetivo:** Quantificar o impacto da normalização em cada família de algoritmos.

**Conclusões Documentadas:**
- Clustering baseado em distância: normalização obrigatória
- Redes neuronais (LSTM, MLP): normalização recomendada
- Modelos baseados em árvores (RF, XGBoost): normalização sem impacto

---

## 📊 Visualizações e Gráficos

### Notebook 01 - Data Understanding

| Visualização | Descrição |
|--------------|-----------|
| Histogramas por variável | Distribuição de PotActiva, PotReactIndut, PotReactCapac, DadosDeConsumo |
| Box plots | Identificação visual de outliers |
| Matriz de correlação (heatmap) | Correlações entre variáveis numéricas |
| **Pairplot (scatter matrix)** | Relações entre pares de features |
| Consumo por hora do dia | Padrão de consumo horário agregado |
| Consumo por dia da semana | Padrão semanal |
| Heatmap hora × dia | Consumo médio por hora e dia da semana |
| Séries temporais exemplo | Consumo ao longo do tempo para CPEs exemplo |

### Notebook 03 - Clustering

| Visualização | Descrição |
|--------------|-----------|
| Elbow Method | Inércia vs número de clusters (K-Means) |
| Silhouette Score | Qualidade dos clusters por k |
| K-distance plot | Escolha automática de eps (DBSCAN) |
| PCA 2D | Projeção dos clusters em 2 dimensões |
| Heatmap de perfis | Características médias por cluster |
| Distribuição de clusters | Número de CPEs por cluster |
| Comparação K-Means vs DBSCAN | Tabela de contingência |

### Notebook 04 - Séries Temporais

| Visualização | Descrição |
|--------------|-----------|
| Boxplot MAE por modelo | Distribuição do erro por modelo |
| Boxplot RMSE por modelo | Distribuição do erro quadrático |
| Barplot MAE por CPE | Comparação por CPE e modelo |
| Previsão vs Real | Série temporal prevista vs observada |

### Notebook 05 - Modelos Supervisionados

| Visualização | Descrição |
|--------------|-----------|
| Boxplot MAE por modelo | RF, XGBoost, MLP (com/sem normalização) |
| Barplot melhoria vs baseline | Percentagem de melhoria |
| Comparação normalização | Impacto visual da normalização |

### Notebook 06 - Comparação Final

| Visualização | Descrição |
|--------------|-----------|
| Distribuição K-Means e DBSCAN | Número de CPEs por cluster |
| Boxplot MAE séries temporais | Baseline vs ARIMA vs LSTM |
| Boxplot MAE supervisionados | Todos os modelos |
| Barplot normalização | Impacto em RF, XGBoost, MLP |
| Comparação global | Todos os modelos ordenados por MAE |
| **Erro por cluster** | MAE por cluster K-Means e DBSCAN |
| MAE vs variabilidade | Scatter plot correlacionando erro com CV |

---

## 🚀 Instalação e Execução

### Pré-requisitos

- **Python:** 3.9, 3.10 ou 3.11
- **pip:** Gestor de pacotes Python
- **RAM:** Mínimo 8GB (recomendado 16GB)
- **Espaço em disco:** ~2GB (incluindo dataset e resultados)

### Instalação

**Opção 1: Windows (PowerShell)**
```powershell
cd ProjetoFinal\requisitos
.\setup_windows.bat
```

**Opção 2: Linux/macOS**
```bash
cd ProjetoFinal/requisitos
chmod +x setup_linux_mac.sh
./setup_linux_mac.sh
```

**Opção 3: Manual**
```bash
cd ProjetoFinal/requisitos
pip install -r requirements.txt
```

### Execução dos Notebooks

1. **Navegar para a pasta do projeto:**
   ```bash
   cd ProjetoFinal
   ```

2. **Iniciar Jupyter ou VS Code:**
   ```bash
   jupyter notebook
   # ou abrir no VS Code com extensão Jupyter
   ```

3. **Executar notebooks em ordem numérica (01 → 06)**

> ⚠️ **Importante:** Executar os notebooks sequencialmente, pois cada um depende dos ficheiros gerados pelos anteriores.

---

## 📦 Requisitos Python

### Ficheiro `requirements.txt`

```
# =============================================================================
# D4Maia Project - Requisitos Python
# =============================================================================

# --- Manipulação de Dados ---
pandas>=2.0.0
numpy>=1.24.0

# --- Computação Científica ---
scipy>=1.11.0

# --- Visualização ---
matplotlib>=3.7.0
seaborn>=0.12.0

# --- Machine Learning (Scikit-learn) ---
scikit-learn>=1.3.0

# --- Séries Temporais (ARIMA) ---
statsmodels>=0.14.0

# --- Deep Learning (LSTM) ---
tensorflow>=2.13.0

# --- Gradient Boosting ---
xgboost>=2.0.0

# --- Utilitários Jupyter ---
jupyter>=1.0.0
ipykernel>=6.25.0
notebook>=7.0.0
```

### Instalação Individual (se necessário)

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels tensorflow xgboost jupyter ipykernel notebook
```

---

## 💾 Datasets Gerados

Durante a execução dos notebooks, são gerados os seguintes ficheiros intermédios:

| Ficheiro | Notebook | Descrição | Tamanho |
|----------|----------|-----------|---------|
| `d4maia_series_per_cpe.csv` | 02 | Séries temporais limpas por CPE | ~400 MB |
| `d4maia_cpe_features.csv` | 02 | 38 features agregadas por CPE | ~5 KB |
| `d4maia_ts_train_test_index.csv` | 04 | Índices de split temporal | ~2 KB |
| `d4maia_cpe_clusters.csv` | 03 | Labels de clusters (K-Means, DBSCAN) | ~3 KB |
| `d4maia_ts_results.csv` | 04 | Métricas de ARIMA/LSTM por CPE | ~4 KB |
| `d4maia_feature_models_results.csv` | 05 | Métricas de RF/XGB/MLP | ~5 KB |
| `d4maia_final_summary.csv` | 06 | Tabela resumo comparativa | ~2 KB |

---

## 💡 Conclusões e Aplicações Práticas

### Resultados Técnicos

1. **ARIMA é o melhor modelo** para previsão de séries temporais neste dataset, superando a baseline em 34.4%
2. **Normalização é crítica** para clustering e redes neuronais, mas irrelevante para modelos baseados em árvores
3. **8 clusters K-Means** identificam perfis distintos de consumo com distribuição equilibrada
4. **DBSCAN identifica 14.9% de outliers**, proporção adequada para detecção de anomalias
5. **Modelos supervisionados** não superam baseline semanal, mas fornecem baseline técnica sólida

### Aplicações para o Município da Maia

| Aplicação | Benefício | Modelo Recomendado |
|-----------|-----------|-------------------|
| Previsão semanal para contratos | Redução de custos | ARIMA |
| Caracterização de edifícios | Tarifas diferenciadas | K-Means |
| Deteção de anomalias | Identificação de desperdícios | DBSCAN |
| Benchmarking entre instalações | Priorização de investimentos | Clustering |

### Limitações e Trabalho Futuro

- **Variáveis exógenas:** Incluir temperatura, feriados, eventos especiais
- **Granularidade:** Testar previsões diárias vs semanais
- **Modelos híbridos:** Combinar ARIMA com features para melhorar previsões
- **Mais dados:** Expandir período temporal para capturar sazonalidade anual

---

## 📚 Referências

- **CRISP-DM:** Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining.
- **ARIMA:** Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control.
- **LSTM:** Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- **Random Forest:** Breiman, L. (2001). Random Forests. Machine Learning.
- **XGBoost:** Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- **K-Means:** MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- **DBSCAN:** Ester, M., et al. (1996). A density-based algorithm for discovering clusters.

---

## 👥 Informação Académica

| Campo | Valor |
|-------|-------|
| **Unidade Curricular** | Introdução à Aprendizagem Automática |
| **Curso** | Mestrado em Engenharia Informática (MEI) |
| **Ano Letivo** | 2025/2026 |
| **Dataset** | D4Maia (Município da Maia) |
| **Metodologia** | CRISP-DM |

---

## 📝 Checklist de Entrega

- [x] Notebooks executados do início ao fim (01 → 06)
- [x] Todos os artifacts gerados em `data/intermediate/`
- [x] Comparação gráfica clara entre técnicas
- [x] Baseline implementado e comparado
- [x] Normalização testada em todos os modelos
- [x] Clusters caracterizados com perfis interpretáveis
- [x] Métricas de clustering (Silhouette, Davies-Bouldin)
- [x] Métricas de previsão (MAE, RMSE, MAPE)
- [x] Distribuições e correlações (Data Understanding)
- [x] Pairplot (relações entre pares de features)
- [x] Documentação em PT-PT
- [x] README.md com requisitos e instruções

---

**Última atualização:** Janeiro 2026  
**Status:** ✅ Projeto Completo
