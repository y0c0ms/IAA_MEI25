# 🔌 D4Maia - Previsão de Consumo Energético

## Projeto de Machine Learning | Introdução à Aprendizagem Automática

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

---

## 📋 Descrição

Este projeto desenvolve modelos de **previsão de consumo energético** para a rede de distribuição D4Maia, utilizando dados de consumo em intervalos de 15 minutos. O objetivo é prever o consumo da próxima hora com base em dados históricos de 24 horas.

### 🎯 Objetivos

- Prever consumo energético com **1 hora de antecedência**
- Comparar abordagens de **séries temporais** vs **modelos supervisionados com features**
- Avaliar impacto da **normalização** nos diferentes algoritmos
- Identificar **perfis de consumo** através de clustering

---

## 📊 Metodologia CRISP-DM

O projeto segue a metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

| Fase | Notebook | Descrição |
|------|----------|-----------|
| 1. Business Understanding | `01_business_data_understanding.ipynb` | Compreensão do problema e dados |
| 2. Data Understanding | `01_business_data_understanding.ipynb` | Análise exploratória (EDA) |
| 3. Data Preparation | `02_data_preparation_feature_engineering.ipynb` | Limpeza e engenharia de features |
| 4. Modeling (Clustering) | `03_clustering_kmeans_dbscan.ipynb` | K-Means e DBSCAN |
| 4. Modeling (Time Series) | `04_timeseries_ARIMA_LSTM.ipynb` | ARIMA e LSTM |
| 4. Modeling (Supervised) | `05_supervised_features_RF_XGB_MLP.ipynb` | RF, XGBoost, MLP |
| 5. Evaluation | `06_normalization_and_comparisons.ipynb` | Comparação final e conclusões |

---

## 🏗️ Estrutura do Projeto

```
ProjetoFinal/
│
├── 📓 Notebooks (executar em ordem)
│   ├── 01_business_data_understanding.ipynb      # EDA e compreensão
│   ├── 02_data_preparation_feature_engineering.ipynb  # Preparação de dados
│   ├── 03_clustering_kmeans_dbscan.ipynb         # Clustering
│   ├── 04_timeseries_ARIMA_LSTM.ipynb            # Séries temporais
│   ├── 05_supervised_features_RF_XGB_MLP.ipynb   # Modelos supervisionados
│   └── 06_normalization_and_comparisons.ipynb    # Avaliação final
│
├── 📁 data/
│   ├── intermediate/          # Dados processados entre notebooks
│   │   ├── d4maia_cleaned.csv
│   │   ├── d4maia_features.csv
│   │   └── ...
│   └── output/               # Resultados finais
│
├── 📁 requisitos/            # Scripts de instalação
│   ├── requirements.txt
│   ├── install_requirements.py
│   ├── setup_windows.bat
│   └── setup_linux_mac.sh
│
├── 📄 consumo15m_11_2025.csv  # Dataset original
├── 📄 IAA_Project_2025_2026_v1.pdf  # Enunciado do projeto
└── 📄 README.md              # Este ficheiro
```

---

## 🚀 Instalação Rápida

### Pré-requisitos

- Python 3.9, 3.10 ou 3.11
- pip (gestor de pacotes Python)

### Passos

1. **Clonar/Descarregar o projeto**

2. **Instalar dependências:**

   **Windows:**
   ```powershell
   cd ProjetoFinal\requisitos
   pip install -r requirements.txt
   ```

   **Linux/macOS:**
   ```bash
   cd ProjetoFinal/requisitos
   pip install -r requirements.txt
   ```

   **Ou usar scripts automáticos:**
   ```powershell
   # Windows
   .\requisitos\setup_windows.bat
   
   # Linux/macOS
   chmod +x requisitos/setup_linux_mac.sh
   ./requisitos/setup_linux_mac.sh
   ```

3. **Executar notebooks em ordem** (01 → 06)

---

## 📦 Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| pandas | ≥2.0.0 | Manipulação de dados |
| numpy | ≥1.24.0 | Computação numérica |
| matplotlib | ≥3.7.0 | Visualização |
| seaborn | ≥0.12.0 | Visualização estatística |
| scikit-learn | ≥1.3.0 | ML (clustering, RF, MLP) |
| statsmodels | ≥0.14.0 | ARIMA |
| tensorflow | ≥2.13.0 | LSTM |
| xgboost | ≥2.0.0 | Gradient Boosting |

---

## 📈 Resultados Principais

### Modelos de Séries Temporais

| Modelo | MAE | RMSE | MAPE | Melhoria vs Baseline |
|--------|-----|------|------|---------------------|
| **Baseline** | 1.1672 | 1.6841 | 37.63% | - |
| **ARIMA** | **0.9083** | **1.2861** | **26.50%** | **+22.2%** ✅ |
| LSTM | 1.0521 | 1.4328 | 35.32% | +9.9% |

### Modelos Supervisionados (Features)

| Modelo | Normalização | MAE | RMSE | Melhoria vs Baseline |
|--------|--------------|-----|------|---------------------|
| Random Forest | Não | 1.0633 | 1.4779 | +1.9% |
| XGBoost | Não | 1.0693 | 1.4769 | +1.4% |
| **MLP** | **Não** | **1.0425** | **1.4500** | **+3.9%** ✅ |
| MLP | StandardScaler | 1.0706 | 1.4914 | +1.2% |

### Clustering

| Algoritmo | Clusters | Observações |
|-----------|----------|-------------|
| K-Means | 2 | Silhouette: 0.45 |
| DBSCAN | 2 + 35 ruído | 39.3% outliers |

---

## 🔍 Conclusões

1. **ARIMA** é o melhor modelo para séries temporais (+22.2% vs baseline)
2. **MLP sem normalização** é o melhor modelo supervisionado (+3.9% vs baseline)
3. Séries temporais superam modelos baseados em features para este problema
4. Normalização **não melhora** o desempenho na maioria dos casos
5. Existem **2 perfis de consumo** distintos na rede D4Maia

---

## 💼 Contexto de Negócio

### Aplicações Práticas

- **Gestão de rede:** Previsão de picos de consumo
- **Planeamento:** Otimização de recursos energéticos
- **Manutenção:** Deteção de anomalias de consumo
- **Tarifação:** Suporte a tarifas dinâmicas

### Limitações

- Dados limitados a Novembro 2025
- Sem variáveis externas (temperatura, feriados)
- Previsão limitada a 1 hora de horizonte

---

## 👥 Equipa

**Projeto Final - Introdução à Aprendizagem Automática**  
Mestrado em Engenharia Informática (MEI) 2025/2026

---

## 📄 Licença

Este projeto foi desenvolvido para fins académicos no âmbito da UC de Introdução à Aprendizagem Automática.

---

## 📚 Referências

- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
