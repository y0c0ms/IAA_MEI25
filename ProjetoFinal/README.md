# 🔌 D4Maia - Análise e Previsão de Consumo Energético

## Projeto Final | Introdução à Aprendizagem Automática - MEI 2025/2026

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

---

## 📋 Descrição

Este projeto aplica a metodologia **CRISP-DM** para analisar e prever o consumo energético do **Município da Maia** utilizando o dataset D4Maia com **~6 milhões de registos** de leituras realizadas a cada **15 minutos** em 23 instalações municipais.

### 🎯 Objetivos

1. **Caracterização de Consumidores** - Identificar perfis distintos através de clustering (K-Means, DBSCAN)
2. **Previsão de Séries Temporais** - Avaliar capacidade preditiva de ARIMA e LSTM para uma semana à frente
3. **Modelos Supervisionados** - Comparar Random Forest, XGBoost e MLP com features agregadas
4. **Análise de Normalização** - Avaliar impacto sistemático da normalização em todos os modelos

### 🏆 Principais Resultados

| Abordagem | Melhor Modelo | MAE | Melhoria vs Baseline |
|-----------|---------------|-----|---------------------|
| **Séries Temporais** | 🥇 **ARIMA** | **0.908 kWh** | **-22.2%** ✓ |
| **Features Agregadas** | 🥈 **MLP** | **1.043 kWh** | **-13.8%** ✓ |
| **Baseline** (semana anterior) | - | 1.167 kWh | - |

**Conclusão**: ARIMA demonstrou ser o modelo mais eficaz, superando consistentemente a baseline em 22%.

---

## 📊 Metodologia CRISP-DM

O projeto segue rigorosamente a metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

| Fase | Notebook | Descrição | Status |
|------|----------|-----------|--------|
| **1. Business Understanding** | `01_business_data_understanding.ipynb` | Compreensão do problema de negócio | ✅ Completo |
| **2. Data Understanding** | `01_business_data_understanding.ipynb` | Análise exploratória (EDA) | ✅ Completo |
| **3. Data Preparation** | `02_data_preparation_feature_engineering.ipynb` | Limpeza e engenharia de features | ✅ Completo |
| **4. Modeling - Clustering** | `03_clustering_kmeans_dbscan.ipynb` | K-Means e DBSCAN | ✅ Completo |
| **4. Modeling - Time Series** | `04_timeseries_ARIMA_LSTM.ipynb` | ARIMA e LSTM | ✅ Completo |
| **4. Modeling - Supervised** | `05_supervised_features_RF_XGB_MLP.ipynb` | RF, XGBoost, MLP | ✅ Completo |
| **5. Evaluation** | `06_normalization_and_comparisons.ipynb` | Comparação final e conclusões | ✅ Completo |

---

## 🔬 Experiências Realizadas

### 🎯 Experiência 1: Clustering
- **Algoritmos**: K-Means (k=2 a 10) e DBSCAN (múltiplos parâmetros)
- **Resultado**: Identificados **2 clusters principais** + **3 outliers**
  - **Cluster 0**: "Alto Volume" (8 CPEs) - Consumo médio 15.3 kWh/dia
  - **Cluster 1**: "Baixo Volume Constante" (15 CPEs) - Consumo médio 3.2 kWh/dia
- **Métricas**: Silhouette Score = 0.42 (boa separação)
- **Insight**: Normalização é **CRÍTICA** para clustering baseado em distância

### ⏰ Experiência 2(a): Séries Temporais
- **Modelos**: Baseline, ARIMA, LSTM
- **Divisão**: 70% treino / 30% teste (split temporal)
- **Resultados**:
  - 🥇 **ARIMA**: MAE = 0.908 kWh (-22.2% vs baseline)
  - 🥈 **Baseline**: MAE = 1.167 kWh
  - 🥉 **LSTM**: MAE = 1.187 kWh (+1.7% vs baseline)
- **Insight**: ARIMA superou expectativas; LSTM requer mais dados/tuning

### 🎓 Experiência 2(b): Features Agregadas
- **Modelos**: Random Forest, XGBoost, MLP
- **Features**: 38 features temporais e estatísticas (com lag de 1 semana)
- **Resultados**:
  - 🥇 **MLP (sem norm)**: MAE = 1.043 kWh
  - 🥈 **RF**: MAE = 1.209 kWh
  - 🥉 **XGBoost**: MAE = 1.440 kWh
- **Insight**: MLP teve performance surpreendente sem normalização neste dataset pequeno

### 🔄 Experiência de Normalização
- **Conclusões por modelo**:
  - ⚠️ **K-Means/DBSCAN**: Normalização **OBRIGATÓRIA**
  - ⚠️ **LSTM**: Normalização **ESSENCIAL**
  - ⚡ **MLP**: Impacto **VARIÁVEL** (testar ambas configurações)
  - ✅ **RF/XGBoost**: Normalização **SEM IMPACTO** (baseados em árvores)

---

## 🏗️ Estrutura do Projeto

```
ProjetoFinal/
│
├── 📓 Notebooks (executar em ordem numérica)
│   ├── 01_business_data_understanding.ipynb      # Business & Data Understanding
│   ├── 02_data_preparation_feature_engineering.ipynb  # Data Preparation
│   ├── 03_clustering_kmeans_dbscan.ipynb         # Clustering (K-Means, DBSCAN)
│   ├── 04_timeseries_ARIMA_LSTM.ipynb            # Séries Temporais (ARIMA, LSTM)
│   ├── 05_supervised_features_RF_XGB_MLP.ipynb   # Modelos Supervisionados
│   └── 06_normalization_and_comparisons.ipynb    # Avaliação Final
│
├── 📁 data/
│   └── intermediate/          # Dados processados entre notebooks
│       ├── d4maia_series_per_cpe.csv            # Séries temporais por CPE
│       ├── d4maia_cpe_features.csv               # Features agregadas
│       ├── d4maia_ts_train_test_index.csv        # Índices de split temporal
│       ├── d4maia_cpe_clusters.csv               # Resultados de clustering
│       ├── d4maia_ts_results.csv                 # Resultados ARIMA/LSTM
│       ├── d4maia_feature_models_results.csv     # Resultados RF/XGB/MLP
│       └── d4maia_final_summary.csv              # Resumo final comparativo
│
├── 📁 requisitos/            # Scripts de instalação
│   ├── requirements.txt       # Dependências Python
│   ├── install_requirements.py
│   ├── setup_windows.bat      # Setup automático Windows
│   ├── setup_linux_mac.sh     # Setup automático Linux/macOS
│   └── README.md
│
├── 📄 consumo15m_11_2025.csv  # Dataset original D4Maia (~6M registos)
├── 📄 RELATORIO_FINAL.md      # ⭐ Relatório completo do projeto (NOVO)
├── 📄 IAA_projeto_prompt_completo.txt  # Enunciado e plano detalhado
└── 📄 README.md              # Este ficheiro
```

---

## 🚀 Instalação e Execução

### Pré-requisitos

- **Python**: 3.9, 3.10 ou 3.11
- **pip**: Gestor de pacotes Python
- **Jupyter**: Para executar notebooks (incluído nas dependências)
- **RAM**: 8GB mínimo (16GB recomendado)

### Instalação Rápida

**Opção 1: Instalação Automática (Windows)**
```powershell
cd ProjetoFinal\requisitos
.\setup_windows.bat
```

**Opção 2: Instalação Automática (Linux/macOS)**
```bash
cd ProjetoFinal/requisitos
chmod +x setup_linux_mac.sh
./setup_linux_mac.sh
```

**Opção 3: Instalação Manual**
```bash
cd ProjetoFinal/requisitos
pip install -r requirements.txt
```

### Dependências Principais

```
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.3.0
tensorflow >= 2.13.0
statsmodels >= 0.14.0
xgboost >= 1.7.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
jupyter >= 1.0.0
```

### Execução dos Notebooks

1. **Navegar para a pasta do projeto**
   ```bash
   cd ProjetoFinal
   ```

2. **Iniciar Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **Executar notebooks em ordem**:
   - `01_business_data_understanding.ipynb`
   - `02_data_preparation_feature_engineering.ipynb`
   - `03_clustering_kmeans_dbscan.ipynb`
   - `04_timeseries_ARIMA_LSTM.ipynb`
   - `05_supervised_features_RF_XGB_MLP.ipynb`
   - `06_normalization_and_comparisons.ipynb`

**⚠️ Importante**: Executar os notebooks na ordem indicada, pois cada um depende dos ficheiros gerados pelos anteriores.

---

## 📊 Datasets Gerados

Durante a execução dos notebooks, são gerados os seguintes ficheiros em `data/intermediate/`:

| Ficheiro | Descrição | Tamanho Aprox. |
|----------|-----------|----------------|
| `d4maia_series_per_cpe.csv` | Séries temporais limpas e organizadas | ~400 MB |
| `d4maia_cpe_features.csv` | 38 features agregadas por CPE | 5 KB |
| `d4maia_ts_train_test_index.csv` | Índices temporais de split | 2 KB |
| `d4maia_cpe_clusters.csv` | Labels de clusters (K-Means, DBSCAN) | 3 KB |
| `d4maia_ts_results.csv` | Métricas de ARIMA/LSTM por CPE | 4 KB |
| `d4maia_feature_models_results.csv` | Métricas de RF/XGB/MLP | 5 KB |
| `d4maia_final_summary.csv` | Tabela resumo final | 2 KB |

---

## 📖 Documentação

### Relatório Final

📄 **[RELATORIO_FINAL.md](RELATORIO_FINAL.md)** - Relatório completo do projeto com:
- Análise detalhada seguindo CRISP-DM
- Resultados de todos os modelos
- Comparações e conclusões
- Referências bibliográficas
- Recomendações para o Município da Maia

### Notebooks Documentados

Cada notebook contém:
- ✅ Explicações teóricas de cada técnica
- ✅ Código comentado linha a linha
- ✅ Visualizações detalhadas
- ✅ Análise crítica dos resultados
- ✅ Ligação explícita à metodologia CRISP-DM
---

## 💡 Resultados e Aplicações Práticas

### Para o Município da Maia

**🎯 Planeamento Energético**
- Previsões semanais para negociação de contratos
- Precisão: Erro típico de 3-10% (MAPE)
- Benefício: Redução de custos por evitar penalizações

**🔍 Deteção de Anomalias**
- 3 outliers identificados para auditoria
- Alertas quando consumo real > previsão + 2σ
- Benefício: Identificação precoce de desperdícios

**📊 Tarifas Diferenciadas**
- Cluster 0: Tarifa por demanda (picos altos)
- Cluster 1: Tarifa flat (consumo constante)
- Benefício: Otimização de custos por perfil

**📈 Benchmarking entre Edifícios**
- Comparar CPEs do mesmo cluster
- Identificar ineficiências relativas
- Priorizar investimentos em eficiência energética

---

## 🔬 Contribuições Técnicas

### Inovações Implementadas

1. **Sistema de Configuração Dinâmica**
   - Parâmetros adaptados automaticamente ao tamanho do dataset
   - Thresholds ajustados dinamicamente (exemplo: silhouette)

2. **Pipeline Completo de Séries Temporais**
   - Baseline inteligente (semana anterior)
   - ARIMA com seleção automática de ordem
   - LSTM com early stopping e normalização por CPE

3. **Feature Engineering Temporal Rigoroso**
   - Respeitando lag de 1 semana (evita data leakage)
   - 38 features agregadas interpretáveis

4. **Análise Sistemática de Normalização**
   - Testado em TODOS os modelos
   - Documentação clara do impacto por tipo de algoritmo

---

## 🏆 Destaques do Projeto

✅ **Metodologia Rigorosa**: CRISP-DM seguida em todas as fases  
✅ **Reprodutibilidade**: Seeds fixadas, código documentado  
✅ **Comparações Justas**: Mesmos CPEs, mesmas métricas, mesmos splits  
✅ **Interpretabilidade**: Perfis de consumo claramente caracterizados  
✅ **Aplicabilidade**: Recomendações práticas para o município  
✅ **Documentação Completa**: Relatório técnico + notebooks anotados  

---

## 📚 Referências

- **CRISP-DM**: Wirth & Hipp (2000) - Processo padrão de Data Mining
- **ARIMA**: Box & Jenkins (2015) - Time Series Analysis
- **LSTM**: Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
- **Random Forest**: Breiman (2001) - Ensemble Learning
- **XGBoost**: Chen & Guestrin (2016) - Gradient Boosting
- **Scikit-learn**: Pedregosa et al. (2011) - ML em Python

Ver [RELATORIO_FINAL.md](RELATORIO_FINAL.md) para referências completas.

---

## 👥 Projeto Desenvolvido Para

- **UC**: Introdução à Aprendizagem Automática
- **Mestrado**: Engenharia Informática (MEI)
- **Ano Letivo**: 2025/2026
- **Dataset**: D4Maia (Município da Maia)

---

## 📞 Suporte

Para questões sobre o projeto:
1. Consultar [RELATORIO_FINAL.md](RELATORIO_FINAL.md) para detalhes técnicos
2. Verificar notebooks individuais para implementação específica
3. Consultar `requisitos/README.md` para problemas de instalação

---

## 📝 Licença

Este projeto foi desenvolvido para fins académicos como parte da UC "Introdução à Aprendizagem Automática" do Mestrado em Engenharia Informática.

---

**Última atualização**: Dezembro 2025  
**Status**: ✅ Projeto Completo
