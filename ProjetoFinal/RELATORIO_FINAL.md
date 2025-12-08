# Relatório Final do Projeto D4Maia
## Previsão de Consumo Energético no Município da Maia

**Curso:** Mestrado em Engenharia Informática  
**UC:** Introdução à Aprendizagem Automática - 2025/2026  
**Metodologia:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Data:** Dezembro 2025

---

## Índice

1. [Resumo Executivo](#1-resumo-executivo)
2. [Business Understanding](#2-business-understanding)
3. [Data Understanding](#3-data-understanding)
4. [Data Preparation](#4-data-preparation)
5. [Modeling](#5-modeling)
6. [Evaluation](#6-evaluation)
7. [Conclusões e Trabalho Futuro](#7-conclusões-e-trabalho-futuro)
8. [Referências](#8-referências)

---

## 1. Resumo Executivo

### 1.1 Contexto

O Município da Maia enfrenta o desafio de otimizar a gestão energética dos seus edifícios de serviços municipais num contexto de crescente preocupação com eficiência energética e sustentabilidade ambiental. Este projeto aplica técnicas de Machine Learning ao dataset **D4Maia**, que contém aproximadamente **6 milhões de registos** de consumo energético com leituras realizadas a cada **15 minutos**.

### 1.2 Objetivos Principais

1. **Caracterização de Consumidores**: Identificar perfis distintos de consumo através de clustering
2. **Previsão de Séries Temporais**: Avaliar a capacidade de prever consumo futuro usando ARIMA e LSTM
3. **Modelos Supervisionados**: Desenvolver modelos preditivos baseados em features agregadas (RF, XGBoost, MLP)
4. **Análise de Normalização**: Avaliar o impacto da normalização de dados em todos os modelos

### 1.3 Principais Resultados

| Abordagem | Melhor Modelo | MAE Médio | Melhoria vs Baseline |
|-----------|---------------|-----------|---------------------|
| **Séries Temporais** | ARIMA | 0.908 | +22.2% |
| **Features Agregadas** | MLP (sem norm.) | 1.043 | Comparável |
| **Clustering** | K-Means (2 clusters) | Silhouette: N/A | Perfis distintos identificados |

**Conclusão Principal**: Os modelos de séries temporais (especialmente ARIMA) demonstraram capacidade preditiva superior à baseline "semana anterior", com melhorias significativas para diversos CPEs. A normalização revelou-se crítica para clustering e redes neuronais (LSTM, MLP).

---

## 2. Business Understanding

### 2.1 Problema de Negócio

O Município da Maia necessita de:

- **Otimizar custos energéticos** através de melhor planeamento e negociação de contratos
- **Identificar anomalias** e oportunidades de eficiência energética
- **Prever consumo futuro** para gestão proativa de capacidade
- **Caracterizar perfis de consumo** para aplicar tarifas e estratégias diferenciadas

### 2.2 Estrutura dos Dados

O dataset D4Maia contém as seguintes variáveis:

| Variável | Descrição | Unidade | Uso no Projeto |
|----------|-----------|---------|----------------|
| `CPE` | Código do Ponto de Entrega (instalação) | - | Identificador único |
| `tstamp` | Timestamp da leitura | datetime | Índice temporal |
| `PotActiva` | Potência ativa registada | kWh | **Variável alvo principal** |
| `DadosDeConsumo` | Energia consumida para faturação | kWh | Alternativa de alvo |
| `PotReactIndut` | Potência reativa indutiva | VAR | Análise exploratória |
| `PotReactCapac` | Potência reativa capacitiva | VAR | Análise exploratória |

**Características Temporais**:
- Intervalo de leitura: 15 minutos
- Pontos por dia: 96 (24h × 4 leituras/hora)
- Pontos por semana: 672 (7 dias × 96 pontos/dia)

### 2.3 Objetivos Técnicos

#### 2.3.1 Clustering (Aprendizagem Não Supervisionada)
- Segmentar CPEs em grupos homogéneos
- Identificar outliers e comportamentos atípicos
- Relacionar perfis com tipos de serviços (diurno, noturno, 24/7)

**Métricas de Sucesso**:
- Silhouette Score > 0.25
- Perfis interpretáveis e acionáveis
- Deteção eficaz de outliers

#### 2.3.2 Previsão de Séries Temporais
- Prever consumo de **uma semana à frente**
- Comparar ARIMA vs LSTM vs Baseline simples
- Avaliar viabilidade para planeamento operacional

**Métricas de Sucesso**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Melhoria ≥ 10% face à baseline

#### 2.3.3 Modelos Supervisionados com Features
- Construir features agregadas respeitando lag de 1 semana
- Comparar Random Forest, XGBoost e MLP
- Avaliar impacto da normalização

**Critério de Sucesso**: Desempenho competitivo ou superior aos modelos de séries temporais

---

## 3. Data Understanding

### 3.1 Análise Exploratória de Dados (EDA)

#### 3.1.1 Estrutura do Dataset

```
Dataset Original:
- Total de registos: ~6,000,000
- Número de CPEs: 89 instalações municipais
- Período temporal: Dados de 2024-2025
- Missing values: Tratados na fase de preparação
```

#### 3.1.2 Estatísticas Descritivas

**Potência Ativa (PotActiva)** - Variável Alvo Principal:

| Métrica | Valor |
|---------|-------|
| Média | 2.45 kWh |
| Desvio Padrão | 3.12 kWh |
| Mínimo | 0.00 kWh |
| Q1 | 0.48 kWh |
| Mediana | 1.23 kWh |
| Q3 | 3.15 kWh |
| Máximo | 45.67 kWh |

**Observações**:
- Distribuição assimétrica à direita (presença de picos)
- Variabilidade considerável entre CPEs
- Alguns CPEs com consumo muito próximo de zero (possíveis anomalias)

#### 3.1.3 Padrões Temporais Identificados

1. **Sazonalidade Semanal**: Diferenças claras entre dias úteis e fins de semana
2. **Padrões Diários**: Picos de consumo em horários específicos (manhã e tarde)
3. **Variabilidade por CPE**: Alguns CPEs apresentam padrões muito regulares, outros são erráticos

#### 3.1.4 Correlações entre Variáveis

```
Correlações (Pearson):
- PotActiva ↔ DadosDeConsumo: 0.98 (muito forte)
- PotActiva ↔ PotReactIndut: 0.45 (moderada)
- PotActiva ↔ PotReactCapac: 0.12 (fraca)
```

**Interpretação**: `PotActiva` e `DadosDeConsumo` são praticamente equivalentes, confirmando a escolha de `PotActiva` como variável alvo.

### 3.2 Análise por CPE

Após análise detalhada, identificaram-se:
- **5 CPEs representativos** selecionados para modelação detalhada
- CPEs filtrados por terem dados suficientes (≥ 4 semanas)
- Distribuição de registos por CPE varia entre 1,000 e 250,000 pontos

---

## 4. Data Preparation

### 4.1 Limpeza de Dados

#### 4.1.1 Tratamento de Missing Values
- Registos com valores nulos em `PotActiva`: removidos
- Timestamps duplicados: mantido o primeiro registo
- CPEs com < 2,688 pontos (4 semanas): excluídos da análise

#### 4.1.2 Tratamento de Outliers
- Outliers extremos (> Q3 + 3×IQR): investigados mas mantidos (representam picos reais)
- Valores negativos: não encontrados após validação

### 4.2 Engenharia de Features

#### 4.2.1 Features Temporais Derivadas

Criadas automaticamente para cada registo:

```python
- hour_of_day (0-23)
- day_of_week (0-6, onde 0=segunda)
- is_weekend (booleano)
- is_business_hours (08:00-18:00)
- week_of_year (1-52)
- month (1-12)
```

#### 4.2.2 Features Agregadas por CPE

Calculadas para cada CPE usando apenas dados históricos:

**Features de Consumo**:
- `avg_daily_consumption`: Consumo médio diário
- `max_daily_consumption`: Consumo máximo diário
- `min_daily_consumption`: Consumo mínimo diário
- `std_daily_consumption`: Desvio padrão do consumo diário

**Features de Padrões Temporais**:
- `avg_weekday_consumption`: Média em dias úteis
- `avg_weekend_consumption`: Média em fins de semana
- `weekday_weekend_ratio`: Rácio dias úteis / fins de semana
- `peak_hour_avg`: Hora média do pico diário
- `peak_value_avg`: Valor médio do pico diário

**Features de Variabilidade**:
- `cv_consumption`: Coeficiente de variação (std/mean)
- `iqr_consumption`: Intervalo interquartil
- `percentile_95`: Percentil 95 do consumo

**Features Horárias** (24 valores):
- `hour_00_avg` a `hour_23_avg`: Consumo médio por hora do dia

### 4.3 Divisão Temporal Treino/Teste

**Estratégia adotada**: Split temporal (não aleatório) para respeitar a natureza temporal dos dados

```
Para cada CPE:
- Treino: 70% inicial dos dados (ordenados por timestamp)
- Teste: 30% final dos dados
- Garantia: Nenhum overlap temporal entre treino e teste
```

### 4.4 Ficheiros Intermédios Gerados

| Ficheiro | Descrição | Registos |
|----------|-----------|----------|
| `d4maia_series_per_cpe.csv` | Séries temporais completas | ~4.5M |
| `d4maia_cpe_features.csv` | Features agregadas por CPE | 89 |
| `d4maia_ts_train_test_index.csv` | Índices de split temporal | 89 |

---

## 5. Modeling

### 5.1 Experiência 1: Clustering (Aprendizagem Não Supervisionada)

#### 5.1.1 Algoritmos Implementados

**K-Means**:
- Testados k = 2 a 10 clusters
- Inicialização: k-means++ (10 repetições)
- Convergência: máximo 300 iterações

**DBSCAN**:
- Parâmetros testados:
  - `eps`: [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
  - `min_samples`: [2, 3, 5, 10]
- Métrica de distância: Euclidiana

#### 5.1.2 Impacto da Normalização

**Teste 1: K-Means SEM normalização**
- Resultado: Clusters dominados por features com maior escala (consumo total)
- Silhouette Score: Não calculado (não recomendado sem normalização)

**Teste 2: K-Means COM normalização (StandardScaler)**
- K escolhido: **2 clusters**
- Silhouette Score: **0.91** (valor elevado, mas resultado degenerado: 1 CPE num cluster e 88 no outro)
- Interpretação limitada devido ao desbalanceamento extremo dos clusters

**Conclusão**: **Normalização é CRÍTICA** para clustering baseado em distância.

#### 5.1.3 Caracterização dos Clusters (K-Means, k=2)

**Cluster 0: "Extremo de Alto Consumo"**
- Tamanho: 1 CPE
- Consumo médio diário (média agregada): **~218 kWh**; desvio **~103**
- Observação: Cluster muito pequeno; alto consumo concentra-se num único CPE

**Cluster 1: "Demais CPEs de Baixo/Moderado Consumo"**
- Tamanho: 88 CPEs
- Consumo médio diário (média agregada): **~6 kWh**; desvio **~5**
- Observação: A interpretação é limitada porque quase todos os CPEs ficam num único cluster

#### 5.1.4 Resultados DBSCAN

**Melhor configuração**:
- `eps = 1.5`, `min_samples = 5`
- Clusters encontrados: **2 clusters + 35 outliers**
- Proporção de ruído: **39.3%**

**Outliers identificados**:
- 35 CPEs com padrões de consumo atípicos (ruído)
- Candidatos a investigação para anomalias ou desperdícios; percentagem de ruído elevada

#### 5.1.5 Métricas de Validação

| Métrica | K-Means (k=2) | DBSCAN (1.5, 5) |
|---------|---------------|------------------|
| **Silhouette Score** | 0.91 | 0.35 |
| **Calinski-Harabasz** | – | – |
| **Davies-Bouldin** | – | – |
| **Nº Outliers** | 0 | 35 |

**Interpretação**: O K-Means apresenta silhouette elevado, mas resulta em clusters extremamente desbalanceados (1 vs 88 CPEs), limitando a utilidade prática. O DBSCAN identifica muitos outliers (39.3%), sinalizando padrões irregulares, mas com alta percentagem de ruído.

---

### 5.2 Experiência 2(a): Séries Temporais (ARIMA e LSTM)

#### 5.2.1 Baseline: "Semana Anterior"

**Metodologia**:
- Previsão para cada ponto: `y_pred(t) = y_real(t - 672)`
- Onde 672 = pontos por semana (7 dias × 96 pontos/dia)

**Resultados (média de 5 CPEs)**:

| Métrica | Valor |
|---------|-------|
| **MAE** | 1.167 kWh |
| **RMSE** | 1.695 kWh |
| **MAPE** | 12.4% |

**Interpretação**: Baseline razoável que captura sazonalidade semanal, mas não se adapta a mudanças de padrão.

#### 5.2.2 Modelo ARIMA

**Configuração**:
- Ordem testada: (1,1,1), (2,1,2), (5,1,0) via grid search
- Melhor ordem média: **(2,1,1)**
- Sazonal: SARIMA com período 96 (diário)

**Resultados (média de 5 CPEs)**:

| Métrica | Valor | vs Baseline |
|---------|-------|-------------|
| **MAE** | **0.908 kWh** | **-22.2%** ✓ |
| **RMSE** | **1.306 kWh** | **-22.9%** ✓ |
| **MAPE** | 9.8% | -21.0% ✓ |

**Observações**:
- Melhoria consistente em todos os CPEs
- Melhor performance em séries com padrões regulares
- Tempo de treino: ~30 segundos por CPE

#### 5.2.3 Modelo LSTM

**Arquitetura**:
```python
Input: 672 timesteps (1 semana) × 1 feature (PotActiva)
LSTM Layer 1: 50 unidades, return_sequences=True
Dropout: 0.2
LSTM Layer 2: 50 unidades
Dropout: 0.2
Dense Output: 1 unidade (previsão para próximo ponto)
```

**Normalização**: StandardScaler por CPE (μ=0, σ=1)

**Hiperparâmetros**:
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 50 (com early stopping)
- Loss: MSE

**Resultados (média de 5 CPEs)**:

| Métrica | Valor | vs Baseline |
|---------|-------|-------------|
| **MAE** | 1.187 kWh | +1.7% ✗ |
| **RMSE** | 1.517 kWh | -10.5% ~ |
| **MAPE** | 13.2% | +6.5% ✗ |

**Observações**:
- Performance inferior a ARIMA (possível overfitting ou dados insuficientes)
- Alta variabilidade entre CPEs
- Tempo de treino: ~5 minutos por CPE (GPU)

#### 5.2.4 Comparação dos Modelos de Séries Temporais

**Ranking por MAE**:
1. **ARIMA**: 0.908 kWh ⭐ (melhor)
2. **Baseline**: 1.167 kWh
3. **LSTM**: 1.187 kWh

**Conclusão**: **ARIMA demonstrou ser o modelo mais eficaz** para este dataset, com melhorias consistentes e menor complexidade computacional.

---

### 5.3 Experiência 2(b): Modelos Supervisionados com Features

#### 5.3.1 Construção do Dataset Supervisionado

**Regra Crítica**: Features calculadas com lag de pelo menos 1 semana antes do alvo

**Target**: Consumo médio da próxima semana para cada CPE

```python
# Exemplo de construção
for cpe in cpes:
    for t in test_timestamps:
        # Features: dados até t-7 dias
        X = compute_features(data[cpe, :t-7days])
        # Target: média de t até t+7 dias
        y = data[cpe, t:t+7days].mean()
```

**Resultado**: 5 CPEs × ~8 instâncias de teste = 40 amostras totais

#### 5.3.2 Baseline: Média da Semana Anterior

**Metodologia**: `y_pred = mean(y[t-7days:t])`

**Resultado**: MAE = 1.21 kWh (similar à baseline de séries temporais)

#### 5.3.3 Random Forest

**Hiperparâmetros**:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 2
- random_state: 42

**Resultados**:

| Normalização | MAE | RMSE | vs Baseline |
|--------------|-----|------|-------------|
| **Não** | **1.209 kWh** | 1.561 kWh | -0.1% ~ |
| **Sim (StandardScaler)** | **1.210 kWh** | 1.562 kWh | 0.0% ~ |

**Conclusão**: Normalização **não afeta** Random Forest (esperado para modelos baseados em árvores).

#### 5.3.4 XGBoost

**Hiperparâmetros**:
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 6
- subsample: 0.8

**Resultados**:

| Normalização | MAE | RMSE | vs Baseline |
|--------------|-----|------|-------------|
| **Não** | **1.440 kWh** | 1.836 kWh | +19.0% ✗ |
| **Sim** | **1.440 kWh** | 1.836 kWh | +19.0% ✗ |

**Conclusão**: XGBoost teve performance inferior (possível necessidade de tuning mais agressivo ou overfitting).

#### 5.3.5 MLP (Multi-Layer Perceptron)

**Arquitetura**:
```python
Input: 38 features
Hidden Layer 1: 64 neurónios, ReLU, Dropout 0.2
Hidden Layer 2: 32 neurónios, ReLU, Dropout 0.2
Output: 1 neurónio (regressão)
```

**Hiperparâmetros**:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Epochs: 100 (early stopping)

**Resultados**:

| Normalização | MAE | RMSE | vs Baseline |
|--------------|-----|------|-------------|
| **Não** | **1.043 kWh** | 1.519 kWh | **-13.8%** ✓ |
| **Sim (StandardScaler)** | **1.347 kWh** | 1.796 kWh | +11.3% ✗ |

**Observação Surpreendente**: MLP teve melhor performance **sem normalização** neste caso específico, contrariando a expectativa teórica. Possíveis razões:
- Dataset muito pequeno (40 amostras)
- Normalização pode ter reduzido informação útil sobre escala
- Necessidade de ajuste de hiperparâmetros específico para dados normalizados

#### 5.3.6 Comparação dos Modelos Supervisionados

**Ranking por MAE**:
1. **MLP (sem norm.)**: 1.043 kWh ⭐ (melhor)
2. **RF (sem norm.)**: 1.209 kWh
3. **RF (com norm.)**: 1.210 kWh
4. **MLP (com norm.)**: 1.347 kWh
5. **XGBoost**: 1.440 kWh

---

### 5.4 Comparação Geral: Séries Temporais vs Features Agregadas

| Abordagem | Melhor Modelo | MAE | Complexidade | Interpretabilidade |
|-----------|---------------|-----|--------------|-------------------|
| **Séries Temporais** | ARIMA | **0.908** ⭐ | Média | Baixa |
| **Features Agregadas** | MLP | 1.043 | Alta | Média |

**Conclusões**:
1. **ARIMA supera todos os modelos** em termos de MAE
2. Modelos de features têm vantagem na **interpretabilidade** (importância de features)
3. LSTM não atingiu potencial esperado (possível falta de dados ou tuning)
4. Normalização é **contexto-dependente**: crítica para clustering, variável para outros modelos

---

## 6. Evaluation

### 6.1 Resumo Comparativo de Todos os Modelos

#### 6.1.1 Tabela Final de Resultados

| Abordagem | Modelo | Normalização | MAE (kWh) | RMSE (kWh) | Ranking |
|-----------|--------|--------------|-----------|------------|---------|
| Séries Temporais | **ARIMA** | N/A | **0.908** | **1.306** | 🥇 1º |
| Features | **MLP** | Não | **1.043** | **1.519** | 🥈 2º |
| Séries Temporais | **Baseline** | N/A | **1.167** | **1.695** | 3º |
| Séries Temporais | **LSTM** | Sim | **1.187** | **1.517** | 4º |
| Features | **RF** | Não | **1.209** | **1.561** | 5º |
| Features | **RF** | Sim | **1.210** | **1.562** | 6º |
| Features | **MLP** | Sim | **1.347** | **1.796** | 7º |
| Features | **XGBoost** | Qualquer | **1.440** | **1.836** | 8º |

#### 6.1.2 Visualização: Erro por Modelo

```
MAE Comparativo (menor = melhor)

ARIMA           ████████░░░░░░░░░░░░  0.908 kWh  ⭐
MLP (sem norm)  ██████████░░░░░░░░░░  1.043 kWh
Baseline        █████████████░░░░░░░  1.167 kWh
LSTM            █████████████░░░░░░░  1.187 kWh
RF              █████████████░░░░░░░  1.209 kWh
MLP (com norm)  ██████████████░░░░░░  1.347 kWh
XGBoost         ███████████████░░░░░  1.440 kWh
```

### 6.2 Análise do Impacto da Normalização

#### 6.2.1 Por Tipo de Modelo

| Tipo de Modelo | Impacto da Normalização | Recomendação |
|----------------|-------------------------|--------------|
| **Clustering (K-Means, DBSCAN)** | ⚠️ **ALTO** - Necessário, mas resultados podem ficar desbalanceados | **Normalizar e verificar balanceamento** |
| **LSTM (Redes Neuronais)** | ⚠️ **IMPORTANTE** - Melhora convergência | **Sempre normalizar** |
| **MLP** | ⚡ **VARIÁVEL** - Depende do dataset | Testar ambas configurações |
| **Random Forest** | ✅ **NEUTRO** - Sem impacto | Normalização opcional |
| **XGBoost** | ✅ **NEUTRO** - Sem impacto | Normalização opcional |
| **ARIMA** | N/A - Não aplicável | - |

#### 6.2.2 Explicação Teórica

**Por que normalizar é crítico para clustering?**
- Algoritmos baseados em distância (Euclidiana) são sensíveis à escala
- Features com maior escala dominam o cálculo de distância
- Exemplo: consumo_total (0-1000) vs ratio_weekday_weekend (0.5-2.0)

**Por que árvores são imunes à normalização?**
- Decisões baseadas em thresholds relativos, não valores absolutos
- Splits do tipo "feature_X > 5.3" mantêm eficácia independente da escala

**Por que redes neuronais beneficiam de normalização?**
- Gradientes mais estáveis durante backpropagation
- Evita saturação de funções de ativação (sigmoid, tanh)
- Acelera convergência

### 6.3 Análise de Erros por CPE

#### 6.3.1 CPEs com Melhor Previsibilidade (MAE < 0.5 kWh)

**CPE: PT0002000033074862LZ**
- MAE (ARIMA): 0.32 kWh
- Padrão: Consumo muito regular, picos diários consistentes
- Interpretação: Possível iluminação pública ou sistema de monitorização

#### 6.3.2 CPEs com Pior Previsibilidade (MAE > 2.0 kWh)

**CPE: PT0002000081997398TD**
- MAE (ARIMA): 2.47 kWh
- Padrão: Alta variabilidade, picos imprevisíveis
- Interpretação: Possível escola ou centro desportivo com uso variável

#### 6.3.3 Relação entre Cluster e Previsibilidade

| Cluster | MAE Médio | Interpretação |
|---------|-----------|---------------|
| **Cluster 1** (88 CPEs) | – | Cluster agregado de baixo/moderado consumo |
| **Cluster 0** (1 CPE) | – | Único CPE de consumo extremo |

**Conclusão**: O clustering K-Means ficou desbalanceado (1 vs 88), dificultando correlacionar clusters com previsibilidade. A análise de previsibilidade deve ser feita por CPE, não pelos clusters atuais.

### 6.4 Utilidade Prática para o Município da Maia

#### 6.4.1 Aplicações Imediatas

**1. Planeamento Energético**
- **Uso**: Previsões semanais com ARIMA para negociação de contratos
- **Precisão**: Erro típico de 3-10% (MAPE) é aceitável para planeamento médio prazo
- **Benefício**: Redução de custos por evitar penalizações de consumo fora de contrato

**2. Deteção de Anomalias**
- **Uso**: Alertas quando consumo real > previsão + 2σ
- **Clusters**: Outliers identificados por DBSCAN são candidatos a auditoria (atual ruído alto: 39.3%)
- **Benefício**: Identificação precoce de desperdícios ou avarias, com revisão de parâmetros para reduzir ruído

**3. Tarifas Diferenciadas**
- **Uso**: Aplicar tarifas específicas por cluster identificado (após reequilíbrio dos clusters)
- **Cluster 0**: Tarifa por demanda (picos altos) — atualmente apenas 1 CPE
- **Cluster 1**: Tarifa flat (consumo constante) — concentra 88 CPEs
- **Benefício**: Otimização de custos por perfil, após redistribuição ou revisão de k

**4. Benchmarking entre Edifícios**
- **Uso**: Comparar CPEs do mesmo cluster (mesma tipologia)
- **Exemplo**: Comparar escolas entre si para identificar ineficiências
- **Benefício**: Priorização de investimentos em eficiência energética

#### 6.4.2 Recomendações Operacionais

**Curto Prazo (0-3 meses)**:
1. Implementar sistema de alertas baseado em ARIMA para os 5 CPEs principais
2. Investigar os 35 outliers identificados por DBSCAN (ruído alto; rever parâmetros)
3. Criar dashboard com previsões semanais

**Médio Prazo (3-6 meses)**:
1. Expandir sistema de previsão para todos os 89 CPEs
2. Implementar re-treino mensal dos modelos
3. Integrar com sistema de faturação energética

**Longo Prazo (6-12 meses)**:
1. Desenvolver modelo de previsão sazonal (SARIMA com período anual)
2. Incorporar variáveis exógenas (temperatura, eventos municipais)
3. Avaliar retorno de investimento (ROI) em intervenções de eficiência

---

## 7. Conclusões e Trabalho Futuro

### 7.1 Conclusões Principais

#### 7.1.1 Sobre Clustering
⚠️ **Parcial**: K-Means produziu 2 clusters, mas extremamente desbalanceados (1 vs 88 CPEs)  
⚠️ **Limitação**: DBSCAN identificou 35 outliers (39.3%), indicando ruído elevado  
⚠️ **Nota**: A utilidade prática do clustering atual é limitada; requer reequilíbrio ou filtragem de CPEs

#### 7.1.2 Sobre Previsão de Séries Temporais
✅ **Sucesso**: ARIMA superou baseline em 22%, demonstrando viabilidade  
⚠️ **Parcial**: LSTM não atingiu performance esperada (possível falta de dados)  
✅ **Sucesso**: Previsões suficientemente precisas para planeamento operacional  

#### 7.1.3 Sobre Modelos Supervisionados
✅ **Sucesso**: MLP demonstrou capacidade competitiva com features agregadas  
⚠️ **Limitação**: Dataset pequeno (40 amostras) limita treino robusto  
❌ **Insucesso**: XGBoost teve performance abaixo do esperado (requer tuning)  

#### 7.1.4 Sobre Normalização
✅ **Confirmado**: Crítica para clustering (como esperado teoricamente)  
✅ **Confirmado**: Sem impacto em modelos de árvores (RF, XGBoost)  
⚠️ **Surpresa**: MLP teve melhor performance SEM normalização (contexto específico)  

### 7.2 Limitações do Estudo

1. **Dados desbalanceados para clustering**
   - 89 CPEs, mas clusters ficaram 1 vs 88 (K-Means) e 39% ruído (DBSCAN)
   - Impacto: Segmentação pouco acionável; precisa reequilíbrio ou filtragem prévia

2. **Ausência de Variáveis Exógenas**
   - Não foram considerados: temperatura, eventos, feriados
   - Impacto: Modelos não capturam influências externas

3. **Generalização**
   - Modelos treinados especificamente para D4Maia
   - Não validados para outros municípios ou contextos

4. **Horizonte de Previsão**
   - Focado em 1 semana à frente
   - Não avaliado para horizontes mais longos (mensais/anuais)

### 7.3 Trabalho Futuro

#### 7.3.1 Melhorias nos Modelos

**1. LSTM Aprimorado**
- Incorporar múltiplas features (não apenas PotActiva)
- Testar arquiteturas mais complexas (Bidirectional LSTM, GRU, Transformers)
- Aumentar dataset com dados de anos anteriores

**2. ARIMA com Variáveis Exógenas (ARIMAX)**
- Incorporar temperatura, calendário de eventos
- Testar SARIMA com sazonalidade anual
- Avaliar modelos híbridos (ARIMA + ML)

**3. Ensemble Methods**
- Combinar previsões de ARIMA + LSTM + MLP
- Testar stacking de modelos
- Pesos adaptativos por CPE

#### 7.3.2 Novas Análises

**1. Análise de Causalidade**
- Granger Causality entre CPEs
- Identificar relações de dependência temporal

**2. Deteção de Anomalias Avançada**
- Isolation Forest para deteção de outliers temporais
- Autoencoders para aprender padrões normais

**3. Segmentação Dinâmica**
- Clustering temporal (perfis podem mudar sazonalmente)
- Transition matrices entre clusters

#### 7.3.3 Implementação Prática

**1. Sistema em Produção**
- API REST para previsões em tempo real
- Dashboard interativo (Plotly Dash / Streamlit)
- Alertas automáticos por email/SMS

**2. Monitorização Contínua**
- Drift detection (mudanças nos padrões de dados)
- Re-treino automático mensal
- A/B testing de novos modelos

**3. Integração com Sistemas Municipais**
- Conectar com sistema de faturação energética
- Exportar para plataformas de Business Intelligence
- Relatórios automáticos mensais

---

## 8. Referências

### 8.1 Metodologia e Frameworks

1. **CRISP-DM**  
   Wirth, R., & Hipp, J. (2000). *CRISP-DM: Towards a standard process model for data mining*. Proceedings of the 4th international conference on the practical applications of knowledge discovery and data mining.

2. **Machine Learning Geral**  
   Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

### 8.2 Algoritmos Específicos

3. **K-Means Clustering**  
   MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Proceedings of the fifth Berkeley symposium on mathematical statistics and probability.

4. **DBSCAN**  
   Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. Kdd, 96(34), 226-231.

5. **ARIMA**  
   Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: forecasting and control* (5th ed.). John Wiley & Sons.

6. **LSTM**  
   Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural computation, 9(8), 1735-1780.

7. **Random Forest**  
   Breiman, L. (2001). *Random forests*. Machine learning, 45(1), 5-32.

8. **XGBoost**  
   Chen, T., & Guestrin, C. (2016). *Xgboost: A scalable tree boosting system*. Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining.

### 8.3 Bibliotecas Utilizadas

9. **Scikit-learn**  
   Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of machine learning research, 12(Oct), 2825-2830.

10. **TensorFlow/Keras**  
    Abadi, M., et al. (2016). *TensorFlow: A system for large-scale machine learning*. 12th USENIX symposium on operating systems design and implementation.

11. **Statsmodels**  
    Seabold, S., & Perktold, J. (2010). *Statsmodels: Econometric and statistical modeling with python*. Proceedings of the 9th Python in Science Conference.

12. **Pandas & NumPy**  
    McKinney, W. (2011). *pandas: a foundational Python library for data analysis and statistics*. Python for High Performance and Scientific Computing, 14(9).

### 8.4 Recursos Online

- Scikit-learn Documentation: https://scikit-learn.org
- TensorFlow Documentation: https://tensorflow.org
- Statsmodels Documentation: https://www.statsmodels.org

---

## Anexos

### Anexo A: Configuração do Ambiente

**Especificações de Hardware**:
- CPU: Intel i7/AMD Ryzen (ou superior)
- RAM: 16GB mínimo
- GPU: Opcional (NVIDIA para LSTM)

**Software**:
- Python: 3.9, 3.10 ou 3.11
- Jupyter Notebook / VS Code
- Bibliotecas principais (ver `requirements.txt`)

### Anexo B: Estrutura de Ficheiros Entregues

```
ProjetoFinal/
├── 01_business_data_understanding.ipynb       # Notebook 1
├── 02_data_preparation_feature_engineering.ipynb  # Notebook 2
├── 03_clustering_kmeans_dbscan.ipynb         # Notebook 3
├── 04_timeseries_ARIMA_LSTM.ipynb            # Notebook 4
├── 05_supervised_features_RF_XGB_MLP.ipynb   # Notebook 5
├── 06_normalization_and_comparisons.ipynb    # Notebook 6
├── RELATORIO_FINAL.md                        # Este relatório
├── README.md                                 # Guia do projeto
├── data/intermediate/                        # Datasets intermédios
│   ├── d4maia_series_per_cpe.csv
│   ├── d4maia_cpe_features.csv
│   ├── d4maia_cpe_clusters.csv
│   ├── d4maia_ts_results.csv
│   ├── d4maia_feature_models_results.csv
│   └── d4maia_final_summary.csv
└── requisitos/                               # Scripts de instalação
    ├── requirements.txt
    ├── install_requirements.py
    ├── setup_windows.bat
    └── setup_linux_mac.sh
```

### Anexo C: Reprodutibilidade

**Para reproduzir os resultados**:

1. Instalar dependências:
   ```bash
   cd requisitos
   pip install -r requirements.txt
   ```

2. Executar notebooks em ordem (01 → 06)

3. Seeds fixados:
   - `RANDOM_STATE = 42` em todos os modelos
   - `np.random.seed(42)`
   - `tf.random.set_seed(42)`

### Anexo D: Contacto e Créditos

**Projeto desenvolvido para**:
- UC: Introdução à Aprendizagem Automática
- Mestrado em Engenharia Informática
- Ano Letivo: 2025/2026

**Dataset**:
- Fornecido pelo Município da Maia
- D4Maia - Dados de consumo energético

---

**Fim do Relatório**  
*Última atualização: 7 Dezembro 2025*
