# 📦 Requisitos do Projeto D4Maia

## Visão Geral

Esta pasta contém todos os ficheiros necessários para configurar o ambiente Python e instalar as dependências do projeto D4Maia.

## 📁 Ficheiros Disponíveis

| Ficheiro | Descrição |
|----------|-----------|
| `requirements.txt` | Lista de pacotes Python necessários |
| `install_requirements.py` | Script Python com opções avançadas |
| `setup_windows.bat` | Script de setup para Windows |
| `setup_linux_mac.sh` | Script de setup para Linux/macOS |

---

## 🚀 Instalação Rápida

### Windows

**Opção 1 - Duplo clique:**
```
Duplo clique em setup_windows.bat
```

**Opção 2 - PowerShell:**
```powershell
cd ProjetoFinal\requisitos
pip install -r requirements.txt
```

### Linux/macOS

```bash
cd ProjetoFinal/requisitos
chmod +x setup_linux_mac.sh
./setup_linux_mac.sh
```

---

## 🐍 Script Python Avançado

O script `install_requirements.py` oferece opções adicionais:

```bash
# Verificar requisitos (sem instalar)
python install_requirements.py --check

# Instalação básica
python install_requirements.py

# Com ambiente virtual
python install_requirements.py --venv

# Atualizar para últimas versões
python install_requirements.py --upgrade

# Com suporte GPU (NVIDIA)
python install_requirements.py --gpu
```

---

## 📋 Dependências

### Core (Obrigatórias)

| Pacote | Versão Mínima | Uso |
|--------|---------------|-----|
| pandas | 2.0.0 | Manipulação de dados |
| numpy | 1.24.0 | Computação numérica |
| matplotlib | 3.7.0 | Visualização |
| seaborn | 0.12.0 | Visualização estatística |
| scikit-learn | 1.3.0 | Machine Learning |
| statsmodels | 0.14.0 | ARIMA (séries temporais) |
| xgboost | 2.0.0 | Gradient Boosting |
| tensorflow | 2.13.0 | Deep Learning (LSTM) |

### Jupyter

| Pacote | Versão Mínima | Uso |
|--------|---------------|-----|
| jupyter | 1.0.0 | Interface notebooks |
| ipykernel | 6.25.0 | Kernel Jupyter |
| notebook | 7.0.0 | Jupyter Notebook |

---

## ⚠️ Notas Importantes

### Python
- **Versão recomendada:** Python 3.9, 3.10, ou 3.11
- Python 3.12+ pode ter incompatibilidades com TensorFlow

### TensorFlow
- A instalação pode demorar vários minutos
- Para GPUs NVIDIA, usar `--gpu` flag
- Requer CUDA e cuDNN para suporte GPU

### Ambiente Virtual (Recomendado)
Usar ambiente virtual evita conflitos com outros projetos:

```bash
# Criar
python -m venv venv_d4maia

# Ativar (Windows)
venv_d4maia\Scripts\activate

# Ativar (Linux/macOS)
source venv_d4maia/bin/activate

# Instalar requisitos
pip install -r requirements.txt
```

---

## 🔧 Resolução de Problemas

### Erro: "pip not found"
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Erro: TensorFlow não instala
```bash
# Tentar versão específica
pip install tensorflow==2.13.0

# Ou versão CPU-only
pip install tensorflow-cpu
```

### Erro: Conflito de versões
```bash
# Criar ambiente virtual limpo
python -m venv venv_clean
# Ativar e reinstalar
pip install -r requirements.txt
```

---

## 📊 Verificação

Após instalação, verificar se tudo funciona:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels
import tensorflow as tf
import xgboost as xgb

print("✅ Todos os pacotes instalados corretamente!")
print(f"TensorFlow: {tf.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

---

## 📝 Estrutura do Projeto

Após instalação, executar notebooks nesta ordem:

```
ProjetoFinal/
├── 01_data_exploration.ipynb       # Exploração inicial
├── 02_clustering.ipynb             # K-Means e DBSCAN
├── 03_feature_engineering.ipynb    # Engenharia de features
├── 04_time_series_models.ipynb     # ARIMA e LSTM
├── 05_supervised_models.ipynb      # RF, XGBoost, MLP
├── 06_normalization_and_comparisons.ipynb  # Avaliação final
├── data/                           # Dados processados
└── requisitos/                     # Esta pasta
```

---

**Projeto:** D4Maia - Previsão de Consumo Energético  
**UC:** Introdução à Aprendizagem Automática - MEI 2025/2026  
**Metodologia:** CRISP-DM
