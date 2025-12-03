#!/bin/bash
# =============================================================================
# D4Maia Project - Setup Linux/macOS (Bash)
# =============================================================================
# Uso: chmod +x setup_linux_mac.sh && ./setup_linux_mac.sh
# =============================================================================

echo ""
echo "============================================================"
echo "     D4Maia Project - Instalação de Requisitos (Unix)"
echo "============================================================"
echo ""

# Cores
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERRO]${NC} Python3 não encontrado!"
    echo "Por favor, instale Python 3.9+ usando:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  macOS: brew install python3"
    exit 1
fi

echo -e "${CYAN}[INFO]${NC} Python encontrado:"
python3 --version
echo ""

# Perguntar sobre ambiente virtual
read -p "Criar ambiente virtual? (s/n): " USEVENV

if [[ "$USEVENV" == "s" || "$USEVENV" == "S" ]]; then
    echo -e "${CYAN}[INFO]${NC} Criando ambiente virtual..."
    python3 -m venv venv_d4maia
    
    echo -e "${CYAN}[INFO]${NC} Ativando ambiente virtual..."
    source venv_d4maia/bin/activate
    
    echo -e "${GREEN}[OK]${NC} Ambiente virtual ativado!"
    echo ""
fi

# Atualizar pip
echo -e "${CYAN}[INFO]${NC} Atualizando pip..."
pip install --upgrade pip

# Instalar requirements
echo ""
echo -e "${CYAN}[INFO]${NC} Instalando requisitos do projeto..."
echo ""

pip install -r requirements.txt

echo ""
echo "============================================================"
echo -e "${GREEN}     Instalação Concluída!${NC}"
echo "============================================================"
echo ""
echo "Próximos passos:"
echo "  1. Abrir pasta ProjetoFinal no VS Code ou Jupyter"
echo "  2. Executar notebooks: 01, 02, 03, 04, 05, 06"
echo "  3. Dados de entrada: consumo15m_11_2025.csv"
echo ""

if [[ "$USEVENV" == "s" || "$USEVENV" == "S" ]]; then
    echo "Para ativar o ambiente virtual no futuro:"
    echo "  source venv_d4maia/bin/activate"
    echo ""
fi

echo "============================================================"
