#!/usr/bin/env python3
"""
Script para verificação de qualidade de código do projeto D4Maia.

Executa verificações automáticas de:
- Formatação de código (Black)
- Estilo de código (Flake8) 
- Análise estática (Pylint)
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Executa um comando e reporta o resultado."""
    print(f"\n{description}")
    print("-" * len(description))
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("[OK] OK")
            return True
        else:
            print("[WARN]  PROBLEMAS ENCONTRADOS")
            if result.stdout:
                # Mostrar apenas as primeiras linhas de erro
                lines = result.stdout.split('\n')[:10]
                for line in lines:
                    if line.strip():
                        print(f"  {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]  TIMEOUT")
        return False
    except Exception as e:
        print(f"[ERROR] ERRO: {e}")
        return False

def main():
    """Função principal."""
    print("VERIFICAÇÃO DE QUALIDADE - PROJETO D4MAIA")
    print("=" * 50)
    
    # Verificar se estamos no diretório correto
    if not Path("03_clustering_kmeans_dbscan.ipynb").exists():
        print("[ERROR] Execute este script do diretório 'ProjetoFinal'")
        sys.exit(1)
    
    notebooks = [
        "03_clustering_kmeans_dbscan.ipynb",
        "05_supervised_features_RF_XGB_MLP.ipynb", 
        "06_normalization_and_comparisons.ipynb"
    ]
    
    success_count = 0
    
    # 1. Verificação de formatação com Black
    if run_command("black --check --diff --line-length=88 *.ipynb", 
                   "1. Verificando formatação (Black)"):
        success_count += 1
    
    # 2. Verificação de estilo com Flake8 (mais permissivo para notebooks)
    flake8_cmd = "flake8 --max-line-length=120 --extend-ignore=E203,W503,E501,W292,F821"
    if run_command(f"{flake8_cmd} {' '.join(notebooks)}",
                   "2. Verificando estilo (Flake8)"):
        success_count += 1
    
    # 3. Análise com Pylint (apenas no notebook modificado, mais permissivo para notebooks)
    pylint_cmd = "pylint --disable=C0114,C0115,C0116,R0903,R0912,R0915,C0103,W0613,W0611,C0301,C0304,C0302,W0104,E0602"
    if run_command(f"{pylint_cmd} 05_supervised_features_RF_XGB_MLP.ipynb",
                   "3. Análise estática (Pylint)"):
        success_count += 1
    
    print(f"\n{'=' * 50}")
    print(f"RESULTADO: {success_count}/3 verificações passaram")
    
    if success_count == 3:
        print("[SUCCESS] Codigo esta em conformidade com os padroes de qualidade!")
        return 0
    else:
        print("[WARN]  Algumas verificações falharam. Execute 'python check_quality.py --fix' para corrigir automaticamente.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
