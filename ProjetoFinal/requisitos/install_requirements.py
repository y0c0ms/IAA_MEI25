#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
D4Maia Project - Script de Instalação de Requisitos
=============================================================================
Projeto: Previsão de Consumo Energético
UC: Introdução à Aprendizagem Automática - MEI 2025/2026
Metodologia: CRISP-DM
=============================================================================

Este script instala automaticamente todos os requisitos necessários para
executar os notebooks do projeto D4Maia.

Uso:
    python install_requirements.py [opções]

Opções:
    --venv          Criar ambiente virtual antes de instalar
    --upgrade       Atualizar pacotes para última versão
    --gpu           Instalar TensorFlow com suporte GPU
    --check         Apenas verificar requisitos (não instala)
    --help, -h      Mostrar ajuda
"""

import subprocess
import sys
import os
from pathlib import Path


# =============================================================================
# Configuração
# =============================================================================

# Pacotes obrigatórios (core)
CORE_PACKAGES = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scikit-learn>=1.3.0",
    "statsmodels>=0.14.0",
    "xgboost>=2.0.0",
]

# TensorFlow (CPU por padrão)
TENSORFLOW_CPU = ["tensorflow>=2.13.0"]
TENSORFLOW_GPU = ["tensorflow[and-cuda]>=2.13.0"]  # Para GPUs NVIDIA

# Pacotes Jupyter
JUPYTER_PACKAGES = [
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
    "notebook>=7.0.0",
]

# Cores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# =============================================================================
# Funções Auxiliares
# =============================================================================

def print_header(text):
    """Imprime cabeçalho formatado."""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def print_info(text):
    """Imprime informação."""
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {text}")


def print_success(text):
    """Imprime mensagem de sucesso."""
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {text}")


def print_warning(text):
    """Imprime aviso."""
    print(f"{Colors.WARNING}[AVISO]{Colors.ENDC} {text}")


def print_error(text):
    """Imprime erro."""
    print(f"{Colors.FAIL}[ERRO]{Colors.ENDC} {text}")


def run_pip(args, check=True):
    """Executa comando pip."""
    cmd = [sys.executable, "-m", "pip"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def check_package(package_name):
    """Verifica se pacote está instalado."""
    # Extrair apenas o nome do pacote (sem versão)
    name = package_name.split('>=')[0].split('==')[0].split('[')[0]
    try:
        __import__(name.replace('-', '_'))
        return True
    except ImportError:
        return False


def get_installed_version(package_name):
    """Obtém versão instalada de um pacote."""
    name = package_name.split('>=')[0].split('==')[0].split('[')[0]
    success, stdout, _ = run_pip(["show", name], check=False)
    if success and stdout:
        for line in stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    return None


def create_virtual_env():
    """Cria ambiente virtual Python."""
    venv_path = Path.cwd() / "venv_d4maia"
    
    if venv_path.exists():
        print_warning(f"Ambiente virtual já existe: {venv_path}")
        return str(venv_path)
    
    print_info(f"Criando ambiente virtual em: {venv_path}")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    print_success("Ambiente virtual criado com sucesso!")
    
    # Instruções de ativação
    if sys.platform == "win32":
        activate_cmd = f"{venv_path}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_path}/bin/activate"
    
    print_info(f"Para ativar: {activate_cmd}")
    return str(venv_path)


# =============================================================================
# Funções Principais
# =============================================================================

def check_requirements():
    """Verifica estado dos requisitos."""
    print_header("Verificação de Requisitos - D4Maia Project")
    
    all_packages = CORE_PACKAGES + TENSORFLOW_CPU + JUPYTER_PACKAGES
    installed = 0
    missing = 0
    
    print(f"{'Pacote':<30} {'Estado':<15} {'Versão':<15}")
    print("-" * 60)
    
    for package in all_packages:
        name = package.split('>=')[0].split('==')[0].split('[')[0]
        is_installed = check_package(package)
        version = get_installed_version(package) if is_installed else "-"
        
        if is_installed:
            status = f"{Colors.GREEN}Instalado{Colors.ENDC}"
            installed += 1
        else:
            status = f"{Colors.FAIL}Falta{Colors.ENDC}"
            missing += 1
        
        print(f"{name:<30} {status:<24} {version:<15}")
    
    print("-" * 60)
    print(f"\n{Colors.BOLD}Resumo:{Colors.ENDC}")
    print(f"  Instalados: {Colors.GREEN}{installed}{Colors.ENDC}")
    print(f"  Em falta:   {Colors.FAIL}{missing}{Colors.ENDC}")
    
    if missing == 0:
        print_success("\nTodos os requisitos estão instalados!")
        return True
    else:
        print_warning(f"\n{missing} pacote(s) em falta. Execute sem --check para instalar.")
        return False


def install_packages(packages, upgrade=False):
    """Instala lista de pacotes."""
    for package in packages:
        name = package.split('>=')[0].split('==')[0]
        print_info(f"Instalando {name}...")
        
        args = ["install"]
        if upgrade:
            args.append("--upgrade")
        args.append(package)
        
        success, stdout, stderr = run_pip(args, check=False)
        
        if success:
            print_success(f"{name} instalado com sucesso!")
        else:
            print_error(f"Falha ao instalar {name}")
            if stderr:
                print(f"    {stderr[:200]}...")


def install_all(upgrade=False, use_gpu=False):
    """Instala todos os requisitos."""
    print_header("Instalação de Requisitos - D4Maia Project")
    
    # Atualizar pip primeiro
    print_info("Atualizando pip...")
    run_pip(["install", "--upgrade", "pip"], check=False)
    
    # Instalar pacotes core
    print_info("\n--- Pacotes Core ---")
    install_packages(CORE_PACKAGES, upgrade)
    
    # Instalar TensorFlow
    print_info("\n--- TensorFlow ---")
    tf_packages = TENSORFLOW_GPU if use_gpu else TENSORFLOW_CPU
    install_packages(tf_packages, upgrade)
    
    # Instalar Jupyter
    print_info("\n--- Jupyter ---")
    install_packages(JUPYTER_PACKAGES, upgrade)
    
    print_header("Instalação Concluída!")
    
    # Verificar resultado
    check_requirements()


def show_help():
    """Mostra ajuda."""
    print(__doc__)


# =============================================================================
# Main
# =============================================================================

def main():
    """Função principal."""
    args = sys.argv[1:]
    
    # Processar argumentos
    if "--help" in args or "-h" in args:
        show_help()
        return
    
    if "--check" in args:
        check_requirements()
        return
    
    create_venv = "--venv" in args
    upgrade = "--upgrade" in args
    use_gpu = "--gpu" in args
    
    # Criar ambiente virtual se solicitado
    if create_venv:
        venv_path = create_virtual_env()
        print_warning("Ative o ambiente virtual e execute este script novamente.")
        return
    
    # Instalar pacotes
    install_all(upgrade=upgrade, use_gpu=use_gpu)
    
    print("\n" + "="*60)
    print(f"{Colors.GREEN}Projeto D4Maia pronto para execução!{Colors.ENDC}")
    print("="*60)
    print("\nPróximos passos:")
    print("  1. Abrir pasta ProjetoFinal no VS Code ou Jupyter")
    print("  2. Executar notebooks na ordem: 01 → 02 → 03 → 04 → 05 → 06")
    print("  3. Os dados de entrada devem estar em: consumo15m_11_2025.csv")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
