@echo off
REM =============================================================================
REM D4Maia Project - Setup Windows (Batch)
REM =============================================================================
REM Uso: Duplo clique ou executar em cmd.exe
REM =============================================================================

echo.
echo ============================================================
echo      D4Maia Project - Instalacao de Requisitos (Windows)
echo ============================================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado!
    echo Por favor, instale Python 3.9+ de https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo [INFO] Python encontrado:
python --version
echo.

REM Perguntar sobre ambiente virtual
set /p USEVENV="Criar ambiente virtual? (s/n): "
if /i "%USEVENV%"=="s" (
    echo [INFO] Criando ambiente virtual...
    python -m venv venv_d4maia
    echo [INFO] Ativando ambiente virtual...
    call venv_d4maia\Scripts\activate.bat
    echo [OK] Ambiente virtual ativado!
    echo.
)

REM Atualizar pip
echo [INFO] Atualizando pip...
python -m pip install --upgrade pip

REM Instalar requirements
echo.
echo [INFO] Instalando requisitos do projeto...
echo.

pip install -r requirements.txt

echo.
echo ============================================================
echo      Instalacao Concluida!
echo ============================================================
echo.
echo Proximos passos:
echo   1. Abrir pasta ProjetoFinal no VS Code
echo   2. Executar notebooks: 01, 02, 03, 04, 05, 06
echo   3. Dados de entrada: consumo15m_11_2025.csv
echo.

if /i "%USEVENV%"=="s" (
    echo Para ativar o ambiente virtual no futuro:
    echo   venv_d4maia\Scripts\activate
    echo.
)

pause
