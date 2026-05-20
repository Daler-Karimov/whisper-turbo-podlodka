@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ==================================================
echo   Whisper Multi-Model Transcription (CPU)
echo ==================================================
echo.

if not exist "models" (
    echo [INFO] Creating models folder...
    mkdir models
)

set "HF_HOME=%~dp0models"
set "HF_HUB_CACHE=%~dp0models"
set "TRANSFORMERS_CACHE=%~dp0models"
set "WHISPER_MODELS_DIR=%~dp0models"

echo [INFO] Python:
python --version
echo.

echo [INFO] Device check:
python -c "import torch; print('  PyTorch:', torch.__version__); c=torch.cuda.is_available(); print('  CUDA:', 'yes' if c else 'no (CPU mode)'); print('  GPU:', torch.cuda.get_device_name(0)) if c else print('  Mode: CPU (float32)')"
echo.

echo --------------------------------------------------
echo [INFO] Models dir: %~dp0models
echo [INFO] Starting app...
echo [INFO] Open in browser: http://localhost:7860
echo.
echo [!] First run will download model (1-6 GB).
echo [!] CPU transcription is slower than GPU.
echo --------------------------------------------------
echo.

python app.py

echo.
echo [INFO] App stopped.
pause