@echo off
chcp 65001 >nul
echo Installing dependencies...
python -m pip install -r requirements.txt -q
echo =========================================
echo Starting MC Translator...
python -m mc_translator
pause
