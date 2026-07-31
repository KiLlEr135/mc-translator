@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo =========================================
echo  MC Translator — сборка EXE
echo =========================================
echo.

echo [1/3] Установка и обновление зависимостей...
python -m pip install -r requirements-dev.txt -q
if errorlevel 1 (
    echo Ошибка: не удалось установить пакеты. Проверьте Python 3.10+
    pause
    exit /b 1
)

echo [2/3] Запуск сборки через PyInstaller...
REM MC_Translator.spec is the single source of truth for build options
REM (onefile, noconsole, icon, --add-data, --collect-all ruamel.yaml --
REM see the .spec file's own comment for why that one matters, upx, module
REM excludes). Don't pass conflicting CLI flags (--onefile/--name/--icon/
REM etc.) alongside a .spec file -- PyInstaller reads them from the spec,
REM and a previous version of this script duplicated them here as raw CLI
REM flags, which could silently drift from the .spec file over time.
python -m PyInstaller --noconfirm --clean MC_Translator.spec
if errorlevel 1 (
    echo Ошибка сборки.
    pause
    exit /b 1
)

echo.
echo [3/3] Готово!
echo    EXE-файл успешно создан: dist\MC_Translator.exe
echo.
echo Рядом с EXE положите при необходимости:
echo    settings.ini, dictionary.json, cache.json
echo.
pause