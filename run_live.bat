@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv venv || goto :error
    ) else (
        python -m venv venv || goto :error
    )

    echo Upgrading pip...
    call "venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
)

echo Syncing requirements...
call "venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

echo Running live engine...
call "venv\Scripts\python.exe" main_live.py
if errorlevel 1 goto :error

echo.
echo Live engine finished.
pause
exit /b 0

:error
echo.
echo Launcher failed.
pause
exit /b 1
