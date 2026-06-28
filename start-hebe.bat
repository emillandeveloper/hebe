@echo off
setlocal

echo.
echo Starting Hebe...
echo.

set "PROJECT_ROOT=%~dp0"
set "FRONTEND_DIR=%PROJECT_ROOT%frontend"

echo Checking Node...
where node >nul 2>nul
if errorlevel 1 (
    echo ERROR: Node.js was not found on PATH.
    echo Install Node.js or add it to PATH, then try again.
    echo.
    echo Startup failed. Press any key to close.
    pause >nul
    exit /b 1
)

echo Checking npm...
where npm >nul 2>nul
if errorlevel 1 (
    echo ERROR: npm was not found on PATH.
    echo Install Node.js/npm or add it to PATH, then try again.
    echo.
    echo Startup failed. Press any key to close.
    pause >nul
    exit /b 1
)

if not exist "%FRONTEND_DIR%\package.json" (
    echo ERROR: Could not find frontend\package.json.
    echo Project root: "%PROJECT_ROOT%"
    echo Expected frontend folder: "%FRONTEND_DIR%"
    echo.
    echo Startup failed. Press any key to close.
    pause >nul
    exit /b 1
)

echo Changing directory to:
echo "%FRONTEND_DIR%"
cd /d "%FRONTEND_DIR%"
if errorlevel 1 (
    echo ERROR: Could not change directory to "%FRONTEND_DIR%".
    echo.
    echo Startup failed. Press any key to close.
    pause >nul
    exit /b 1
)

echo.
echo Running electron dev:
echo npm run electron:dev
echo.

call npm run electron:dev
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Startup failed with exit code %EXIT_CODE%.
    echo Press any key to close.
    pause >nul
)

exit /b %EXIT_CODE%
