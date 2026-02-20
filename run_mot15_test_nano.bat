@echo off
REM Quick start batch script for MOT15 testing with nano model
REM Usage: run_mot15_test_nano.bat

echo.
echo =====================================
echo ByteTrack MOT15 Test - Nano Model
echo =====================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python not found in PATH
    echo Please install Python or add it to PATH
    pause
    exit /b 1
)

REM Display configuration
echo Configuration:
echo   Model: Nano (bytetrack_nano_mot17.pth.tar^)
echo   Dataset: MOT15 Test Set (11 sequences^)
echo   Device: CPU (add --device gpu for GPU^)
echo.

REM Ask for confirmation
set /p response="Start testing? (Y/N): "
if /i not "%response%"=="Y" (
    echo Testing cancelled.
    pause
    exit /b 0
)

echo.
echo Starting test...
echo.

REM Run the test script
python test_mot15_nano.py

REM Check exit code
if %errorlevel% equ 0 (
    echo.
    echo Testing completed successfully!
) else (
    echo.
    echo Testing failed with errors.
)

echo.
pause
