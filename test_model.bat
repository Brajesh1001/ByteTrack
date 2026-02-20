@echo off
REM Simple Model Testing Batch File for ByteTrack
REM Usage: test_model.bat [nano|tiny|s|m|l|x]

setlocal enabledelayedexpansion

REM Default to nano if no argument provided
set MODEL=%1
if "%MODEL%"=="" set MODEL=nano

REM Define model configurations
if "%MODEL%"=="nano" (
    set CONFIG=exps/example/mot/yolox_nano_mix_det.py
    set CHECKPOINT=pretrained/bytetrack_nano_mot17.pth.tar
    set DESC=NANO - Fastest, smallest model
)
if "%MODEL%"=="tiny" (
    set CONFIG=exps/example/mot/yolox_tiny_mix_det.py
    set CHECKPOINT=pretrained/bytetrack_tiny_mot17.pth.tar
    set DESC=TINY - Small model
)
if "%MODEL%"=="s" (
    set CONFIG=exps/example/mot/yolox_s_mix_det.py
    set CHECKPOINT=pretrained/bytetrack_s_mot17.pth.tar
    set DESC=S - Small, balanced model
)
if "%MODEL%"=="m" (
    set CONFIG=exps/example/mot/yolox_m_mix_det.py
    set CHECKPOINT=pretrained/bytetrack_m_mot17.pth.tar
    set DESC=M - Medium model
)
if "%MODEL%"=="l" (
    set CONFIG=exps/example/mot/yolox_l_mix_det.py
    set CHECKPOINT=pretrained/bytetrack_l_mot17.pth.tar
    set DESC=L - Large, high accuracy model
)
if "%MODEL%"=="x" (
    set CONFIG=exps/example/mot/yolox_x_mix_det.py
    set CHECKPOINT=pretrained/bytetrack_x_mot17.pth.tar
    set DESC=X - Extra large, best accuracy
)

REM Check if config was set (valid model name)
if not defined CONFIG (
    echo Error: Invalid model name: %MODEL%
    echo.
    echo Available models: nano, tiny, s, m, l, x
    echo.
    echo Usage: test_model.bat [model_name]
    echo Example: test_model.bat nano
    exit /b 1
)

REM Display info
echo ========================================
echo Testing ByteTrack Model: %DESC%
echo ========================================
echo Config: %CONFIG%
echo Checkpoint: %CHECKPOINT%
echo.

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo Error: Model checkpoint not found: %CHECKPOINT%
    echo Please download it from Google Drive
    echo See TESTING_VARIOUS_MODELS.md for download links
    exit /b 1
)

REM Check if config exists
if not exist "%CONFIG%" (
    echo Error: Model config not found: %CONFIG%
    exit /b 1
)

REM Set video path (can be modified)
set VIDEO_PATH=videos/palace.mp4

if not exist "%VIDEO_PATH%" (
    echo Error: Video not found: %VIDEO_PATH%
    echo Available videos:
    dir /b videos\*.mp4
    exit /b 1
)

echo Video: %VIDEO_PATH%
echo Device: cpu
echo.
echo Starting test...
echo.

REM Run the test
python tools/demo_track.py video -f %CONFIG% -c %CHECKPOINT% --path %VIDEO_PATH% --device cpu --save_result

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Test completed successfully!
    echo ========================================
    echo.
    echo Output saved to: YOLOX_outputs\
    echo.
    echo To view output:
    echo   explorer YOLOX_outputs
    echo.
) else (
    echo.
    echo ========================================
    echo Test failed with error code: %ERRORLEVEL%
    echo ========================================
    echo.
)

endlocal
