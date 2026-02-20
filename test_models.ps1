#!/usr/bin/env pwsh
# Test Multiple ByteTrack Models Script
# This script helps you test different YOLOX model sizes

param(
    [string]$VideoPath = "videos/palace.mp4",
    [string]$Device = "cpu",
    [switch]$Fp16,
    [switch]$Fuse,
    [switch]$AllModels,
    [string[]]$Models = @("nano", "x")
)

# Color output functions
function Write-Header {
    param([string]$Text)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Green
}

function Write-Error {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Red
}

function Write-Info {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Blue
}

# Model definitions
$modelConfigs = @{
    "nano" = @{
        config = "exps/example/mot/yolox_nano_mix_det.py"
        checkpoint = "pretrained/bytetrack_nano_mot17.pth.tar"
        description = "Nano - Fastest, smallest model (0.90M params)"
    }
    "tiny" = @{
        config = "exps/example/mot/yolox_tiny_mix_det.py"
        checkpoint = "pretrained/bytetrack_tiny_mot17.pth.tar"
        description = "Tiny - Small model (5.03M params)"
    }
    "s" = @{
        config = "exps/example/mot/yolox_s_mix_det.py"
        checkpoint = "pretrained/bytetrack_s_mot17.pth.tar"
        description = "S - Small model, balanced (9M params)"
    }
    "m" = @{
        config = "exps/example/mot/yolox_m_mix_det.py"
        checkpoint = "pretrained/bytetrack_m_mot17.pth.tar"
        description = "M - Medium model (25M params)"
    }
    "l" = @{
        config = "exps/example/mot/yolox_l_mix_det.py"
        checkpoint = "pretrained/bytetrack_l_mot17.pth.tar"
        description = "L - Large model, high accuracy (54M params)"
    }
    "x" = @{
        config = "exps/example/mot/yolox_x_mix_det.py"
        checkpoint = "pretrained/bytetrack_x_mot17.pth.tar"
        description = "X - Extra large, best accuracy (99M params)"
    }
}

# Verify video exists
if (-not (Test-Path $VideoPath)) {
    Write-Error "Error: Video file not found: $VideoPath"
    Write-Info "Available videos:"
    Get-ChildItem -Path "videos" -Filter "*.mp4" | ForEach-Object { Write-Host "  - videos/$($_.Name)" }
    exit 1
}

# Determine which models to test
if ($AllModels) {
    $Models = @("nano", "tiny", "s", "m", "l", "x")
}

# Results tracking
$results = @()

Write-Header "ByteTrack Multi-Model Testing"
Write-Info "Video: $VideoPath"
Write-Info "Device: $Device"
Write-Info "Models to test: $($Models -join ', ')"
Write-Info "FP16: $Fp16"
Write-Info "Fuse: $Fuse"
Write-Host ""

# Test each model
foreach ($modelName in $Models) {
    if (-not $modelConfigs.ContainsKey($modelName)) {
        Write-Error "Unknown model: $modelName"
        Write-Info "Available models: $($modelConfigs.Keys -join ', ')"
        continue
    }
    
    $model = $modelConfigs[$modelName]
    
    Write-Header "Testing: $($model.description)"
    
    # Check if checkpoint exists
    if (-not (Test-Path $model.checkpoint)) {
        Write-Error "Model checkpoint not found: $($model.checkpoint)"
        Write-Info "Please download it from Google Drive (see TESTING_VARIOUS_MODELS.md)"
        
        $results += [PSCustomObject]@{
            Model = $modelName
            Status = "MISSING"
            Time = "N/A"
            FPS = "N/A"
        }
        continue
    }
    
    # Check if config exists
    if (-not (Test-Path $model.config)) {
        Write-Error "Model config not found: $($model.config)"
        
        $results += [PSCustomObject]@{
            Model = $modelName
            Status = "ERROR"
            Time = "N/A"
            FPS = "N/A"
        }
        continue
    }
    
    # Build command
    $cmdArgs = @(
        "tools/demo_track.py",
        "video",
        "-f", $model.config,
        "-c", $model.checkpoint,
        "--path", $VideoPath,
        "--device", $Device,
        "--save_result"
    )
    
    if ($Fp16) { $cmdArgs += "--fp16" }
    if ($Fuse) { $cmdArgs += "--fuse" }
    
    Write-Info "Running: python $($cmdArgs -join ' ')"
    Write-Host ""
    
    # Time the execution
    $startTime = Get-Date
    
    try {
        & python @cmdArgs
        $exitCode = $LASTEXITCODE
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($exitCode -eq 0) {
            Write-Success "`n✓ Model '$modelName' completed successfully!"
            Write-Success "  Time: $($duration.TotalSeconds.ToString('F2')) seconds"
            
            # Try to extract FPS from output (if available)
            $fps = "N/A"
            
            $results += [PSCustomObject]@{
                Model = $modelName
                Status = "SUCCESS"
                Time = "$($duration.TotalSeconds.ToString('F2'))s"
                FPS = $fps
            }
        }
        else {
            Write-Error "`n✗ Model '$modelName' failed with exit code: $exitCode"
            
            $results += [PSCustomObject]@{
                Model = $modelName
                Status = "FAILED"
                Time = "$($duration.TotalSeconds.ToString('F2'))s"
                FPS = "N/A"
            }
        }
    }
    catch {
        Write-Error "`n✗ Model '$modelName' encountered an error: $_"
        
        $results += [PSCustomObject]@{
            Model = $modelName
            Status = "ERROR"
            Time = "N/A"
            FPS = "N/A"
        }
    }
    
    Write-Host "`n"
}

# Summary
Write-Header "Test Results Summary"
$results | Format-Table -AutoSize

Write-Host "`n"
Write-Info "Output videos saved to: YOLOX_outputs/"
Write-Info "To view results:"
Write-Host "  ls YOLOX_outputs -Recurse -Filter *.mp4" -ForegroundColor White

# Count successes
$successCount = ($results | Where-Object { $_.Status -eq "SUCCESS" }).Count
$totalCount = $results.Count

Write-Host "`n"
Write-Success "Completed: $successCount/$totalCount models tested successfully"

# Find outputs
Write-Host "`n"
Write-Info "Generated output files:"
Get-ChildItem -Path "YOLOX_outputs" -Recurse -Filter "*.mp4" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 10 |
    ForEach-Object {
        $relPath = $_.FullName.Replace((Get-Location).Path + "\", "")
        Write-Host "  - $relPath" -ForegroundColor White
    }

Write-Host "`n"
