# Start the QA API on 0.0.0.0:8000 (CPU only) from a clean checkout (Windows PowerShell).
#
#   $env:QA_API_TOKEN = "<token>"          # required, never committed
#   $env:TEAM_NAME = "<registered team name>"
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
#
# Optional: $env:QA_MODEL_NAME = "<folder under weights\>" (default: modelo-final-v3),
#           $env:SKIP_INSTALL = "1" to skip the dependency install.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$VenvDir = ".venv"
$ModelName = if ($env:QA_MODEL_NAME) { $env:QA_MODEL_NAME } else { "modelo-final-v3" }
$ModelDir = Join-Path "weights" $ModelName
$TorchCpuIndex = "https://download.pytorch.org/whl/cpu"

if (-not $env:QA_API_TOKEN) {
    Write-Host "ERROR: QA_API_TOKEN is not set. Refusing to start." -ForegroundColor Red
    Write-Host '       Set it in this terminal first: $env:QA_API_TOKEN = "<token>"'
    exit 1
}

if (-not $env:TEAM_NAME) {
    Write-Host "WARNING: TEAM_NAME is not set; /info will report the placeholder team name." -ForegroundColor Yellow
    Write-Host '         Set it with: $env:TEAM_NAME = "<registered team name>"'
}

if (-not (Test-Path (Join-Path $ModelDir "model.safetensors"))) {
    Write-Host "ERROR: $ModelDir\model.safetensors not found." -ForegroundColor Red
    Write-Host "       The weights are not in git. Download $ModelName.zip from the team's GitHub Release"
    Write-Host "       or shared folder and extract it so the files sit directly in $ModelDir\"
    Write-Host "       (model.safetensors, config.json, tokenizer files), not in $ModelDir\$ModelName\."
    exit 1
}
if (-not (Test-Path (Join-Path $ModelDir "metrics.json")) -and -not (Test-Path (Join-Path "logs\$ModelName" "metrics.json"))) {
    Write-Host "ERROR: no metrics.json (no_answer_threshold) in $ModelDir\ or logs\$ModelName\." -ForegroundColor Red
    exit 1
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in $VenvDir ..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $VenvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvDir
    } else {
        Write-Host "ERROR: Python 3.10+ not found (neither 'py' nor 'python'). Install it from python.org." -ForegroundColor Red
        exit 1
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: venv creation failed." -ForegroundColor Red; exit 1 }
}

if ($env:SKIP_INSTALL -ne "1") {
    Write-Host "Installing serving dependencies (CPU torch) ..."
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pip upgrade failed." -ForegroundColor Red; exit 1 }
    & $VenvPython -m pip install -r requirements-serve.txt --extra-index-url $TorchCpuIndex
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: dependency install failed." -ForegroundColor Red; exit 1 }
}

Write-Host ""
Write-Host "Reminder for the dry run and the presentation (not executed automatically):" -ForegroundColor Cyan
Write-Host "  - Keep the laptop plugged in."
Write-Host "  - Disable sleep on AC power (admin PowerShell): powercfg /change standby-timeout-ac 0"
Write-Host "  - In another terminal start the tunnel: ngrok http 8000   (backup: cloudflared tunnel --url http://localhost:8000)"
Write-Host ""

$env:QA_MODEL_NAME = $ModelName
Write-Host "Starting API on http://0.0.0.0:8000 with model $ModelName (Ctrl+C to stop) ..."
& $VenvPython -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
exit $LASTEXITCODE
