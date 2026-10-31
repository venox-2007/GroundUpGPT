# GroundUpGPT Launcher Script
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ' GroundUpGPT — One-Click Application Launcher' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''

# 1. Check Python Virtual Environment
$pythonExe = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
    Write-Host ('[ERROR] Virtual environment Python not found at: ' + $pythonExe) -ForegroundColor Red
    Write-Host 'Please create the virtual environment (.venv) first as documented in README.md.' -ForegroundColor Red
    Write-Host ''
    Read-Host 'Press Enter to exit...'
    exit 1
}

# 2. Check Node.js and npm
$nodePath = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodePath) {
    Write-Host '[ERROR] Node.js is not found in PATH.' -ForegroundColor Red
    Write-Host 'Please install Node.js 18+ from https://nodejs.org/ to run the web UI.' -ForegroundColor Red
    Write-Host ''
    Read-Host 'Press Enter to exit...'
    exit 1
}

$npmCmd = 'npm.cmd'
$npmPath = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npmPath) {
    $npmPath = Get-Command npm -ErrorAction SilentlyContinue
    if ($npmPath) {
        $npmCmd = 'npm'
    } else {
        Write-Host '[ERROR] npm was not found. Please verify your Node.js installation.' -ForegroundColor Red
        Write-Host ''
        Read-Host 'Press Enter to exit...'
        exit 1
    }
}

# 3. Check for Model Artifacts
$ckptPath = Join-Path $root 'checkpoints\demo_checkpoint.pt'
if (-not (Test-Path $ckptPath)) {
    Write-Host '[NOTICE] Demo model checkpoint was not found at:' -ForegroundColor Yellow
    Write-Host ('  ' + $ckptPath) -ForegroundColor Yellow
    Write-Host 'GroundUpGPT includes an automated lightweight generator that trains' -ForegroundColor Yellow
    Write-Host 'a tiny experimental demo model in ~10 seconds.' -ForegroundColor Yellow
    Write-Host ''
    $ans = Read-Host 'Would you like to generate demo artifacts now? [Y/N]'
    if ($ans -match '^[Yy]') {
        Write-Host ''
        Write-Host '[*] Training demo model on offline fixture...' -ForegroundColor Cyan
        & $pythonExe (Join-Path $root 'scripts\generate_demo.py')
        if (-not (Test-Path $ckptPath)) {
            Write-Host '[ERROR] Failed to generate demo model artifacts.' -ForegroundColor Red
            Read-Host 'Press Enter to exit...'
            exit 1
        }
    } else {
        Write-Host '[*] Starting backend in uninitialized mode without model artifacts.' -ForegroundColor Yellow
    }
    Write-Host ''
}

# 4. Check Frontend Dependencies
$frontendDir = Join-Path $root 'frontend'
$nodeModulesDir = Join-Path $frontendDir 'node_modules'
if (-not (Test-Path $nodeModulesDir)) {
    Write-Host '[NOTICE] Frontend packages not found. Installing node dependencies (one-time setup)...' -ForegroundColor Cyan
    Push-Location $frontendDir
    try {
        & $npmCmd install
    } finally {
        Pop-Location
    }
    if (-not (Test-Path $nodeModulesDir)) {
        Write-Host '[ERROR] Failed to install frontend packages.' -ForegroundColor Red
        Read-Host 'Press Enter to exit...'
        exit 1
    }
    Write-Host '[*] Frontend dependencies installed successfully.' -ForegroundColor Green
    Write-Host ''
}

# 5. Check if services are already running
$backendConn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
$frontendConn = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue

if ($backendConn -and $frontendConn) {
    Write-Host '[INFO] GroundUpGPT backend (port 8000) and frontend (port 3000) are already running!' -ForegroundColor Green
    Write-Host 'Opening browser: http://localhost:3000' -ForegroundColor Cyan
    Start-Process 'http://localhost:3000'
    exit 0
}

# 6. Launch FastAPI Backend
if (-not $backendConn) {
    Write-Host '[*] Launching FastAPI Backend on http://127.0.0.1:8000...' -ForegroundColor Cyan
    $bArgs = '/k title GroundUpGPT Backend (Port 8000) && cd /d "' + $root + '" && "' + $pythonExe + '" -m backend.main'
    Start-Process -FilePath 'cmd.exe' -ArgumentList $bArgs
} else {
    Write-Host '[*] Backend is already running on port 8000.' -ForegroundColor Yellow
}

# 7. Launch Next.js Frontend
if (-not $frontendConn) {
    Write-Host '[*] Launching Next.js Frontend on http://localhost:3000...' -ForegroundColor Cyan
    $fArgs = '/k title GroundUpGPT Frontend (Port 3000) && cd /d "' + $frontendDir + '" && ' + $npmCmd + ' run dev'
    Start-Process -FilePath 'cmd.exe' -ArgumentList $fArgs
} else {
    Write-Host '[*] Frontend is already running on port 3000.' -ForegroundColor Yellow
}

# 8. Wait for Both Services to Become Ready
Write-Host '[*] Waiting for services to initialize...' -ForegroundColor Gray
$ready = $false

for ($i = 0; $i -lt 40; $i++) {
    try {
        $bRes = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        $fRes = Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($bRes.StatusCode -eq 200 -and $fRes.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {}
    Start-Sleep -Milliseconds 750
}

# 9. Record PIDs for Stop GroundUpGPT.bat
$bPid = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -First 1)
$fPid = (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -First 1)
$pidContent = @()
if ($bPid) { $pidContent += ('BACKEND_PID=' + $bPid) }
if ($fPid) { $pidContent += ('FRONTEND_PID=' + $fPid) }
if ($pidContent.Count -gt 0) {
    Set-Content -Path (Join-Path $root '.groundupgpt_pids') -Value ($pidContent -join "`n")
}

Write-Host ''
Write-Host '=====================================================================' -ForegroundColor Green
Write-Host ' GroundUpGPT is ready!' -ForegroundColor Green
Write-Host ' Opening http://localhost:3000 in your browser...' -ForegroundColor Green
Write-Host '=====================================================================' -ForegroundColor Green
Write-Host ''
Write-Host 'Keep the Backend and Frontend terminal windows open while using the app.' -ForegroundColor Gray
Write-Host 'To safely stop all GroundUpGPT services, double-click "Stop GroundUpGPT.bat".' -ForegroundColor Gray
Write-Host ''

Start-Process 'http://localhost:3000'
Start-Sleep -Seconds 3
