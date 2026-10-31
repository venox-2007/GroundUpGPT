# GroundUpGPT Launcher Script
param(
    [switch]$NonInteractive,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ' GroundUpGPT — One-Click Application Launcher' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''

$isInteractive = (-not $NonInteractive) -and ($env:GROUNDUPGPT_NONINTERACTIVE -ne '1') -and (-not [Console]::IsInputRedirected)

function Wait-Exit {
    param([int]$code = 1)
    if ($script:isInteractive) {
        try {
            Read-Host 'Press Enter to exit...'
        } catch {}
    }
    exit $code
}

function Test-IsGroundUpGPTProcess {
    param([int]$procId)
    if ($procId -le 4) { return $false }
    try {
        $p = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $procId) -ErrorAction SilentlyContinue
        if (-not $p) { return $false }
        $cmdLine = if ($p.CommandLine) { $p.CommandLine.ToLower() } else { "" }
        $exePath = if ($p.ExecutablePath) { $p.ExecutablePath.ToLower() } else { "" }
        $rootLower = $root.ToLower()

        if ($exePath.StartsWith($rootLower)) { return $true }
        if ($cmdLine.Contains($rootLower)) { return $true }
        if ($cmdLine.Contains("backend.main")) { return $true }
        if ($cmdLine.Contains("groundupgpt")) { return $true }
        if ($cmdLine.Contains("next") -and $cmdLine.Contains("frontend")) { return $true }
        return $false
    } catch {
        return $false
    }
}

# 1. Check Python Virtual Environment
$pythonExe = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
    Write-Host ('[ERROR] Virtual environment Python not found at: ' + $pythonExe) -ForegroundColor Red
    Write-Host 'Please create the virtual environment (.venv) first as documented in README.md.' -ForegroundColor Red
    Write-Host ''
    Wait-Exit 1
}

# 2. Check Node.js and npm
$nodePath = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodePath) {
    Write-Host '[ERROR] Node.js is not found in PATH.' -ForegroundColor Red
    Write-Host 'Please install Node.js 18+ from https://nodejs.org/ to run the web UI.' -ForegroundColor Red
    Write-Host ''
    Wait-Exit 1
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
        Wait-Exit 1
    }
}

# 3. Check for Model Artifacts
# Verify BOTH demo_checkpoint.pt and demo_tokenizer.json exist before declaring the model ready.
# Do not automatically retrain.
$ckptPath = Join-Path $root 'checkpoints\demo_checkpoint.pt'
$tokPath = Join-Path $root 'checkpoints\demo_tokenizer.json'

$hasCkpt = Test-Path $ckptPath
$hasTok = Test-Path $tokPath

if (-not ($hasCkpt -and $hasTok)) {
    Write-Host '[NOTICE] Required demo model artifacts are missing:' -ForegroundColor Yellow
    if (-not $hasCkpt) {
        Write-Host ('  - Missing: ' + $ckptPath) -ForegroundColor Yellow
    }
    if (-not $hasTok) {
        Write-Host ('  - Missing: ' + $tokPath) -ForegroundColor Yellow
    }
    Write-Host 'GroundUpGPT includes an automated lightweight generator that trains' -ForegroundColor Yellow
    Write-Host 'a tiny experimental demo model in ~10 seconds.' -ForegroundColor Yellow
    Write-Host ''
    $ans = 'N'
    if ($script:isInteractive) {
        $ans = Read-Host 'Would you like to generate demo artifacts now? [Y/N]'
    } else {
        Write-Host '[*] Non-interactive mode: Skipping artifact generation.' -ForegroundColor Yellow
    }
    if ($ans -match '^[Yy]') {
        Write-Host ''
        Write-Host '[*] Training demo model on offline fixture...' -ForegroundColor Cyan
        & $pythonExe (Join-Path $root 'scripts\generate_demo.py')
        if (-not ((Test-Path $ckptPath) -and (Test-Path $tokPath))) {
            Write-Host '[ERROR] Failed to generate both demo model artifacts.' -ForegroundColor Red
            Wait-Exit 1
        }
        Write-Host '[*] Demo model artifacts successfully generated.' -ForegroundColor Green
    } else {
        Write-Host '[*] Starting backend in uninitialized mode without model artifacts.' -ForegroundColor Yellow
    }
    Write-Host ''
} else {
    Write-Host '[*] Verified demo model checkpoint and tokenizer artifacts.' -ForegroundColor Green
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
        Wait-Exit 1
    }
    Write-Host '[*] Frontend dependencies installed successfully.' -ForegroundColor Green
    Write-Host ''
}

# 5. Check Port Status and Handle Conflicts
$backendRunning = $false
$frontendRunning = $false

$bConn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$fConn = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1

if ($bConn) {
    # Check if this listener is an active GroundUpGPT backend
    $isOurs = $false
    try {
        $res = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2 -ErrorAction Stop
        if ($res.status -eq 'ok') {
            $isOurs = $true
        }
    } catch {}

    if ($isOurs -or (Test-IsGroundUpGPTProcess $bConn.OwningProcess)) {
        $backendRunning = $true
        Write-Host '[*] GroundUpGPT backend is already active on port 8000.' -ForegroundColor Yellow
    } else {
        $proc = Get-Process -Id $bConn.OwningProcess -ErrorAction SilentlyContinue
        $pName = if ($proc) { $proc.ProcessName } else { "Unknown" }
        Write-Host ('[ERROR] Port 8000 is occupied by an unrelated application (PID: ' + $bConn.OwningProcess + ', Process: ' + $pName + ').') -ForegroundColor Red
        Write-Host 'GroundUpGPT backend requires port 8000. Please terminate that application or free the port before running GroundUpGPT.' -ForegroundColor Red
        Write-Host ''
        Wait-Exit 1
    }
}

if ($fConn) {
    # Check if this listener is an active GroundUpGPT frontend
    $isOurs = $false
    try {
        $res = Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($res.StatusCode -eq 200 -and $res.Content -match 'GroundUpGPT') {
            $isOurs = $true
        }
    } catch {}

    if ($isOurs -or (Test-IsGroundUpGPTProcess $fConn.OwningProcess)) {
        $frontendRunning = $true
        Write-Host '[*] GroundUpGPT frontend is already active on port 3000.' -ForegroundColor Yellow
    } else {
        $proc = Get-Process -Id $fConn.OwningProcess -ErrorAction SilentlyContinue
        $pName = if ($proc) { $proc.ProcessName } else { "Unknown" }
        Write-Host ('[ERROR] Port 3000 is occupied by an unrelated application (PID: ' + $fConn.OwningProcess + ', Process: ' + $pName + ').') -ForegroundColor Red
        Write-Host 'GroundUpGPT frontend requires port 3000. Please terminate that application or free the port before running GroundUpGPT.' -ForegroundColor Red
        Write-Host ''
        Wait-Exit 1
    }
}

if ($backendRunning -and $frontendRunning) {
    Write-Host ''
    Write-Host '[INFO] GroundUpGPT backend (port 8000) and frontend (port 3000) are already running and responsive!' -ForegroundColor Green
    Write-Host 'Opening browser: http://localhost:3000' -ForegroundColor Cyan
    Start-Process 'http://localhost:3000'
    exit 0
}

if ($CheckOnly) {
    Write-Host '[*] Preflight verification complete.' -ForegroundColor Green
    exit 0
}

# 6. Launch Services
$bCmdProc = $null
$fCmdProc = $null

if (-not $backendRunning) {
    Write-Host '[*] Launching FastAPI Backend on http://127.0.0.1:8000...' -ForegroundColor Cyan
    $bArgs = '/k title GroundUpGPT Backend (Port 8000) && cd /d "' + $root + '" && "' + $pythonExe + '" -m backend.main'
    $bCmdProc = Start-Process -FilePath 'cmd.exe' -ArgumentList $bArgs -PassThru
}

if (-not $frontendRunning) {
    Write-Host '[*] Launching Next.js Frontend on http://localhost:3000...' -ForegroundColor Cyan
    $fArgs = '/k title GroundUpGPT Frontend (Port 3000) && cd /d "' + $frontendDir + '" && ' + $npmCmd + ' run dev'
    $fCmdProc = Start-Process -FilePath 'cmd.exe' -ArgumentList $fArgs -PassThru
}

# 7. Wait for Both Services to Become Ready
Write-Host '[*] Waiting for services to initialize...' -ForegroundColor Gray

$backendReady = $false
$frontendReady = $false
$maxAttempts = 40
$attempt = 0

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Seconds 1
    $attempt++

    if (-not $backendReady) {
        try {
            $bRes = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 1 -ErrorAction Stop
            if ($bRes.status -eq 'ok') {
                $backendReady = $true
            }
        } catch {}
    }

    if (-not $frontendReady) {
        try {
            $fRes = Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
            if ($fRes.StatusCode -eq 200) {
                $frontendReady = $true
            }
        } catch {}
    }

    if ($backendReady -and $frontendReady) {
        break
    }
}

# If readiness failed or timed out: DO NOT print success and DO NOT open browser!
if (-not ($backendReady -and $frontendReady)) {
    Write-Host ''
    Write-Host '=====================================================================' -ForegroundColor Red
    Write-Host ' [ERROR] GroundUpGPT startup timed out or failed to become ready!' -ForegroundColor Red
    Write-Host '=====================================================================' -ForegroundColor Red
    $bStatus = if ($backendReady) { 'READY' } else { 'FAILED / NOT RESPONDING' }
    $fStatus = if ($frontendReady) { 'READY' } else { 'FAILED / NOT RESPONDING' }
    Write-Host ('  - Backend (http://127.0.0.1:8000/api/health): ' + $bStatus) -ForegroundColor $(if ($backendReady) { 'Green' } else { 'Red' })
    Write-Host ('  - Frontend (http://localhost:3000):             ' + $fStatus) -ForegroundColor $(if ($frontendReady) { 'Green' } else { 'Red' })
    Write-Host ''
    Write-Host 'Please inspect the Backend and Frontend terminal windows for crash or error logs.' -ForegroundColor Yellow
    Write-Host ''
    Wait-Exit 1
}

# 8. Record Verified Process Identities for Stop GroundUpGPT.bat
$trackedProcesses = @()

if ($bCmdProc -and (Get-Process -Id $bCmdProc.Id -ErrorAction SilentlyContinue)) {
    $trackedProcesses += @{
        role = 'backend-console'
        pid = $bCmdProc.Id
        startTime = $bCmdProc.StartTime.ToString("o")
        name = $bCmdProc.ProcessName
    }
}

$bConnAfter = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($bConnAfter) {
    $p = Get-Process -Id $bConnAfter.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and (Test-IsGroundUpGPTProcess $p.Id)) {
        $trackedProcesses += @{
            role = 'backend-server'
            pid = $p.Id
            startTime = $p.StartTime.ToString("o")
            name = $p.ProcessName
        }
    }
}

if ($fCmdProc -and (Get-Process -Id $fCmdProc.Id -ErrorAction SilentlyContinue)) {
    $trackedProcesses += @{
        role = 'frontend-console'
        pid = $fCmdProc.Id
        startTime = $fCmdProc.StartTime.ToString("o")
        name = $fCmdProc.ProcessName
    }
}

$fConnAfter = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($fConnAfter) {
    $p = Get-Process -Id $fConnAfter.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and (Test-IsGroundUpGPTProcess $p.Id)) {
        $trackedProcesses += @{
            role = 'frontend-server'
            pid = $p.Id
            startTime = $p.StartTime.ToString("o")
            name = $p.ProcessName
        }
    }
}

if ($trackedProcesses.Count -gt 0) {
    $json = $trackedProcesses | ConvertTo-Json
    Set-Content -Path (Join-Path $root '.groundupgpt_pids') -Value $json -Force
}

# 9. Successful Startup Notification
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
