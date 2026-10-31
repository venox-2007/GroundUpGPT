# Test Suite for GroundUpGPT Windows Launchers
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Running GroundUpGPT Launcher Verification Tests" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$testsPassed = 0
$testsFailed = 0

function Assert-True {
    param([bool]$condition, [string]$message)
    if ($condition) {
        Write-Host ("[PASS] " + $message) -ForegroundColor Green
        $script:testsPassed++
    } else {
        Write-Host ("[FAIL] " + $message) -ForegroundColor Red
        $script:testsFailed++
    }
}

# -------------------------------------------------------------------
# Test 1: Stale / Reused PID Protection in stop_app.ps1
# -------------------------------------------------------------------
Write-Host "Test 1: Stale PID Protection..." -ForegroundColor Yellow
$currentProc = Get-Process -Id $PID
$fakeStartTime = "2020-01-01T00:00:00.0000000+00:00"
$pidFile = Join-Path $root ".groundupgpt_pids"

$mockEntry = @(
    @{
        pid = $PID
        startTime = $fakeStartTime
        role = "stale-test"
        name = $currentProc.ProcessName
    }
)
$mockEntry | ConvertTo-Json | Set-Content $pidFile -Force

# Run stop_app.ps1
$output = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
$isStillAlive = Get-Process -Id $PID -ErrorAction SilentlyContinue

Assert-True ($isStillAlive -ne $null) "Current process was NOT terminated despite being in PID file"
Assert-True ($output -match "Stale PID detected") "stop_app.ps1 detected stale PID and warned"
Assert-True (-not (Test-Path $pidFile)) "PID file was cleaned up"

# -------------------------------------------------------------------
# Test 2: Unrelated Application on Target Port 8000 Protected in stop_app.ps1
# -------------------------------------------------------------------
Write-Host "`nTest 2: Unrelated Port 8000 Protection in stop_app.ps1..." -ForegroundColor Yellow

$listenerScript = @"
`$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 8000)
`$listener.Start()
Start-Sleep -Seconds 15
`$listener.Stop()
"@
$tempFile = [System.IO.Path]::GetTempFileName() + ".ps1"
Set-Content -Path $tempFile -Value $listenerScript

$dummyJob = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempFile`"" -PassThru
Start-Sleep -Milliseconds 900

try {
    $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    Assert-True ($conn -ne $null) "Dummy listener is active on port 8000"

    # Run stop_app.ps1 - it MUST NOT terminate this unrelated dummy process!
    $stopOut = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
    
    $dummyStillAlive = Get-Process -Id $dummyJob.Id -ErrorAction SilentlyContinue
    Assert-True ($dummyStillAlive -ne $null) "Unrelated process on port 8000 was NOT terminated by stop_app.ps1"
    Assert-True ($stopOut -match "Preserving unrelated application on port 8000") "stop_app.ps1 explicitly noted preservation of unrelated application"
} finally {
    if ($dummyJob -and -not $dummyJob.HasExited) {
        Stop-Process -Id $dummyJob.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $tempFile) {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }
}

# -------------------------------------------------------------------
# Test 3: Unrelated Port Conflict Detection in start_app.ps1 (Port 8000 & 3000)
# -------------------------------------------------------------------
Write-Host "`nTest 3: Port Conflict Handling in start_app.ps1..." -ForegroundColor Yellow

$tempFile2 = [System.IO.Path]::GetTempFileName() + ".ps1"
Set-Content -Path $tempFile2 -Value $listenerScript
$dummyJob2 = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempFile2`"" -PassThru
Start-Sleep -Milliseconds 900

try {
    $outLog1 = [System.IO.Path]::GetTempFileName()
    $startProc = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -CheckOnly -NonInteractive" -PassThru -Wait -RedirectStandardOutput $outLog1
    
    Assert-True ($startProc.ExitCode -eq 1) "start_app.ps1 exited with code 1 upon port 8000 conflict"
    $log1 = Get-Content $outLog1 -Raw
    Assert-True ($log1 -match "Port 8000 is occupied by an unrelated application") "start_app.ps1 logged clear port conflict message"
    $dummyStillAlive2 = Get-Process -Id $dummyJob2.Id -ErrorAction SilentlyContinue
    Assert-True ($dummyStillAlive2 -ne $null) "Conflicting application on port 8000 was untouched"
    if (Test-Path $outLog1) { Remove-Item $outLog1 -Force }
} finally {
    if ($dummyJob2 -and -not $dummyJob2.HasExited) {
        Stop-Process -Id $dummyJob2.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $tempFile2) {
        Remove-Item $tempFile2 -Force -ErrorAction SilentlyContinue
    }
}

# Port 3000 conflict check
$listenerScript3000 = @"
`$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 3000)
`$listener.Start()
Start-Sleep -Seconds 15
`$listener.Stop()
"@
$tempFile3 = [System.IO.Path]::GetTempFileName() + ".ps1"
Set-Content -Path $tempFile3 -Value $listenerScript3000
$dummyJob3 = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempFile3`"" -PassThru
Start-Sleep -Milliseconds 900

try {
    $outLog3 = [System.IO.Path]::GetTempFileName()
    $startProc3 = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -CheckOnly -NonInteractive" -PassThru -Wait -RedirectStandardOutput $outLog3
    
    Assert-True ($startProc3.ExitCode -eq 1) "start_app.ps1 exited with code 1 upon port 3000 conflict"
    $log3 = Get-Content $outLog3 -Raw
    Assert-True ($log3 -match "Port 3000 is occupied by an unrelated application") "start_app.ps1 logged clear port 3000 conflict message"
    $dummyStillAlive3 = Get-Process -Id $dummyJob3.Id -ErrorAction SilentlyContinue
    Assert-True ($dummyStillAlive3 -ne $null) "Conflicting application on port 3000 was untouched"
    if (Test-Path $outLog3) { Remove-Item $outLog3 -Force }
} finally {
    if ($dummyJob3 -and -not $dummyJob3.HasExited) {
        Stop-Process -Id $dummyJob3.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $tempFile3) {
        Remove-Item $tempFile3 -Force -ErrorAction SilentlyContinue
    }
}

# -------------------------------------------------------------------
# Test 4: Complete Model Artifact Verification (Checkpoint + Tokenizer)
# -------------------------------------------------------------------
Write-Host "`nTest 4: Artifact Verification in start_app.ps1..." -ForegroundColor Yellow

$tokPath = Join-Path $root "checkpoints\demo_tokenizer.json"
$tokBackup = Join-Path $root "checkpoints\demo_tokenizer.json.bak"
$ckptPath = Join-Path $root "checkpoints\demo_checkpoint.pt"
$ckptBackup = Join-Path $root "checkpoints\demo_checkpoint.pt.bak"

try {
    # 4a: Missing demo_tokenizer.json
    if (Test-Path $tokPath) {
        Move-Item $tokPath $tokBackup -Force
    }
    $outLogA = [System.IO.Path]::GetTempFileName()
    $pA = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -CheckOnly -NonInteractive" -PassThru -Wait -RedirectStandardOutput $outLogA
    $logA = Get-Content $outLogA -Raw

    Assert-True ($logA -match "Missing: .*demo_tokenizer\.json") "Reported missing demo_tokenizer.json"
    Assert-True (-not (Test-Path $tokPath)) "Did NOT automatically retrain demo_tokenizer.json"
    if (Test-Path $outLogA) { Remove-Item $outLogA -Force }
} finally {
    if (Test-Path $tokBackup) {
        Move-Item $tokBackup $tokPath -Force
    }
}

try {
    # 4b: Missing demo_checkpoint.pt
    if (Test-Path $ckptPath) {
        Move-Item $ckptPath $ckptBackup -Force
    }
    $outLogB = [System.IO.Path]::GetTempFileName()
    $pB = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -CheckOnly -NonInteractive" -PassThru -Wait -RedirectStandardOutput $outLogB
    $logB = Get-Content $outLogB -Raw

    Assert-True ($logB -match "Missing: .*demo_checkpoint\.pt") "Reported missing demo_checkpoint.pt"
    Assert-True (-not (Test-Path $ckptPath)) "Did NOT automatically retrain demo_checkpoint.pt"
    if (Test-Path $outLogB) { Remove-Item $outLogB -Force }
} finally {
    if (Test-Path $ckptBackup) {
        Move-Item $ckptBackup $ckptPath -Force
    }
}

# 4c: Both artifacts verified
$outLogC = [System.IO.Path]::GetTempFileName()
$pC = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -CheckOnly -NonInteractive" -PassThru -Wait -RedirectStandardOutput $outLogC
$logC = Get-Content $outLogC -Raw
Assert-True ($logC -match "Verified demo model checkpoint and tokenizer artifacts") "Declared model ready when both checkpoint and tokenizer exist"
if (Test-Path $outLogC) { Remove-Item $outLogC -Force }

# -------------------------------------------------------------------
# Test 5: Safe Full Lifecycle (Start & Stop)
# -------------------------------------------------------------------
Write-Host "`nTest 5: Full GroundUpGPT Lifecycle Test..." -ForegroundColor Yellow

# Launch start_app.ps1
$startOutput = [System.IO.Path]::GetTempFileName()
$launchProc = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -NonInteractive" -PassThru -RedirectStandardOutput $startOutput

# Wait for both services to become ready and .groundupgpt_pids to be created (up to 30s)
$pidFile = Join-Path $root ".groundupgpt_pids"
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $pidFile) {
        $ready = $true
        break
    }
}

Assert-True $ready "GroundUpGPT services initialized and .groundupgpt_pids tracking file created"

# Verify health endpoint
$backendOk = $false
try {
    $b = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2 -ErrorAction Stop
    if ($b.status -eq "ok") { $backendOk = $true }
} catch {}
Assert-True $backendOk "GroundUpGPT backend is responding on port 8000"

# Now execute stop_app.ps1
$stopOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
Assert-True ($stopOutput -match "SUCCESS") "stop_app.ps1 reported successful termination"

# Verify ports are freed
Start-Sleep -Seconds 1
$remainingConn = Get-NetTCPConnection -LocalPort 8000, 3000 -State Listen -ErrorAction SilentlyContinue
Assert-True ($remainingConn -eq $null) "Ports 8000 and 3000 are freed"
Assert-True (-not (Test-Path $pidFile)) "Tracking file removed after shutdown"

if (Test-Path $startOutput) { Remove-Item $startOutput -Force }

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host " Launcher Tests Complete: $testsPassed Passed, $testsFailed Failed" -ForegroundColor $(if ($testsFailed -eq 0) { "Green" } else { "Red" })
Write-Host "========================================================" -ForegroundColor Cyan

if ($testsFailed -gt 0) {
    exit 1
}
