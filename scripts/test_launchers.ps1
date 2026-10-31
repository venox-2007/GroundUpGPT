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

$output = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
$isStillAlive = Get-Process -Id $PID -ErrorAction SilentlyContinue

Assert-True ($isStillAlive -ne $null) "Current process was NOT terminated despite being in PID file"
Assert-True ($output -match "Stale PID detected") "stop_app.ps1 detected stale PID and warned"
Assert-True (-not (Test-Path $pidFile)) "PID file was cleaned up"

# -------------------------------------------------------------------
# Test 2: Missing Tracking File Behavior
# -------------------------------------------------------------------
Write-Host "`nTest 2: Missing Tracking File Behavior..." -ForegroundColor Yellow
if (Test-Path $pidFile) { Remove-Item $pidFile -Force }

$outputMissing = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
Assert-True ($outputMissing -match "No launcher session tracking file found") "stop_app.ps1 safely handled missing tracking file"
Assert-True ($outputMissing -match "Preserving untracked processes") "Warned and preserved untracked processes"

# -------------------------------------------------------------------
# Test 3: Untracked Listener Protected Even With Matching Strings
# -------------------------------------------------------------------
Write-Host "`nTest 3: Untracked Listener Protected..." -ForegroundColor Yellow

# Dummy listener script that intentionally contains GroundUpGPT keywords in its text/commandline
$listenerScript = @"
# groundupgpt backend.main mock test listener
`$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 8000)
`$listener.Start()
Start-Sleep -Seconds 10
`$listener.Stop()
"@
$tempFile = [System.IO.Path]::GetTempFileName() + ".ps1"
Set-Content -Path $tempFile -Value $listenerScript

$dummyJob = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempFile`"" -PassThru
Start-Sleep -Milliseconds 900

try {
    $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    Assert-True ($conn -ne $null) "Dummy listener is active on port 8000"

    # Run stop_app.ps1 without tracking file - it MUST NOT terminate this untracked process!
    $stopOut = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
    
    $dummyStillAlive = Get-Process -Id $dummyJob.Id -ErrorAction SilentlyContinue
    Assert-True ($dummyStillAlive -ne $null) "Untracked process on port 8000 was NOT terminated by stop_app.ps1"

    # Even if an unrelated tracking file exists without this PID
    $mockEntry2 = @(
        @{
            pid = 999999
            startTime = "2026-01-01T00:00:00.0000000+00:00"
            role = "other"
            name = "dummy"
        }
    )
    $mockEntry2 | ConvertTo-Json | Set-Content $pidFile -Force
    $stopOut2 = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
    $dummyStillAlive2 = Get-Process -Id $dummyJob.Id -ErrorAction SilentlyContinue
    Assert-True ($dummyStillAlive2 -ne $null) "Untracked process was preserved even when stop_app processed another session"
    Assert-True ($stopOut2 -match "Preserving untracked process on port 8000") "Explicitly reported preserving untracked process on port 8000"
} finally {
    if ($dummyJob -and -not $dummyJob.HasExited) {
        Stop-Process -Id $dummyJob.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $tempFile) {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $pidFile) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# -------------------------------------------------------------------
# Test 4: Owned-Process Verified Termination
# -------------------------------------------------------------------
Write-Host "`nTest 4: Owned-Process Verified Termination..." -ForegroundColor Yellow

# Start a dummy process to simulate a launcher-owned background worker
$ownedScript = @"
Start-Sleep -Seconds 60
"@
$tempFileOwned = [System.IO.Path]::GetTempFileName() + ".ps1"
Set-Content -Path $tempFileOwned -Value $ownedScript
$ownedJob = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempFileOwned`"" -PassThru
Start-Sleep -Milliseconds 500

try {
    $procObj = Get-Process -Id $ownedJob.Id
    $ownedEntry = @(
        @{
            pid = $ownedJob.Id
            startTime = $procObj.StartTime.ToString("o")
            role = "test-backend"
            name = $procObj.ProcessName
        }
    )
    $ownedEntry | ConvertTo-Json | Set-Content $pidFile -Force

    # Run stop_app.ps1 - it MUST stop this verified owned process
    $stopOutOwned = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_app.ps1") | Out-String
    $ownedStillAlive = Get-Process -Id $ownedJob.Id -ErrorAction SilentlyContinue
    Assert-True ($ownedStillAlive -eq $null) "Verified owned process was stopped"
    Assert-True ($stopOutOwned -match "Stopping verified GroundUpGPT process") "Reported stopping verified process"
} finally {
    if ($ownedJob -and -not $ownedJob.HasExited) {
        Stop-Process -Id $ownedJob.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $tempFileOwned) {
        Remove-Item $tempFileOwned -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $pidFile) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# -------------------------------------------------------------------
# Test 5: Unrelated Port Conflict Detection in start_app.ps1
# -------------------------------------------------------------------
Write-Host "`nTest 5: Port Conflict Handling in start_app.ps1..." -ForegroundColor Yellow

$tempFileConflict = [System.IO.Path]::GetTempFileName() + ".ps1"
Set-Content -Path $tempFileConflict -Value $listenerScript
$dummyConflict = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempFileConflict`"" -PassThru
Start-Sleep -Milliseconds 900

try {
    $outLog1 = [System.IO.Path]::GetTempFileName()
    $startProc = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -CheckOnly -NonInteractive" -PassThru -Wait -RedirectStandardOutput $outLog1
    
    Assert-True ($startProc.ExitCode -eq 1) "start_app.ps1 exited with code 1 upon port 8000 conflict"
    $log1 = Get-Content $outLog1 -Raw
    Assert-True ($log1 -match "Port 8000 is occupied by an unrelated application") "start_app.ps1 logged clear port conflict message"
    $dummyStillAliveConflict = Get-Process -Id $dummyConflict.Id -ErrorAction SilentlyContinue
    Assert-True ($dummyStillAliveConflict -ne $null) "Conflicting application on port 8000 was untouched"
    if (Test-Path $outLog1) { Remove-Item $outLog1 -Force }
} finally {
    if ($dummyConflict -and -not $dummyConflict.HasExited) {
        Stop-Process -Id $dummyConflict.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $tempFileConflict) {
        Remove-Item $tempFileConflict -Force -ErrorAction SilentlyContinue
    }
}

# -------------------------------------------------------------------
# Test 6: Complete Model Artifact Verification (Checkpoint + Tokenizer)
# -------------------------------------------------------------------
Write-Host "`nTest 6: Artifact Verification in start_app.ps1..." -ForegroundColor Yellow

$tokPath = Join-Path $root "checkpoints\demo_tokenizer.json"
$tokBackup = Join-Path $root "checkpoints\demo_tokenizer.json.bak"
$ckptPath = Join-Path $root "checkpoints\demo_checkpoint.pt"
$ckptBackup = Join-Path $root "checkpoints\demo_checkpoint.pt.bak"

try {
    # 6a: Missing demo_tokenizer.json
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
    # 6b: Missing demo_checkpoint.pt
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

# 6c: Both artifacts verified
$outLogC = [System.IO.Path]::GetTempFileName()
$pC = Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\start_app.ps1`" -CheckOnly -NonInteractive" -PassThru -Wait -RedirectStandardOutput $outLogC
$logC = Get-Content $outLogC -Raw
Assert-True ($logC -match "Verified demo model checkpoint and tokenizer artifacts") "Declared model ready when both checkpoint and tokenizer exist"
if (Test-Path $outLogC) { Remove-Item $outLogC -Force }

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host " Launcher Tests Complete: $testsPassed Passed, $testsFailed Failed" -ForegroundColor $(if ($testsFailed -eq 0) { "Green" } else { "Red" })
Write-Host "========================================================" -ForegroundColor Cyan

if ($testsFailed -gt 0) {
    exit 1
}
