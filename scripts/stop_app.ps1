$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot

Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ' GroundUpGPT — Service Shutdown' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''

$pidsToKill = @()
$pidFile = Join-Path $root '.groundupgpt_pids'

# 1. Read stored PIDs from previous startup
if (Test-Path $pidFile) {
    Get-Content $pidFile | ForEach-Object {
        if ($_ -match 'PID=(\d+)') {
            $p = [int]$matches[1]
            if ($p -gt 0) { $pidsToKill += $p }
        }
    }
}

# 2. Check active listeners on ports 8000 and 3000
try {
    $connections = Get-NetTCPConnection -LocalPort 8000, 3000 -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($conn in $connections) {
            if ($conn.OwningProcess -gt 0) {
                $pidsToKill += $conn.OwningProcess
            }
        }
    }
} catch {}

$uniquePids = $pidsToKill | Select-Object -Unique

if (-not $uniquePids -or $uniquePids.Count -eq 0) {
    Write-Host '[INFO] No running GroundUpGPT processes found on ports 8000 or 3000.' -ForegroundColor Yellow
} else {
    foreach ($pidNum in $uniquePids) {
        try {
            $proc = Get-Process -Id $pidNum -ErrorAction Stop
            $procName = $proc.ProcessName.ToLower()

            # Safety check: only stop if process is python, node, cmd, or uvicorn
            if ($procName -match 'python|node|cmd|uvicorn') {
                Write-Host ('Stopping GroundUpGPT ' + $proc.ProcessName + ' (PID ' + $pidNum + ')...') -ForegroundColor Cyan
                Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue
            } else {
                Write-Host ('Skipping unrelated process ' + $proc.ProcessName + ' (PID ' + $pidNum + ')') -ForegroundColor Yellow
            }
        } catch {
            # Process may have already exited
        }
    }
    Write-Host '[SUCCESS] GroundUpGPT services stopped successfully.' -ForegroundColor Green
}

# Clean up PID file
if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host 'Shutdown complete.' -ForegroundColor Gray
Start-Sleep -Seconds 1
