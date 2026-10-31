# GroundUpGPT Shutdown Script
$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot

Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ' GroundUpGPT — Safe Service Shutdown' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''

$pidFile = Join-Path $root '.groundupgpt_pids'
$stoppedCount = 0

if (-not (Test-Path $pidFile)) {
    Write-Host '[WARN] No launcher session tracking file found (.groundupgpt_pids).' -ForegroundColor Yellow
    Write-Host '[INFO] Preserving untracked processes. If services were started manually, please close their windows.' -ForegroundColor Gray
    Write-Host ''
    Write-Host 'Shutdown complete.' -ForegroundColor Gray
    exit 0
}

# 1. Read and verify launcher-owned processes
$verifiedOwnedPids = @()
try {
    $raw = Get-Content $pidFile -Raw -ErrorAction SilentlyContinue
    if ($raw) {
        $entries = $raw | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($entries) {
            $list = @($entries)
            foreach ($entry in $list) {
                $pidNum = [int]$entry.pid
                if ($pidNum -le 4) { continue }

                $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
                if (-not $proc) {
                    # Process already exited
                    continue
                }

                # STALE PID CHECK: Strictly verify StartTime matches recorded StartTime
                $currentStartTime = $proc.StartTime.ToString("o")
                if ($entry.startTime -and ($currentStartTime -ne $entry.startTime)) {
                    Write-Host ('[WARN] Stale PID detected: PID ' + $pidNum + ' was reused by another process (started at ' + $currentStartTime + ' vs recorded ' + $entry.startTime + '). Preserving.') -ForegroundColor Yellow
                    continue
                }

                # Add to verified owned PIDs
                if ($verifiedOwnedPids -notcontains $pidNum) {
                    $verifiedOwnedPids += $pidNum
                }

                # Safe child process tracking via verified process ancestry
                try {
                    $children = Get-CimInstance Win32_Process -Filter ("ParentProcessId = " + $pidNum) -ErrorAction SilentlyContinue
                    if ($children) {
                        foreach ($child in $children) {
                            $cId = [int]$child.ProcessId
                            if ($cId -gt 4 -and ($verifiedOwnedPids -notcontains $cId)) {
                                $cProc = Get-Process -Id $cId -ErrorAction SilentlyContinue
                                if ($cProc -and $cProc.StartTime -ge $proc.StartTime) {
                                    $verifiedOwnedPids += $cId
                                }
                            }
                        }
                    }
                } catch {}
            }
        }
    }
} catch {
    Write-Host ('[WARN] Error parsing tracked process file: ' + $_.Exception.Message) -ForegroundColor Yellow
}

# 2. Terminate ONLY verified launcher-owned processes
foreach ($pidNum in $verifiedOwnedPids) {
    try {
        $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host ('[*] Stopping verified GroundUpGPT process ' + $proc.ProcessName + ' (PID ' + $pidNum + ')...') -ForegroundColor Cyan
            Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue
            $stoppedCount++
        }
    } catch {}
}

# 3. Inspect active listeners on ports 8000 and 3000 to notify user about any preserved untracked applications
$ports = @(8000, 3000)
foreach ($port in $ports) {
    try {
        $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($conns) {
            foreach ($conn in $conns) {
                $pidNum = $conn.OwningProcess
                if ($pidNum -gt 4 -and ($verifiedOwnedPids -notcontains $pidNum)) {
                    $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
                    $pName = if ($proc) { $proc.ProcessName } else { 'Unknown' }
                    Write-Host ('[INFO] Preserving untracked process on port ' + $port + ' (PID ' + $pidNum + ', ' + $pName + ').') -ForegroundColor Gray
                }
            }
        }
    } catch {}
}

# 4. Clean up session PID file
if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host ''
if ($stoppedCount -gt 0) {
    Write-Host ('[SUCCESS] GroundUpGPT services stopped successfully (' + $stoppedCount + ' verified process(es) terminated).') -ForegroundColor Green
} else {
    Write-Host '[INFO] No running launcher-owned processes were active.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Shutdown complete.' -ForegroundColor Gray
Start-Sleep -Seconds 1
