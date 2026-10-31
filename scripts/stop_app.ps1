# GroundUpGPT Shutdown Script
$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot

Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ' GroundUpGPT — Safe Service Shutdown' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''

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

$stoppedCount = 0
$pidFile = Join-Path $root '.groundupgpt_pids'

# 1. Stop tracked GroundUpGPT processes with StartTime and identity verification
if (Test-Path $pidFile) {
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
                        # Process has already terminated
                        continue
                    }

                    # STALE PID CHECK: Verify StartTime matches recorded StartTime
                    $currentStartTime = $proc.StartTime.ToString("o")
                    if ($entry.startTime -and ($currentStartTime -ne $entry.startTime)) {
                        Write-Host ('[WARN] Stale PID detected: PID ' + $pidNum + ' was reused by another process (started at ' + $currentStartTime + ' vs recorded ' + $entry.startTime + '). Skipping to protect unrelated application.') -ForegroundColor Yellow
                        continue
                    }

                    # Verify process identity
                    if (-not (Test-IsGroundUpGPTProcess $pidNum) -and ($proc.ProcessName -notmatch '^(cmd|python|node)$')) {
                        Write-Host ('[WARN] Process ' + $proc.ProcessName + ' (PID ' + $pidNum + ') does not match GroundUpGPT identity. Skipping.') -ForegroundColor Yellow
                        continue
                    }

                    $role = if ($entry.role) { $entry.role } else { $proc.ProcessName }
                    Write-Host ('[*] Stopping GroundUpGPT ' + $role + ' (PID ' + $pidNum + ')...') -ForegroundColor Cyan
                    Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue
                    $stoppedCount++
                }
            }
        }
    } catch {
        Write-Host ('[WARN] Error parsing tracked process file: ' + $_.Exception.Message) -ForegroundColor Yellow
    }
}

# 2. Check listeners on ports 8000 and 3000 to catch any remaining GroundUpGPT processes,
# while explicitly protecting and never terminating unrelated applications.
$ports = @(8000, 3000)
foreach ($port in $ports) {
    try {
        $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($conns) {
            foreach ($conn in $conns) {
                $pidNum = $conn.OwningProcess
                if ($pidNum -le 4) { continue }

                $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
                if (-not $proc) { continue }

                if (Test-IsGroundUpGPTProcess $pidNum) {
                    Write-Host ('[*] Stopping GroundUpGPT listener on port ' + $port + ' (PID ' + $pidNum + ', ' + $proc.ProcessName + ')...') -ForegroundColor Cyan
                    Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue
                    $stoppedCount++
                } else {
                    Write-Host ('[INFO] Preserving unrelated application on port ' + $port + ' (PID ' + $pidNum + ', ' + $proc.ProcessName + ').') -ForegroundColor Gray
                }
            }
        }
    } catch {}
}

# 3. Clean up PID file
if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host ''
if ($stoppedCount -gt 0) {
    Write-Host ('[SUCCESS] GroundUpGPT services stopped successfully (' + $stoppedCount + ' process(es) terminated).') -ForegroundColor Green
} else {
    Write-Host '[INFO] No running GroundUpGPT processes found.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Shutdown complete.' -ForegroundColor Gray
Start-Sleep -Seconds 1
