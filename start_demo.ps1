$root = $PSScriptRoot
$positionScript = Join-Path $root "magFieldOptimize_PosEst.py"
$script = Join-Path $root "capsule_orientation.py"
$motorScript = Join-Path $root "motorControl.py"
$telemetryRoot = "R:\Release_Internal_V1A"
$sourceTelemetryRoot = "C:\Users\johncklau\Desktop\Release_Internal_V1A"
$outputDirectory = Join-Path $telemetryRoot "output"
$exportDirectory = Join-Path $telemetryRoot "export"
$capsuleLog = Join-Path $root "capsule_orientation_startup.log"
$childProcessIds = [System.Collections.Generic.List[int]]::new()

function Show-StartupError {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host "Launcher error: $Message" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process `
        -FilePath "powershell.exe" `
        -Verb RunAs `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath)
    exit
}

if (-not (Test-Path -LiteralPath "R:\" -PathType Container)) {
    Show-StartupError "R: drive is not available. Need to set up R disk first."
}
if (-not (Test-Path -LiteralPath $telemetryRoot -PathType Container)) {
    if (-not (Test-Path -LiteralPath $sourceTelemetryRoot -PathType Container)) {
        Show-StartupError "Required folder not found: $telemetryRoot, and copy source not found: $sourceTelemetryRoot."
    }

    try {
        Write-Host "$telemetryRoot not found. Copying Release_Internal_V1A to the R disk..."
        Copy-Item `
            -LiteralPath $sourceTelemetryRoot `
            -Destination "R:\" `
            -Recurse `
            -Force `
            -ErrorAction Stop
    } catch {
        Show-StartupError "Failed to copy Release_Internal_V1A to the R disk: $($_.Exception.Message)"
    }

    if (-not (Test-Path -LiteralPath $telemetryRoot -PathType Container)) {
        Show-StartupError "Copy completed but required folder is still missing: $telemetryRoot"
    }
}

$ramDiskConfirmation = Read-Host "User has set up R disk and placed 'Release_Internal_V1A' in it. Press 'y' to continue"
if ($ramDiskConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because the R disk setup was not confirmed."
}

$coilConfirmation = Read-Host "Coil is set to 10Hz AC Sine wave with current of 12A reading. REMEMBER: Slowly increase the current to 12A, or it will burn the driver. And REMEMBER to slowly decrease to zero for shutting off. Press 'y' to continue"
if ($coilConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because coil setup and safety was not confirmed."
}

$warmupConfirmation = Read-Host "Capsule is warmed up by taking photos continuously for at least 10 minutes so the sensor will not be badly affected by temperature difference. Press 'y' to continue"
if ($warmupConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because capsule warm-up was not confirmed."
}

$hardwareConfirmation = Read-Host "Recorder and capsule are powered on and recorder program is flashed. Press 'y' to continue"
if ($hardwareConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because recorder and capsule readiness was not confirmed."
}

$previousWindowsConfirmation = Read-Host "All previous PowerShell windows have terminated. Press 'y' to continue"
if ($previousWindowsConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because previous PowerShell window termination was not confirmed."
}

function Stop-LauncherProcesses {
    param(
        [int[]]$RootProcessIds
    )

    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $processIds = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($rootProcessId in $RootProcessIds) {
        if ($rootProcessId -gt 0) {
            [void]$processIds.Add($rootProcessId)
        }
    }

    $addedProcess = $true
    while ($addedProcess) {
        $addedProcess = $false
        foreach ($process in $processes) {
            if ($processIds.Contains([int]$process.ParentProcessId) -and
                $processIds.Add([int]$process.ProcessId)) {
                $addedProcess = $true
            }
        }
    }

    foreach ($processId in @($processIds)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

trap {
    Write-Host "`nLauncher error: $($_.Exception.Message)"
    Stop-LauncherProcesses -RootProcessIds @($childProcessIds)
    exit 1
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) {
    throw "Python was not found on PATH."
}
$python = $pythonCommand.Source

if (-not (Test-Path -LiteralPath $positionScript -PathType Leaf)) {
    throw "Position estimation script not found: $positionScript"
}

if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
    throw "Capsule orientation script not found: $script"
}

if (-not (Test-Path -LiteralPath $motorScript -PathType Leaf)) {
    throw "Motor control script not found: $motorScript"
}

if (-not (Test-Path -LiteralPath $telemetryRoot -PathType Container)) {
    throw "Telemetry directory not found: $telemetryRoot"
}

Write-Host "deleting output....."
if (Test-Path -LiteralPath $outputDirectory) {
    Remove-Item -LiteralPath $outputDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

Write-Host "clearing export....."
if (Test-Path -LiteralPath $exportDirectory) {
    Remove-Item -LiteralPath $exportDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $exportDirectory -Force | Out-Null

Write-Host "Checking linux connection set up"
& icacls.exe $exportDirectory /grant "*S-1-1-0:(OI)(CI)(RX)" /T /C | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply Everyone read permission to $exportDirectory."
}

$exportShare = Get-SmbShare -Name "CapNaviExport" -ErrorAction SilentlyContinue
if ($null -eq $exportShare) {
    New-SmbShare -Name "CapNaviExport" -Path $exportDirectory -ReadAccess "Everyone" -ErrorAction Stop | Out-Null
    $exportShare = Get-SmbShare -Name "CapNaviExport" -ErrorAction Stop
}
if ($null -eq $exportShare -or $exportShare.Path -ne $exportDirectory) {
    throw "SMB share CapNaviExport is missing or points to the wrong path. Expected: $exportDirectory"
}

$shareReadAccess = Get-SmbShareAccess -Name "CapNaviExport" -ErrorAction Stop |
    Where-Object {
        $_.AccountName -eq "Everyone" -and
        $_.AccessControlType -eq "Allow" -and
        $_.AccessRight -match "Read"
    }
if ($null -eq $shareReadAccess) {
    throw "SMB share CapNaviExport does not grant Everyone read access."
}

$exportAcl = Get-Acl -LiteralPath $exportDirectory -ErrorAction Stop
$ntfsReadAccess = $exportAcl.Access |
    Where-Object {
        $_.IdentityReference -match "Everyone|S-1-1-0" -and
        $_.AccessControlType -eq "Allow" -and
        $_.FileSystemRights -match "Read"
    }
if ($null -eq $ntfsReadAccess) {
    throw "Export folder does not grant Everyone NTFS read access."
}

if (-not (Test-Connection -ComputerName "192.168.50.2" -Count 1 -Quiet)) {
    throw "Linux machine 192.168.50.2 did not respond to ping."
}
Write-Host "Linux connection and export sharing checks passed."

Write-Host "FX3 setup checklist:"
Write-Host "1. Set Resolution to 1080x720"
Write-Host "2. Enable `"Image Information Frame`""
Write-Host "3. Disable `"Show Image`""
Write-Host "4. Press Start"
$fx3Confirmation = Read-Host "User should open FX3 program, configure the checklist, and press 'y' when complete"
if ($fx3Confirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because FX3 setup was not confirmed."
}

$connectionConfirmation = Read-Host "User has mounted the SMB share in the linux machine. Press 'y' to continue"
if ($connectionConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because the Windows/Linux connection was not confirmed."
}

$reconstructionConfirmation = Read-Host "User can now start the reconstruction module in linux machine. Enter 'y' to continue"
if ($reconstructionConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Launcher startup cancelled because reconstruction module confirmation was not 'y'."
}

$positionCommand = "& '$python' '$positionScript' --no-ml"

$positionProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-NoProfile", "-Command", $positionCommand) `
    -WorkingDirectory $root `
    -PassThru
$childProcessIds.Add($positionProcess.Id)

Start-Sleep -Seconds 2

if (Test-Path -LiteralPath $capsuleLog -PathType Leaf) {
    Remove-Item -LiteralPath $capsuleLog -Force
}

$capsuleCommand = "& '$python' -u '$script' --calibration-trials 19 --calibration-warmup-trials 0 2>&1 | Tee-Object -FilePath '$capsuleLog'"

$capsuleProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-NoProfile", "-Command", $capsuleCommand) `
    -WorkingDirectory $root `
    -PassThru
$childProcessIds.Add($capsuleProcess.Id)

$readyMessage = "TCP server listening on localhost:12347"
$startupTimeoutSeconds = 600
$deadline = (Get-Date).AddSeconds($startupTimeoutSeconds)

Write-Host "Waiting for capsule orientation server: $readyMessage"
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $capsuleLog -PathType Leaf) {
        $capsuleOutput = Get-Content -LiteralPath $capsuleLog -Raw -ErrorAction SilentlyContinue
        if ($capsuleOutput -and $capsuleOutput.Contains($readyMessage)) {
            break
        }
    }

    if ($capsuleProcess.HasExited) {
        throw "capsule_orientation.py exited before becoming ready. See $capsuleLog"
    }

    Start-Sleep -Milliseconds 500
}

$capsuleOutput = if (Test-Path -LiteralPath $capsuleLog -PathType Leaf) {
    Get-Content -LiteralPath $capsuleLog -Raw -ErrorAction SilentlyContinue
}
if (-not $capsuleOutput -or -not $capsuleOutput.Contains($readyMessage)) {
    throw "Timed out waiting for capsule orientation server after $startupTimeoutSeconds seconds. See $capsuleLog"
}

$photoCommand = "import serial, time; connection = serial.Serial('COM11', baudrate=800000, timeout=1, write_timeout=1); connection.write(b'at+photo=p,1,100000\r'); time.sleep(2); connection.write(b'at+photo+\r'); connection.close()"
Write-Host "Starting capsule photo capture..."
& $python -c $photoCommand
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start capsule photo capture on COM11."
}
Write-Host "Capsule photo capture started."

Write-Host ""
Write-Host "================================ ROBOT SAFETY CHECK ================================"
Write-Host "Before starting motorControl.py, confirm all of the following:"
Write-Host "1. Make sure the system is unobstructed."
Write-Host "2. Restart the motor power supply."
Write-Host "3. Restart the CAN bus USB connection."
Write-Host "====================================================================================="
$safetyConfirmation = Read-Host "Enter 'y' to continue and start motorControl.py"
if ($safetyConfirmation.Trim().ToLowerInvariant() -ne "y") {
    throw "Motor control startup cancelled because the safety confirmation was not 'y'."
}

$motorCommand = "& '$python' '$motorScript' --auto-scan --auto-nmt"

Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-NoProfile", "-Command", $motorCommand) `
    -WorkingDirectory $root `
    -PassThru | ForEach-Object { $childProcessIds.Add($_.Id) }
