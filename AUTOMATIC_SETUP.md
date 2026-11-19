# CapNavi Automatic Demo Setup

This guide explains how to start the CapNavi robot demo with the one-click
PowerShell launcher.

The launcher is:

```powershell
.\start_demo.ps1
```

Run the command from:

```text
C:\Users\johncklau\Desktop\CapNaviSystem
```

The launcher opens separate PowerShell windows for the position-estimation,
capsule-orientation, and motor-control programs.

## Before starting

Prepare the following items before running `start_demo.ps1`:

1. Set up the `R:` RAM disk.
2. Make sure `R:\Release_Internal_V1A` is present.
   - If the folder is missing and the local source folder exists, the launcher
     copies it from:

     ```text
     C:\Users\johncklau\Desktop\Release_Internal_V1A
     ```

3. Set the coil to a 10 Hz AC sine wave.
4. Prepare the coil current at a 12 A reading.
5. Make sure the current is increased slowly to 12 A. Increasing it too
   quickly can burn the driver.
6. Remember to decrease the current slowly to zero when shutting down.
7. Warm up the capsule by taking photos continuously for at least 10 minutes.
   This reduces the effect of temperature differences on the sensor.
8. Power on the recorder and capsule.
9. Flash the recorder program.
10. The launcher closes visible PowerShell windows from previous runs.
11. The launcher closes previous FX3 windows and opens FX3 automatically. Set
    up the FX3 program:
    - Resolution: `1080x720`
    - Enable **Image Information Frame**
    - Disable **Show Image**
    - Press **Start**
12. Mount the Windows SMB share and start the reconstruction algorithm on the
    Linux machine.

## Windows and Linux network connection

The Windows computer and Ubuntu computer use a direct Ethernet connection.
Use these static IPv4 settings:

| Computer | IPv4 address | Subnet mask | Default gateway | DNS |
| --- | --- | --- | --- | --- |
| Windows producer | `192.168.50.1` | `255.255.255.0` | Leave blank | Leave blank |
| Ubuntu reconstruction computer | `192.168.50.2` | `255.255.255.0` | Leave blank | Leave blank |

### Configure the Windows Ethernet adapter

1. Open **Settings -> Network & internet -> Advanced network settings**.
2. Select **More network adapter options**.
3. Right-click the wired Ethernet adapter and select **Properties**.
4. Open **Internet Protocol Version 4 (TCP/IPv4)** -> **Properties**.
5. Select **Use the following IP address** and enter:

   ```text
   IP address:    192.168.50.1
   Subnet mask:   255.255.255.0
   Gateway:       blank
   DNS servers:   blank
   ```

6. Set the network profile to **Private**.
7. Enable **Network discovery** and **File and printer sharing** in Windows
   advanced sharing settings.
8. Allow **File and Printer Sharing** through the Windows Defender Firewall.

### Create the Windows SMB share manually

The launcher creates this share automatically when run as Administrator:

```text
Share name: CapNaviExport
Folder:     R:\Release_Internal_V1A\export
Network path: \\LAPTOP-SGFPG505\CapNaviExport
Access:     Everyone - Read
```

To create it manually, right-click
`R:\Release_Internal_V1A\export` -> **Properties -> Sharing -> Advanced
Sharing**, check **Share this folder**, and use the share name
`CapNaviExport`. In **Permissions**, grant `Everyone` **Read** access. On the
**Security** tab, also grant `Everyone` read access to the folder.

The share name must match exactly. A share named `capnavi` is a different
configuration from the launcher’s expected `CapNaviExport`.

### Verify the connection

From Ubuntu, verify the Windows host first:

```bash
ping 192.168.50.1
```

Mount the share at `/mnt/capnavi`:

```bash
sudo mkdir -p /mnt/capnavi
sudo mount -t cifs //192.168.50.1/CapNaviExport /mnt/capnavi \
    -o guest,ro,vers=3.0
ls -la /mnt/capnavi
```

The directory should contain the exported capture bundles. If Windows
returns `mount error(13): Permission denied`, guest SMB access is disabled;
use a Windows account credentials file instead of the `guest` option.

## Startup prompts

Each confirmation requires entering `y` or `Y` and pressing Enter.

### 1. R disk check

The launcher first checks that:

```text
R:
R:\Release_Internal_V1A
```

exist. If `R:\Release_Internal_V1A` is missing, it copies the folder from:

```text
C:\Users\johncklau\Desktop\Release_Internal_V1A
```

The launcher stops with an error if the R disk, source folder, or copy is
unavailable. No user confirmation is required for this step.

### 2. Coil safety

```text
Coil is set to 10Hz AC Sine wave with current of 12A reading.
REMEMBER: Slowly increase the current to 12A, or it will burn the driver.
And REMEMBER to slowly decrease to zero for shutting off.
Press 'y' to continue
```

Do not continue until the coil and driver safety procedure is understood.

### 3. Capsule warm-up

```text
Capsule is warmed up by taking photos continuously for at least 10 minutes
so the sensor will not be badly affected by temperature difference.
Press 'y' to continue
```

### 4. Recorder and capsule

```text
Recorder and capsule are powered on and recorder program is flashed.
Press 'y' to continue
```

### 5. Previous PowerShell windows

The launcher automatically closes visible PowerShell windows from previous
runs. It excludes the active launcher process and does not ask for confirmation.
This prevents old processes from holding serial ports, TCP ports, cameras, or
the CAN bus converter.

## Automatic folder and Linux connection setup

The launcher performs these actions automatically:

1. Deletes and recreates:

   ```text
   R:\Release_Internal_V1A\output
   ```

2. Deletes and recreates:

   ```text
   R:\Release_Internal_V1A\export
   ```

3. Applies `Everyone` read access to the export folder.
4. Verifies the SMB share:

   ```text
   Share name: CapNaviExport
   Share path: R:\Release_Internal_V1A\export
   Share access: Everyone - Read
   ```

5. Recreates the SMB share if it was removed when the export folder was
   recreated.
6. Pings the Linux machine:

   ```text
   192.168.50.2
   ```

Successful checks produce:

```text
Checking linux connection set up
Linux connection and export sharing checks passed.
```

### FX3 prompt

The launcher closes any existing `FX3Display.exe` process and opens:

```text
R:\Release_Internal_V1A\FX3Display.exe
```

The launcher then displays:

```text
FX3 setup checklist:
1. Set Resolution to 1080x720
2. Enable "Image Information Frame"
3. Disable "Show Image"
4. Press Start
```

Configure the newly opened FX3 program and confirm with `y` after it is ready.

### Linux algorithm prompt

The launcher asks:

```text
Start algorithm on linux machine. Mount the SMB share and start the reconstruction algorithm, then press 'y' to continue
```

Mount the `CapNaviExport` share and start the reconstruction algorithm on Linux
before confirming this single prompt.

## Position estimation and capsule orientation

After the preparation prompts, the launcher starts:

```powershell
python .\magFieldOptimize_PosEst.py --no-ml
```

It then starts capsule orientation with:

```powershell
python -u .\capsule_orientation.py `
    --calibration-trials 19 `
    --calibration-warmup-trials 0
```

The capsule-orientation output is saved to:

```text
C:\Users\johncklau\Desktop\CapNaviSystem\capsule_orientation_startup.log
```

For diagnosing unexpected angle transitions, capsule orientation also writes
one row per processed sample to:

```text
C:\Users\johncklau\Desktop\CapNaviSystem\orientation_debug_log.csv
```

The CSV includes sample timing, yaw/pitch/roll, yaw change and rate,
quaternions before and after output smoothing, raw and corrected gyro values,
accelerometer values, raw/lock-in/high-pass/low-pass magnetic vectors,
magnetic ratio and weight, gradient error, and the gyro-predicted field.
`sample_dt_s` records the original telemetry interval. The filter uses a
maximum timestep of `0.05` seconds, and the `filter_dt_limited` column marks
samples where a larger telemetry gap was capped.
The sensor-correction rate is limited to `4` degrees per second, and
`sensor_correction_limited` marks samples where that limit was applied.
Raw gyro input is limited to `30` degrees per second per axis before it is
used by the estimator. The CSV keeps both `raw_gyro_*_dps` and the applied
`gyro_*_dps` values, and `gyro_input_limited` identifies clipped samples.

The magnetic ratio remains available for diagnostics, but it does not control
the filter weight. When valid magnetic measurements are present, the filter
uses a constant magnetic correction weight of `15%`.

When option 4 starts, `motorControl.py` creates and writes the timestamped
motor-reference file:

```text
C:\Users\johncklau\Desktop\CapNaviSystem\motor_motion_debug_log.csv
```

Logging is disabled during setup and other motor-control options, and the file
is reset when option 4 starts. The orientation diagnostic log reads this file
and records the actual rotary angle, the idealized `0 -> +/-180 -> 0` expected
Euler-yaw path, and the wrapped model error. Positive rotary angles map to
negative capsule yaw; for example, a motor angle of `+112` degrees maps to
an expected capsule yaw of `-112` degrees. A motor angle of `206` degrees
reports `+154` degrees because Euler yaw has wrapped past `-180` degrees.

The fresh motor reference also acts as a strong sudden-change guard for the
published capsule yaw. If the Madgwick output changes faster than the
motor-supported rate, the published quaternion is limited to the expected
motor direction plus a small tolerance. This guard alone does not overwrite the
internal Madgwick state. If the capsule remains more than 30 degrees from the motor
model for at least one second, a one-second recovery window allows up to
12 degrees per second when the capsule moves in the motor direction. Opposite
direction jumps remain blocked. The diagnostic CSV records
`yaw_before_motor_guard_deg`, `motor_reference_age_s`,
`motor_expected_rate_dps`, `motor_guard_applied`, and
`motor_guard_reason`. When the capsule has more than 20 degrees of absolute
error, persistent movement opposite to the motor direction is rejected even
if that movement is slower than the normal rate limit. The diagnostic CSV also
records
`motor_recovery_active`.
The opposite-direction hold is limited to 0.5 seconds; after that, the guard
records `opposite_direction_released` and allows the raw estimate through so a
temporarily stuck capsule cannot remain frozen indefinitely.

When the capsule remains more than 20 degrees from the expected motor yaw for
at least one second, a separate position-recovery window can move the
published yaw toward the motor pose at no more than 3 degrees per second for
up to 20 seconds. This correction is applied only while the motor is
stationary, so a capsule temporarily stuck while the motor is moving is not
forced toward a moving target. The correction is bounded and only changes the
published quaternion; this recovery does not itself change the internal Madgwick
state. The
diagnostic CSV records `motor_position_recovery_active` and
`motor_position_recovery_applied`.

Internal motor-yaw correction is enabled by default. It matches each
clock-calibrated capsule sample to the nearest recorded motor
sample, provided the motor logger is still fresh and the matched sample is
within 0.15 seconds. A wrapped filter-yaw error of at least 30 degrees observed for
at least one second applies a nudge of up to 10 degrees per second to Madgwick's
internal yaw while preserving roll and pitch. If the absolute wrapped internal
yaw error reaches 60 degrees, an emergency correction immediately sets both
Madgwick yaw states to the matched motor yaw, including near a half-turn; it
does not wait for the one-second timer or obey the 10-degree-per-second limit.
This can override a genuinely different physical capsule yaw. The existing
output smoothing and guards still apply after the internal snap, so published
yaw need not jump to the motor yaw immediately. Sample gaps of up to one second
since the last valid sample preserve
the accumulated valid waiting time; the gap itself does not count toward that
time and receives no nudge. Longer gaps, invalid timestamps, invalid motor
references, or an error below the threshold reset the timer. The filter timestep
cap remains 0.05 seconds. There is no total correction
or duration limit while the conditions remain met: a stuck capsule may be
pulled toward the motor model even when its physical yaw differs. For
diagnostic-only operation, start the capsule with:

```powershell
python capsule_orientation.py --disable-motor-yaw-correction
```

The fields
`motor_diagnostic_alignment_s`, `motor_diagnostic_filter_error_deg`,
`motor_diagnostic_correction_deg`, `motor_diagnostic_status`,
`motor_diagnostic_reference_timestamp`, `motor_diagnostic_expected_yaw_deg`,
and `motor_filter_yaw_before_correction_deg` describe each sample. Status
`applied` means the nudge was applied to the filter; `snap_applied` means the
emergency correction was applied. In diagnostic-only mode, `proposed` and
`snap_proposed` report what would have been applied. For a snap,
`motor_diagnostic_correction_deg` records the wrapped internal yaw change.
Skipping time calibration normally prevents the correction from activating
because timestamps cannot be aligned.

The log also records `yaw_without_motor_correction_deg`, produced by a
separate Madgwick and lock-in estimator using the same sensor inputs and
calibration events but no motor-yaw correction, rate guard, or position
recovery. Compare it with `yaw_before_motor_guard_deg` and `yaw_deg`; their
differences can include the existing output guards as well as internal motor
correction. This shadow estimate does not control the robot or prove the
capsule's physical orientation. A `nan` indicates shadow tracking or
calibration failed; the capsule terminal reports the failure.

The launcher waits for:

```text
TCP server listening on localhost:12347
```

Only after this message does it continue.

After the TCP server is ready, the launcher sends these serial commands to
`COM11` at `800000` baud:

```text
at+photo=p,1,100000
```

It waits two seconds, then sends:

```text
at+photo+
```

## Motor safety confirmation

Before starting motor control, the launcher displays:

```text
Before starting motorControl.py, confirm all of the following:
1. Make sure the system is unobstructed.
2. Restart the motor power supply.
3. Restart the CAN bus USB connection.
```

Enter `y` to continue.

The launcher then starts:

```powershell
python .\motorControl.py --auto-scan --auto-nmt
```

## Automatic motor sequence

Starting `motorControl.py` directly also runs option 1 automatically, then
option 2 with state `01` after the converter is configured. If the scan fails,
use option 1 to retry. The automatic NMT command skips the motor-power prompt
and option 2 performs a brief vertical pre-move, so ensure the system is
unobstructed and motor power has been restarted before starting the module.
The `--auto-nmt` flag additionally enables the later guided demo prompts.

### Option 1: Connect the CAN bus converter

The motor controller automatically:

- Scans for the CAN bus converter.
- Sets the CAN bitrate.
- Starts `GS_CAN_MODE_NORMAL`.
- Starts the CAN reader thread.
- Starts the SYNC thread.
- Starts vertical synchronization monitoring.
- Starts motor position printing.

### Option 2: Set NMT state

After a successful scan, option 2 runs automatically with:

```text
NMT state: 01
```

The motor-power confirmation is automatically answered with `y`.

### Option 10: Automatic zeroing

The motor terminal asks:

```text
CAN bus is configured and NMT state 01 is active.
Enter 'y' to start automatic zeroing (option 10):
```

Enter `y` to start automatic zeroing.

### Option 11: Rotate to 1000

After option 10, the launcher waits 15 seconds:

```text
Waiting 15 seconds before allowing rotation to start...
```

It then asks:

```text
Make sure the motors all stopped zeroing.
Enter 'y' to start rotation (option 11):
```

Enter `y` only after confirming that zeroing has stopped.

### Demo confirmation

After option 11, the motor terminal asks:

```text
Make sure all motors have stopped moving.
Confirm starting demo sequence (option 4).
Press 'y' to start demo:
```

Enter `y` to continue.

### Option 13: Set motor offsets

After the demo confirmation, option 13 runs automatically. It records the
current position of each motor as its offset.

No confirmation is required for option 13.

### Option 4: Start the demo

Option 4 then starts the demo movement sequence immediately.

At the same time, a background timer starts. After 10 seconds, the launcher
starts:

```powershell
python .\aprilTagLocalization.py `
    --bt_only `
    --export-min-distance-mm 7 `
    --export-min-angle-deg 7
```

The motor terminal displays:

```text
Demo sequence started. AprilTag export will start in 10 seconds.
```

The demo movement and delayed AprilTag/export startup run concurrently.

After the 10-second delay, the motor controller sends a calibration request
to `capsule_orientation.py` before starting AprilTag localization. The request
uses a workspace signal file so it works across the separate PowerShell
windows; manual `c` key calibration remains available. A request remains
pending until the lock-in field estimate is ready, so a request arriving
during startup is retried instead of being discarded.

After the final inspection revolution completes, AprilTag localization writes
this end-of-inspection manifest to the shared export directory:

```text
R:\Release_Internal_V1A\export\inspection_complete.json
```

The manifest is written atomically after all queued capture exports finish and
contains:

```json
{
  "schema_version": 1,
  "frame_type": "inspection_end",
  "event": "inspection_complete",
  "timestamp_unix": 0,
  "timestamp_utc": "2026-01-01T00:00:00Z",
  "source": "aprilTagLocalization.py"
}
```

The Linux reconstruction algorithm can watch for
`inspection_complete.json` and treat its creation as the end of the
inspection. The file is recreated for each new run after the export directory
is cleaned.

## Stopping the system

Press `Ctrl+C` in the main launcher terminal. The launcher attempts to stop
the PowerShell windows and Python processes that it started, including:

- Position estimation
- Capsule orientation
- Motor control
- AprilTag localization

Do not use process-name-based termination commands because other Python or
PowerShell programs may be running.

After stopping:

1. Slowly reduce coil current to zero.
2. Stop the Linux reconstruction module if required.
3. Close remaining application windows.
4. Confirm the CAN bus and motor system are in a safe state.

## Common problems and fixes

### The launcher says the R drive is unavailable

Set up the ImDisk RAM disk first, then run the launcher again.

### `Release_Internal_V1A` is missing

The launcher attempts to copy it from:

```text
C:\Users\johncklau\Desktop\Release_Internal_V1A
```

If that source folder is also missing, create the R disk and copy the folder
manually.

### The launcher displays a UAC prompt

This is expected. Administrator privileges are required to create or recreate
the `CapNaviExport` SMB share.

### The SMB share check fails

Verify:

```text
Share name: CapNaviExport
Share path: R:\Release_Internal_V1A\export
Everyone: Read
```

Also verify that the Linux machine mounts the correct share.

### Ping to `192.168.50.2` fails

Check:

- The Linux machine is powered on.
- The Ethernet connection is connected.
- The Windows and Linux IP addresses are on the same network.
- The Linux firewall allows ICMP and SMB traffic.
- The address is still `192.168.50.2`.

### Capsule orientation does not reach the TCP-ready message

Inspect:

```text
C:\Users\johncklau\Desktop\CapNaviSystem\capsule_orientation_startup.log
```

Check:

- `COM11` is available.
- The recorder is powered on.
- The recorder program is flashed.
- The FX3 program is configured correctly.
- The FX3 program has been started.
- New files are being written to:

  ```text
  R:\Release_Internal_V1A\output
  ```

### The launcher reports a startup error and waits

Read the line beginning with:

```text
Launcher error:
```

Press Enter only after recording the error. Correct the reported setup issue
and run `start_demo.ps1` again.

## Notes for future troubleshooting

Add additional problems and fixes below:

- Problem:
  - Cause:
  - Fix:

- Problem:
  - Cause:
  - Fix:

- Problem:
  - Cause:
  - Fix:
