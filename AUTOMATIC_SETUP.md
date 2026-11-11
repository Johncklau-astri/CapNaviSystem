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
10. Close all PowerShell windows from previous runs.
11. Set up the FX3 program:
    - Resolution: `1080x720`
    - Enable **Image Information Frame**
    - Disable **Show Image**
    - Press **Start**
12. Mount the Windows SMB share from the Linux machine.
13. Ensure the Linux reconstruction module is ready to start.

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

### 1. R disk

```text
User has set up R disk and placed 'Release_Internal_V1A' in it. Press 'y' to continue
```

The launcher checks that:

```text
R:
R:\Release_Internal_V1A
```

exist before showing this prompt.

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

```text
All previous PowerShell windows have terminated. Press 'y' to continue
```

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

The launcher then displays:

```text
FX3 setup checklist:
1. Set Resolution to 1080x720
2. Enable "Image Information Frame"
3. Disable "Show Image"
4. Press Start
```

Confirm with `y` after the FX3 program is configured.

### Linux prompts

The launcher asks for confirmation that the SMB share has been mounted:

```text
User has mounted the SMB share in the linux machine. Press 'y' to continue
```

It then asks:

```text
User can now start the reconstruction module in linux machine.
Enter 'y' to continue
```

Start the reconstruction module on Linux after this prompt and before
continuing.

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
