import os
import glob
import time
import serial

output_dir = r"R:\Release_Internal_V1A\output"

print("--- Snapshot of existing imgInfo_*.txt files ---")
existing_files = set(glob.glob(os.path.join(output_dir, "imgInfo_*.txt")))
print(f"Count of existing imgInfo_*.txt: {len(existing_files)}")

print("Opening serial port COM11 at 800000 baud, timeout=0.1s...")
try:
    ser = serial.Serial(port="COM11", baudrate=800000, timeout=0.1)
    ser.reset_input_buffer()
    print("Serial port COM11 opened and input buffer reset successfully.")
except Exception as e:
    print(f"Exception during serial port opening: {e!r}")
    exit(1)

cmd1 = b"at+photo=p,1,2\r"
print(f"Sending: {cmd1!r}")
try:
    ser.write(cmd1)
except Exception as e:
    print(f"Serial Exception during write of first command: {e!r}")
    ser.close()
    exit(1)

print("Reading until 'Total image number' appears...")
total_image_found = False
buffer = b""
start_time = time.time()
while time.time() - start_time < 10.0:
    try:
        data = ser.read(1024)
    except Exception as e:
        print(f"Serial Exception during read: {e!r}")
        ser.close()
        exit(1)
    if data:
        buffer += data
        if b"Total image number" in buffer:
            total_image_found = True
            print("Found 'Total image number'!")
            try:
                print(buffer.decode("utf-8", errors="replace"))
            except Exception:
                pass
            break
    else:
        time.sleep(0.01)

if not total_image_found:
    print("Warning: 'Total image number' not found within 10 seconds. Buffer contents:")
    print(buffer.decode("utf-8", errors="replace"))

cmd2 = b"at+photo+\r"
print(f"Sending: {cmd2!r}")
try:
    ser.write(cmd2)
except Exception as e:
    print(f"Serial Exception during write of second command: {e!r}")
    ser.close()
    exit(1)

print("Reading raw serial data for up to 8 seconds, timestamping reads...")
buf_line = ""
start_time_8s = time.time()
while time.time() - start_time_8s < 8.0:
    try:
        data = ser.read(4000)
    except Exception as e:
        print(f"Serial Exception during 8-second read: {e!r}")
        break
    
    if data:
        t = time.time()
        text = data.decode("utf-8", errors="replace")
        buf_line += text
        while "\n" in buf_line:
            line, buf_line = buf_line.split("\n", 1)
            line_str = line.strip("\r")
            if any(kw in line_str for kw in ["Image-Start", "ImgCt", "Image Done", "All Image Done"]):
                print(f"[{t:.6f}] {line_str}")
    else:
        time.sleep(0.01)

if buf_line:
    line_str = buf_line.strip("\r")
    if any(kw in line_str for kw in ["Image-Start", "ImgCt", "Image Done", "All Image Done"]):
        print(f"[{time.time():.6f}] {line_str}")

try:
    ser.close()
    print("Serial port closed.")
except Exception as e:
    print(f"Serial Exception during close: {e!r}")

print("Checking for newly appearing imgInfo_*.txt files...")
current_files = set(glob.glob(os.path.join(output_dir, "imgInfo_*.txt")))
new_files = current_files - existing_files

print(f"Number of new files: {len(new_files)}")
for file_path in sorted(new_files):
    filename = os.path.basename(file_path)
    try:
        size = os.path.getsize(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            f.read(100)
        print(f"New File: {filename} | Opened successfully | Size: {size} bytes")
    except Exception as e:
        print(f"New File: {filename} | Failed to open/read: {e} | Size: unknown")
