
import subprocess
import time
import re
import threading
import sys
import os

def get_swap_usage():
    try:
        output = subprocess.check_output(["sysctl", "vm.swapusage"]).decode("utf-8")
        # vm.swapusage: total = 3072.00M  used = 1536.00M  free = 1536.00M  (encrypted)
        match = re.search(r"used = (\d+\.\d+)M", output)
        if match:
            return float(match.group(1)) / 1024  # Convert MB to GB
    except:
        pass
    return 0.0

def get_ram_usage():
    try:
        # simpler: use psutil if available, else subprocess top/vm_stat
        # But here we replace psutil entirely
        # vm_stat gives pages. pages * 16384 (on M1 generally 16K pages? No 4K or 16K)
        # easier: top -l 1 | grep PhysMem
        output = subprocess.check_output("top -l 1 | grep PhysMem", shell=True).decode("utf-8")
        # PhysMem: 14G used (2638M wired), 2004M unused.
        used_match = re.search(r"(\d+)([MG]) used", output)
        if used_match:
            val = float(used_match.group(1))
            unit = used_match.group(2)
            if unit == "M": val /= 1024
            return val
    except:
        pass
    return 0.0

def monitor(stop_event):
    print(f"{'='*20} MONITOR STARTED (Native Mac) {'='*20}")
    while not stop_event.is_set():
        try:
            ram_gb = get_ram_usage()
            swap_gb = get_swap_usage()
            print(f"[MONITOR] RAM Used: {ram_gb:.2f}GB | Swap Used: {swap_gb:.2f}GB")
        except Exception as e:
            print(f"[MONITOR] Error: {e}")
        time.sleep(2)
    print(f"{'='*20} MONITOR STOPPED {'='*20}")

if __name__ == "__main__":
    stop_event = threading.Event()
    t = threading.Thread(target=monitor, args=(stop_event,))
    t.start()

    try:
        # Run the profile command
        # We use unbuffered output to see logs in real-time if possible, 
        # though subprocess.run captures it at the end unless we pipe it.
        # We'll just let it print to stdout which we capture.
        cmd = ["uv", "run", "profile_inference.py", "--no-warmup", "--example", "example_05.json", "--offload-cpu"]
        print(f"Running command: {' '.join(cmd)}")
        
        # We stream output to ensure we capture it alongside monitor logs
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read stdout line by line
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nExecution failed: {e}")
    finally:
        stop_event.set()
        t.join()
