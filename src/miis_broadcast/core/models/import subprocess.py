import subprocess
import time
import os
import datetime

def get_gpu_processes():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            encoding="utf-8"
        )
        processes = []
        for line in result.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    processes.append({
                        "pid": parts[0],
                        "name": parts[1],
                        "vram": parts[2]
                    })
        return processes
    except Exception as e:
        print(f"Error fetching GPU processes: {e}")
        return []

def get_system_vram():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader"],
            encoding="utf-8"
        )
        parts = [p.strip() for p in result.strip().split('\n')[0].split(',')]
        if len(parts) == 3:
            return {
                "total": parts[0],
                "used": parts[1],
                "free": parts[2]
            }
    except Exception as e:
        pass
    return None

def main():
    print("=" * 60)
    print("🚀 VRAM Monitor Started")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] 📊 System VRAM Status")
            print("=" * 60)
            
            sys_vram = get_system_vram()
            if sys_vram:
                print(f"🌍 Overall GPU VRAM:")
                print(f"   Total: {sys_vram['total']} | Used: {sys_vram['used']} | Free: {sys_vram['free']}")
            else:
                print("Could not get overall GPU VRAM.")
                
            print("-" * 60)
            print("🔬 Per-Process VRAM Usage:")
            procs = get_gpu_processes()
            if not procs:
                print("   No processes found using GPU.")
            else:
                # Sort by VRAM usage descending
                procs.sort(key=lambda x: int(x['vram'].replace(' MiB', '')) if 'MiB' in x['vram'] else 0, reverse=True)
                for p in procs:
                    print(f"   PID: {p['pid']:<8} | VRAM: {p['vram']:<10} | Name: {p['name']}")
                    
            print("=" * 60)
            print("Tracking LiveCC and other models. LiveCC internal PyTorch VRAM is also logged when generation stops in LiveCC's console.")
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting VRAM Monitor...")

if __name__ == "__main__":
    main()
