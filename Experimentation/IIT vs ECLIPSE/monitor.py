# monitor_progress.py (ejecuta en otra terminal)
import time
from pathlib import Path

def monitor_progress():
    log_file = Path("./eclipse_results_v3/logs/")
    latest_log = max(log_file.glob("*.log"), key=lambda x: x.stat().st_mtime)
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Últimas líneas del log:")
    for line in lines[-5:]:
        print(f"   {line.strip()}")
    
    checkpoint_file = Path("./eclipse_results_v3/processing_v3_2ch_all_combined.pkl")
    if checkpoint_file.exists():
        size_mb = checkpoint_file.stat().st_size / (1024*1024)
        print(f"💾 Checkpoint: {size_mb:.1f} MB")

while True:
    monitor_progress()
    time.sleep(300)  # Cada 5 minutos