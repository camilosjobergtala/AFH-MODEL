# debug_logs.py
from pathlib import Path

def debug_logs():
    log_dir = Path("./eclipse_results_v3/logs/")
    print(f"📁 Directorio logs existe: {log_dir.exists()}")
    
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        print(f"📄 Archivos log encontrados: {len(log_files)}")
        
        for log_file in log_files:
            size_kb = log_file.stat().st_size / 1024
            print(f"   {log_file.name}: {size_kb:.1f} KB")
            
            # Mostrar primeras líneas
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                print(f"   Líneas: {len(lines)}")
                if lines:
                    print(f"   Última: {lines[-1].strip()}")
            except Exception as e:
                print(f"   Error leyendo: {e}")

debug_logs()