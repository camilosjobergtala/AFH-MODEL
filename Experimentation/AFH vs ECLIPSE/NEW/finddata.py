from pathlib import Path

# Buscar desde la raíz de tu proyecto
search_dir = Path(r"G:/Mi unidad/AFH")

print("Buscando archivos CSV con 'score' o 'subject' en el nombre...")
print("="*80)

csv_files = list(search_dir.rglob("*score*.csv")) + list(search_dir.rglob("*subject*.csv"))

if csv_files:
    print(f"Encontrados {len(csv_files)} archivos:\n")
    for i, f in enumerate(csv_files, 1):
        print(f"{i}. {f}")
        print(f"   Tamaño: {f.stat().st_size / 1024:.1f} KB")
        print(f"   Modificado: {f.stat().st_mtime}")
        print()
else:
    print("No se encontraron archivos.")
    
print("="*80)