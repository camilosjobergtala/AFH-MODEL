from pathlib import Path

data_path = Path("G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette/")

psg_files = list(data_path.rglob("*PSG*.edf"))
hyp_files = list(data_path.rglob("*Hypnogram*.edf"))

print(f"PSG files: {len(psg_files)}")
print(f"Hypnogram files: {len(hyp_files)}")

# Crear diccionario
hyp_dict = {}
for hyp in hyp_files:
    base = hyp.stem.split('-')[0].split('_')[0]
    if len(base) >= 6:
        hyp_dict[base[:6]] = hyp
        hyp_dict[base] = hyp

# Verificar matches
matches = 0
for psg in psg_files[:10]:  # Primeros 10
    base = psg.stem.split('-')[0].split('_')[0]
    subject_base = base[:6] if len(base) >= 6 else base
    
    found = None
    for variant in [base, subject_base, base.replace('E', 'C'), subject_base + 'C']:
        if variant in hyp_dict:
            found = hyp_dict[variant]
            break
    
    if found:
        matches += 1
        print(f"✓ {psg.name} → {found.name}")
    else:
        print(f"✗ {psg.name} → NO MATCH")

print(f"\nMatches: {matches}/{min(10, len(psg_files))}")