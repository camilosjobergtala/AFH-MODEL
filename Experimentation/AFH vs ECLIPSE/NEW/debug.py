#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BÚSQUEDA: Encontrar directorio con archivos Sleep-EDF
"""

from pathlib import Path

# Posibles ubicaciones
POSSIBLE_PATHS = [
    Path(r"G:\Mi unidad\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette"),
    Path(r"G:\Mi unidad\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF"),
    Path(r"G:\Mi unidad\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF"),
    Path(r"G:\Mi unidad\AFH\EXPERIMENTO\FASE 2"),
    Path(r"G:\Mi unidad\AFH\EXPERIMENTO"),
    Path(r"G:\Mi unidad\AFH"),
]

print("="*80)
print("BÚSQUEDA: ARCHIVOS SLEEP-EDF")
print("="*80)

for base_path in POSSIBLE_PATHS:
    print(f"\n{'─'*80}")
    print(f"Explorando: {base_path}")
    print(f"{'─'*80}")
    
    if not base_path.exists():
        print("  ✗ Directorio no existe")
        continue
    
    print("  ✓ Directorio existe")
    
    # Buscar PSG
    psg_files = list(base_path.glob("**/*PSG.edf"))
    print(f"  PSG files (recursivo): {len(psg_files)}")
    
    if len(psg_files) > 0:
        print(f"\n  Primeros 5 PSG encontrados:")
        for i, psg in enumerate(psg_files[:5], 1):
            print(f"    {i}. {psg.relative_to(base_path)}")
        
        # Buscar Hypnogram
        hypno_files = list(base_path.glob("**/*Hypnogram.edf"))
        print(f"\n  Hypnogram files (recursivo): {len(hypno_files)}")
        
        if len(hypno_files) > 0:
            print(f"\n  Primeros 5 Hypnogram encontrados:")
            for i, hypno in enumerate(hypno_files[:5], 1):
                print(f"    {i}. {hypno.relative_to(base_path)}")
        
        # Identificar directorio correcto
        if len(psg_files) > 0 and len(hypno_files) > 0:
            # Directorio común
            common_parent = psg_files[0].parent
            print(f"\n  ✅ DIRECTORIO CORRECTO ENCONTRADO:")
            print(f"     {common_parent}")
            break

print("\n" + "="*80)
print("BÚSQUEDA COMPLETA")
print("="*80)