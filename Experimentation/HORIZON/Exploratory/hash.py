#!/usr/bin/env python3
"""
Genera partición con hash SHA-256 para Registered Report
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ============================================
# CONFIGURACIÓN (debe coincidir con tu análisis)
# ============================================
SACRED_SEED = 2025
DEVELOPMENT_RATIO = 0.70

# Path a tus datos Sleep-EDF
SLEEP_EDF_PATH = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette"

# ============================================
# OBTENER SUBJECT IDs
# ============================================
def get_subject_ids(base_path):
    """Extrae IDs únicos de sujetos de los archivos PSG."""
    base = Path(base_path)
    psg_files = list(base.rglob("*PSG*.edf"))
    
    subject_ids = set()
    for f in psg_files:
        # Extraer primeros 6 caracteres como subject_id
        fname = f.stem
        subj_id = fname[:6] if len(fname) >= 6 else fname
        subject_ids.add(subj_id)
    
    return sorted(list(subject_ids))

# ============================================
# GENERAR PARTICIÓN
# ============================================
def generate_partition():
    # Obtener todos los subject IDs
    all_subjects = get_subject_ids(SLEEP_EDF_PATH)
    print(f"Total sujetos encontrados: {len(all_subjects)}")
    
    # Split determinístico (idéntico a tu análisis)
    rng = np.random.default_rng(SACRED_SEED)
    shuffled = rng.permutation(all_subjects)
    
    n_dev = int(len(all_subjects) * DEVELOPMENT_RATIO)
    development_ids = sorted(shuffled[:n_dev].tolist())
    holdout_ids = sorted(shuffled[n_dev:].tolist())
    
    print(f"Development: {len(development_ids)} sujetos")
    print(f"Holdout: {len(holdout_ids)} sujetos")
    
    # Crear documento de partición
    partition_doc = {
        "project": "Catch22_Consciousness_RR",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sacred_seed": SACRED_SEED,
        "development_ratio": DEVELOPMENT_RATIO,
        "database": "Sleep-EDF Expanded v1.0.0",
        "database_url": "https://physionet.org/content/sleep-edfx/1.0.0/",
        "n_total_subjects": len(all_subjects),
        "partition": {
            "development": {
                "n_allocated": len(development_ids),
                "subject_ids": development_ids
            },
            "holdout": {
                "n_allocated": len(holdout_ids),
                "subject_ids": holdout_ids
            }
        }
    }
    
    # Calcular hash (sin el campo hash)
    json_str = json.dumps(partition_doc, sort_keys=True, ensure_ascii=False)
    sha256_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    # Agregar hash al documento
    partition_doc["sha256_hash"] = sha256_hash
    
    # Guardar
    output_path = Path("catch22_partition.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(partition_doc, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"SHA-256 HASH:")
    print(f"{sha256_hash}")
    print(f"{'='*60}")
    print(f"\nArchivo guardado: {output_path.absolute()}")
    
    # También guardar hash solo
    hash_path = Path("catch22_partition_HASH.txt")
    with open(hash_path, 'w') as f:
        f.write(f"SHA-256: {sha256_hash}\n")
        f.write(f"Generated: {partition_doc['created_utc']}\n")
        f.write(f"Seed: {SACRED_SEED}\n")
    
    print(f"Hash guardado: {hash_path.absolute()}")
    
    return sha256_hash

if __name__ == "__main__":
    generate_partition()