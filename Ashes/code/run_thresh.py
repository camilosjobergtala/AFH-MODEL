# -*- coding: utf-8 -*-
"""
Reproduces Table 9 (§4.4 of "An admissibility test for constitutive claims
about phenomenal presence...", C. A. Sjöberg Tala): sensitivity of the
four-outcome classification to the equivalence threshold delta_min, at
three values (0.005, 0.01, 0.02), with the chain margin scaled
proportionally, delta_epi = 2*delta_min (Table 9's caption), and 150 studies
per cell (S9's documented deviation from the 200-study default).

Reuses run_outcomes.py's scenario definitions and run_study() directly on
the four scenarios Table 9 reports: absent return with an adequate sample,
an effect inside the equivalence zone, a present return, and latent
confounding -- chosen because they are the pair that carries the argument
(return and latent confounding both survive at every threshold) plus the
two boundary-sensitive cases.

RUN: python3 run_thresh.py [delta_min] [n_studies]
     python3 run_thresh.py 0.005 150
     python3 run_thresh.py 0.01  150
     python3 run_thresh.py 0.02  150
TIME: ~5 minutes per threshold on a single core.
"""
import json
import sys
import time
from pathlib import Path

from run_outcomes import SCENARIOS, DISPLAY, run_study

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
THRESH_SCENARIOS = [
    "absent_return_adequate", "effect_within_zone", "return_present", "latent_confounding",
]


def main():
    delta_min = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    n_studies = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    delta_epi = 2 * delta_min  # Table 9 caption: chain margin scaled proportionally

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hdr = f"{'scenario':40s} {'contra':>7s} {'survived':>9s} {'inconcl':>8s} {'not eval':>9s}"
    print(f"delta_min={delta_min}  delta_epi={delta_epi}  n_studies={n_studies}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for key in THRESH_SCENARIOS:
        cfg = SCENARIOS[key]
        t0 = time.time()
        outcomes = []
        for i in range(n_studies):
            outcome, _ = run_study(i, cfg["subjects"], cfg["trials"], cfg["gen"],
                                    cfg.get("permute_control", False),
                                    delta_min=delta_min, delta_epi=delta_epi)
            outcomes.append(outcome)
        counts = {k: outcomes.count(k) for k in ("contradicted", "survived", "inconclusive", "not_evaluable")}
        pct = {k: 100.0 * v / n_studies for k, v in counts.items()}
        print(f"{DISPLAY[key]:40s} {pct['contradicted']:7.1f} {pct['survived']:9.1f} "
              f"{pct['inconclusive']:8.1f} {pct['not_evaluable']:9.1f}  ({time.time()-t0:.0f}s)")
        rows.append({"scenario": key, "display_name": DISPLAY[key], **pct})

    with open(OUT_DIR / f"thresh_{delta_min}.json", "w") as f:
        json.dump({"delta_min": delta_min, "delta_epi": delta_epi, "n_studies": n_studies, "rows": rows}, f, indent=2)
    print(f"\nJSON summary written to {OUT_DIR}/thresh_{delta_min}.json")


if __name__ == "__main__":
    main()
