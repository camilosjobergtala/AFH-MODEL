"""
═══════════════════════════════════════════════════════════════════════════════
STUDY 3 — CODE AUDITOR CONSTRUCT CHECK ON SYNTHETIC SCRIPTS
═══════════════════════════════════════════════════════════════════════════════

WHAT THIS IS (and is NOT):
  A controlled construct check of the canonical CodeAuditor. We generate synthetic
  Python scripts — compliant (clean development code) and violation (clean code with
  exactly ONE designed violation injected) — label them with ground truth, run the
  REAL auditor on each, and report detection by category plus false positives on
  clean scripts and processing time.

  CIRCULAR BY CONSTRUCTION: the violation scripts are built to contain the exact
  patterns the auditor was designed to detect. High detection therefore confirms the
  auditor behaves as designed; it is NOT evidence of generalization to independently
  authored real-world code. The informative quantities are (a) whether each designed
  category is detected, and (b) whether clean development code is left unflagged
  (specificity). This matches §3.3's stated framing.

DESIGN:
  - 100 compliant scripts: k-fold CV development code that never references holdout,
    sets the threshold once, peeks at nothing, runs no uncorrected multiple tests.
  - 100 violation scripts: a clean base with ONE injected violation from a bank of
    seven auditor-detectable categories (roughly balanced).
  - Ground truth = the injected category (compliant = none).

METRICS:
  - Script level (positive = "violation script"): recall, specificity, precision, F1.
  - Per-category recall: of scripts injecting category C, fraction where the auditor
    reported C.
  - False positives on compliant scripts (which categories, if any).
  - Processing time per script.

All generation is deterministic (MASTER_SEED). Outputs in ./outputs/study3/ at the
project root.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import random
import tempfile
import shutil
from pathlib import Path

# --- locate the wrapper / engine (engine/ next to code/) ---
HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "engine", HERE / "engine"):
    if (cand / "auditor_wrapper.py").exists():
        sys.path.insert(0, str(cand))
        break
from auditor_wrapper import audit_script   # noqa: E402

MASTER_SEED = 20250805
N_COMPLIANT = 100
N_VIOLATION = 100
OUT_DIR = HERE.parent / "outputs" / "study3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(MASTER_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Clean (compliant) base: k-fold CV development, threshold set once, no holdout/
# test references, no peeking, no uncorrected multiple testing.
# ─────────────────────────────────────────────────────────────────────────────
COMPLIANT_TEMPLATE = '''\
"""Compliant ECLIPSE development script #{idx} ({model})."""
import numpy as np
from sklearn.model_selection import KFold

X_dev = load_development_features()
y_dev = load_development_labels()

kf = KFold(n_splits={k}, shuffle=True, random_state={seed})
fold_f1 = []
for tr_idx, vl_idx in kf.split(X_dev):
    X_tr, X_vl = X_dev[tr_idx], X_dev[vl_idx]
    y_tr, y_vl = y_dev[tr_idx], y_dev[vl_idx]
    clf = fit_{model}(X_tr, y_tr)
    fold_f1.append(score_f1(clf, X_vl, y_vl))

mean_f1 = float(np.mean(fold_f1))
chosen_threshold = pick_threshold(X_dev, y_dev)   # set once on development data
final_model = fit_{model}(X_dev, y_dev)
'''

MODELS = ["logreg", "svm", "rf", "gbm", "mlp"]


def make_compliant(idx: int) -> str:
    return COMPLIANT_TEMPLATE.format(
        idx=idx, model=rng.choice(MODELS),
        k=rng.choice([3, 5, 10]), seed=rng.randint(1, 9999))


# ─────────────────────────────────────────────────────────────────────────────
# Violation snippets: each injects ONE designed, auditor-detectable category.
# (Some incidentally co-trigger holdout_access because they reference a holdout
#  identifier by name; that is a real detection, not a false positive.)
# ─────────────────────────────────────────────────────────────────────────────
VIOLATION_SNIPPETS = {
    "holdout_access": [
        "leaked = X_test",
    ],
    "conditional_holdout_access": [
        "if score > 0.8:",
        "    kept = holdout_set",
    ],
    "threshold_manipulation": [
        "threshold = 0.50",
        "threshold = 0.65   # nudge",
    ],
    "indirect_holdout_access": [
        "tmp = test_data",
        "feat = tmp",
        "final_model = fit_again(feat)",
    ],
    "result_peeking": [
        "print(test_accuracy)",
    ],
    "multiple_testing": [
        "for col in feature_cols:",
        "    ttest_ind(group_a[col], group_b[col])",
    ],
    "test_tuning": [
        "best = GridSearchCV(estimator, grid).fit(X_test, y_test)",
    ],
}
VIOLATION_CATEGORIES = list(VIOLATION_SNIPPETS.keys())

N_NEARMISS_EACH = 10   # per trap

# Legitimate code that superficially looks suspicious. Truth = NO violation.
# Each trap probes the auditor's specificity (false-positive tendency).
NEARMISS_SNIPPETS = {
    "cv_fold_named_validation_set": [
        "validation_set = X_dev[vl_idx]   # a CV fold, not a held-out test set",
    ],
    "stat_test_object_named_test": [
        "test = ttest_ind(group_a, group_b)   # a statistical test object",
    ],
    "print_development_metric": [
        'print(f"validation accuracy on fold: {val_acc:.3f}")',
    ],
    "default_then_set_threshold": [
        "threshold = 0.5   # provisional default",
        "threshold = pick_threshold(X_dev, y_dev)   # set once on development",
    ],
    "unrelated_print_then_metric_var": [
        'print("starting cross-validation run")',
        "validation_accuracy = float(np.mean(fold_f1))   # never printed",
    ],
}
NEARMISS_TRAPS = list(NEARMISS_SNIPPETS.keys())


def make_nearmiss(idx, trap):
    base = make_compliant(idx).splitlines()
    lines = base[:6] + NEARMISS_SNIPPETS[trap] + base[6:]
    return "\n".join(lines) + "\n", trap


def make_violation(idx: int, category: str):
    """Insert the snippet for `category` into a clean base, at module top level."""
    base = make_compliant(idx).splitlines()
    snippet = VIOLATION_SNIPPETS[category]
    # insert after the imports / data-load block (a fixed, top-level position)
    insert_at = 6
    lines = base[:insert_at] + snippet + base[insert_at:]
    return "\n".join(lines) + "\n", category


def main():
    t0 = time.time()
    print("=" * 72)
    print("STUDY 3 — CODE AUDITOR CONSTRUCT CHECK ON SYNTHETIC SCRIPTS")
    print(f"MASTER_SEED={MASTER_SEED}  compliant={N_COMPLIANT}  violation={N_VIOLATION}")
    print("=" * 72)

    work = Path(tempfile.mkdtemp(prefix="study3_scripts_"))
    records = []   # (path, truth_category_or_None)

    try:
        # generate compliant scripts
        for i in range(N_COMPLIANT):
            p = work / f"compliant_{i:03d}.py"
            p.write_text(make_compliant(i), encoding="utf-8")
            records.append((p, None))

        # generate violation scripts, balanced across categories
        for i in range(N_VIOLATION):
            cat = VIOLATION_CATEGORIES[i % len(VIOLATION_CATEGORIES)]
            code, truth = make_violation(i, cat)
            p = work / f"violation_{i:03d}_{cat}.py"
            p.write_text(code, encoding="utf-8")
            records.append((p, truth))

        # generate near-miss compliant scripts (legit, surface-suspicious)
        nearmiss_truth = {}
        for i in range(len(NEARMISS_TRAPS) * N_NEARMISS_EACH):
            trap = NEARMISS_TRAPS[i % len(NEARMISS_TRAPS)]
            code, _ = make_nearmiss(i, trap)
            p = work / f"nearmiss_{i:03d}_{trap}.py"
            p.write_text(code, encoding="utf-8")
            records.append((p, None))            # truth = no violation
            nearmiss_truth[p.name] = trap

        # audit every script (timed)
        per_script = []
        audit_t0 = time.time()
        for path, truth in records:
            r = audit_script(str(path))
            per_script.append({
                "file": path.name, "truth": truth,
                "is_nearmiss": path.name.startswith("nearmiss_"),
                "trap": nearmiss_truth.get(path.name),
                "flagged": r["n_violations"] > 0,
                "reported": r["categories"],
                "adherence": r["adherence_score"],
            })
        audit_secs = time.time() - audit_t0
        n_total = len(per_script)
        ms_per_script = audit_secs / n_total * 1000

        # ── script-level confusion matrix (positive = violation script) ──
        tp = sum(1 for s in per_script if s["truth"] is not None and s["flagged"])
        fn = sum(1 for s in per_script if s["truth"] is not None and not s["flagged"])
        plain = [s for s in per_script if s["truth"] is None and not s["is_nearmiss"]]
        near = [s for s in per_script if s["is_nearmiss"]]
        tn = sum(1 for s in plain if not s["flagged"])
        fp = sum(1 for s in plain if s["flagged"])
        near_fp = sum(1 for s in near if s["flagged"])
        near_spec = 1 - near_fp / len(near) if near else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else float("nan"))

        print("\n[A] Script-level detection (positive class = violation script)")
        print(f"    TP={tp}  FN={fn}  TN={tn}  FP={fp}")
        print(f"    Recall (sensitivity) : {recall*100:5.1f}%")
        print(f"    Specificity          : {specificity*100:5.1f}%")
        print(f"    Precision            : {precision*100:5.1f}%")
        print(f"    F1                   : {f1:.3f}")

        # ── per-category recall ──
        print("\n[B] Per-category recall (designed category detected when injected)")
        cat_rows = [("category", "n_injected", "n_detected", "recall")]
        per_cat = {}
        for cat in VIOLATION_CATEGORIES:
            injected = [s for s in per_script if s["truth"] == cat]
            detected = [s for s in injected if cat in s["reported"]]
            rec = len(detected) / len(injected) if injected else float("nan")
            per_cat[cat] = {"n_injected": len(injected),
                            "n_detected": len(detected), "recall": round(rec, 4)}
            cat_rows.append((cat, len(injected), len(detected), round(rec, 4)))
            print(f"    {cat:28s} {len(detected):3d}/{len(injected):<3d}  {rec*100:5.1f}%")

        # ── false positives on compliant scripts ──
        fp_scripts = [s for s in per_script if s["truth"] is None and not s["is_nearmiss"] and s["flagged"]]
        fp_cats = {}
        for s in fp_scripts:
            for c in s["reported"]:
                fp_cats[c] = fp_cats.get(c, 0) + 1
        print("\n[C] False positives on compliant scripts")
        if not fp_scripts:
            print(f"    None — all {N_COMPLIANT} compliant scripts passed cleanly.")
        else:
            print(f"    {len(fp_scripts)}/{N_COMPLIANT} compliant scripts flagged; "
                  f"categories: {fp_cats}")

        # ── specificity on near-miss (legit but surface-suspicious) scripts ──
        print("\n[C2] Near-miss specificity (legitimate code that looks suspicious)")
        print(f"    {near_fp}/{len(near)} near-miss scripts wrongly flagged "
              f"(specificity {near_spec*100:.1f}%)")
        trap_fp = {}
        for s in near:
            if s["flagged"]:
                rec = trap_fp.setdefault(s["trap"], {"n": 0, "cats": {}})
                rec["n"] += 1
                for c in s["reported"]:
                    rec["cats"][c] = rec["cats"].get(c, 0) + 1
        for trap in NEARMISS_TRAPS:
            r = trap_fp.get(trap)
            n = r["n"] if r else 0
            cats = r["cats"] if r else {}
            tag = "FALSE POSITIVE" if n else "ok (not flagged)"
            print(f"    {trap:34s} {n:2d}/{N_NEARMISS_EACH}  {tag}  {cats if cats else ''}")

        print(f"\n[D] Processing time: {ms_per_script:.2f} ms/script "
              f"({n_total} scripts in {audit_secs:.2f}s)")

        # ── write outputs ──
        def write_csv(path, rows):
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(",".join(str(x) for x in r) + "\n")

        write_csv(OUT_DIR / "study3_per_category_recall.csv", cat_rows)
        write_csv(OUT_DIR / "study3_confusion.csv",
                  [("tp", "fn", "tn", "fp"), (tp, fn, tn, fp)])
        summary = {
            "config": {"master_seed": MASTER_SEED, "n_compliant": N_COMPLIANT,
                       "n_violation": N_VIOLATION, "pass_threshold": 70.0},
            "script_level": {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
                             "recall": round(recall, 4),
                             "specificity": round(specificity, 4),
                             "precision": round(precision, 4),
                             "f1": round(f1, 4)},
            "per_category_recall": per_cat,
            "near_miss": {"n": len(near), "flagged": near_fp,
                          "specificity": round(near_spec, 4),
                          "by_trap": trap_fp},
            "false_positive_categories_on_compliant": fp_cats,
            "ms_per_script": round(ms_per_script, 3),
        }
        (OUT_DIR / "study3_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

        print("\n" + "=" * 72)
        print(f"Done in {time.time()-t0:.1f}s. Outputs in: {OUT_DIR}")
        print("=" * 72)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()