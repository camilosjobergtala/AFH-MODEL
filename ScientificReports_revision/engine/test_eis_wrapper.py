"""Quick verification that the wrapper correctly bridges to the v3.0 EIS engine."""
from eis_wrapper import ProtocolSpec, score_protocol, components_of

print("=" * 70)
print("WRAPPER VERIFICATION")
print("=" * 70)

# 1. A 'strong' protocol: everything present, balanced split, expected degradation
strong = ProtocolSpec(
    n_development=70, n_holdout=30,
    holdout_metrics={'f1_score': 0.66},  # slightly below dev mean 0.70 -> low leakage risk
)
r_strong = score_protocol(strong)
print("\nSTRONG protocol:")
print(f"  EIS = {r_strong['eis']:.4f}  ({r_strong['interpretation']})")
for k, v in components_of(r_strong).items():
    print(f"    {k:18s} {v:.4f}")

# 2. A 'weak' protocol: no hash, no descriptions, tiny imbalanced split, suspicious holdout
weak = ProtocolSpec(
    validation_completed=True,
    has_regdate_and_hash=False,
    criteria_have_thresholds=False,
    criteria_have_descriptions=False,
    n_development=18, n_holdout=4,
    has_integrity_verification=False,
    has_split_date=False,
    has_binding_declaration=False,
    holdout_metrics={'f1_score': 0.92},  # way above dev mean -> high leakage risk
)
r_weak = score_protocol(weak)
print("\nWEAK protocol:")
print(f"  EIS = {r_weak['eis']:.4f}  ({r_weak['interpretation']})")
for k, v in components_of(r_weak).items():
    print(f"    {k:18s} {v:.4f}")

# 3. Weight sensitivity: same protocol, different weights -> different EIS
equal_w = {'preregistration': 0.2, 'protocol_adherence': 0.2, 'split_strength': 0.2,
           'leakage_risk': 0.2, 'transparency': 0.2}
prereg_dom = {'preregistration': 0.6, 'protocol_adherence': 0.1, 'split_strength': 0.1,
              'leakage_risk': 0.1, 'transparency': 0.1}
r_eq = score_protocol(strong, weights=equal_w)
r_pd = score_protocol(strong, weights=prereg_dom)
print("\nWEIGHT SENSITIVITY (strong protocol):")
print(f"  default weights : {r_strong['eis']:.4f}")
print(f"  equal weights   : {r_eq['eis']:.4f}")
print(f"  prereg-dominant : {r_pd['eis']:.4f}")

# 4. Sanity: components are identical regardless of weights (weights only affect aggregate)
c1 = components_of(r_strong)
c2 = components_of(r_eq)
identical = all(abs(c1[k] - c2[k]) < 1e-12 for k in c1)
print(f"\nComponents invariant to weights: {identical}  (expected: True)")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)