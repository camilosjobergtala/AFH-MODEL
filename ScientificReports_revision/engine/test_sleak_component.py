"""
Sanity checks for the S_leak component (EclipseIntegrityScore.compute_leakage_score).

    g(m) = o(m) * (mu_dev(m) - mu_hold(m)) / abs(mu_dev(m))
    s(m) = clip(g(m) / g_star, 0, 1)
    S_leak = mean_m s(m)

Verifies, for metrics of BOTH orientations:
  - s(m) == 0.0 when holdout == dev (no expected gap)
  - s(m) == 1.0 when the gap in the expected direction is >= g_star

Not pytest; prints PASS/FAIL like test_eis_wrapper.py.
"""
from eis_wrapper import ProtocolSpec, score_protocol

G_STAR = 0.02


def leakage_score(dev_mean, holdout, orientation, g_star=G_STAR):
    spec = ProtocolSpec(
        dev_metrics={'m': {'mean': dev_mean, 'std': 0.0, 'min': dev_mean, 'values': [dev_mean]}},
        holdout_metrics={'m': holdout},
        metric_orientations={'m': orientation},
    )
    result = score_protocol(spec, g_star=g_star)
    return result['components']['leakage_score']


def check(label, value, expected):
    ok = abs(value - expected) < 1e-9
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: got {value:.6f}, expected {expected:.6f}")
    return ok


print("=" * 70)
print("S_leak COMPONENT VERIFICATION")
print("=" * 70)

all_ok = True

print("\nhigher_is_better metric (dev mean = 0.70):")
all_ok &= check("holdout == dev -> s = 0.0",
                 leakage_score(0.70, 0.70, 'higher_is_better'), 0.0)
all_ok &= check("gap == g_star (holdout = dev * (1 - g_star)) -> s = 1.0",
                 leakage_score(0.70, 0.70 * (1 - G_STAR), 'higher_is_better'), 1.0)
all_ok &= check("gap > g_star -> s = 1.0",
                 leakage_score(0.70, 0.55, 'higher_is_better'), 1.0)
all_ok &= check("holdout better than dev (suspicious) -> s = 0.0",
                 leakage_score(0.70, 0.85, 'higher_is_better'), 0.0)

print("\nlower_is_better metric (dev mean = 0.30, e.g. an error rate):")
all_ok &= check("holdout == dev -> s = 0.0",
                 leakage_score(0.30, 0.30, 'lower_is_better'), 0.0)
all_ok &= check("gap == g_star (holdout = dev * (1 + g_star)) -> s = 1.0",
                 leakage_score(0.30, 0.30 * (1 + G_STAR), 'lower_is_better'), 1.0)
all_ok &= check("gap > g_star -> s = 1.0",
                 leakage_score(0.30, 0.45, 'lower_is_better'), 1.0)
all_ok &= check("holdout better than dev (suspicious) -> s = 0.0",
                 leakage_score(0.30, 0.15, 'lower_is_better'), 0.0)

print("\n" + "=" * 70)
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
print("=" * 70)
