# -*- coding: utf-8 -*-
"""
Architecture characterization under the conjunctive return-compatible criterion.

Reports, per architecture, the positive rate of each component and of the
conjunction. 200 replicates, 1200 trials, 200 permutations for the segment
estimands and 100 for each episodic link.

Usage:  python3 run_chain.py <i0>:<i1> [R]
"""
import numpy as np, json, sys, time
from chain_return import (gen_chain, estimand_A1_to_B, estimand_B_to_A2,
                          chain_episodic, ARCHS_CHAIN, wilson)

R_DEF, NP_EST, NP_EPI, N = 200, 200, 100, 1200

if __name__ == "__main__":
    i0, i1 = [int(v) for v in sys.argv[1].split(":")]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else R_DEF
    t0 = time.time()
    out = {}
    for a in ARCHS_CHAIN[i0:i1]:
        c1 = c2 = ce = cc = 0
        for i in range(R):
            A1, B, A2, Xc, Xk, ret = gen_chain(a, n=N, seed=10_000 + i)
            _, p1 = estimand_A1_to_B(A1, B, Xc, Xk, nperm=NP_EST, seed=20_000 + i)
            _, p2 = estimand_B_to_A2(A1, B, A2, Xc, Xk, nperm=NP_EST, seed=21_000 + i)
            _, pe1, _, pe2, both = chain_episodic(A1, B, A2, Xc, Xk,
                                                  nperm=NP_EPI, seed=30_000 + i)
            s1, s2 = p1 < 0.05, p2 < 0.05
            c1 += s1
            c2 += s2
            ce += both
            cc += (s1 and s2 and both)
        out[a] = dict(ret=bool(ret), R=R, seg1=int(c1), seg2=int(c2),
                      chain=int(ce), conj=int(cc))
        w = lambda k: wilson(k, R)
        print(f"{a:28s} ret={str(ret):5s} | A1->B {100*c1/R:5.1f} [{w(c1)[0]:.1f},{w(c1)[1]:.1f}]"
              f" | B->A2 {100*c2/R:5.1f} [{w(c2)[0]:.1f},{w(c2)[1]:.1f}]"
              f" | chain {100*ce/R:5.1f} [{w(ce)[0]:.1f},{w(ce)[1]:.1f}]"
              f" | CONJ {100*cc/R:5.1f} [{w(cc)[0]:.1f},{w(cc)[1]:.1f}]", flush=True)
        json.dump(out, open(f"chain_{i0}-{i1}.json", "w"))
    print(f"[{time.time()-t0:.0f}s]", flush=True)
