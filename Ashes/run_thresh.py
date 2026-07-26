import numpy as np, sys, time
import run_outcomes as R
SC=["absent_power","within_equiv","return_present","latent_confounding"]
d=float(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 150
R.DELTA_R2=d; R.DELTA_EPI=2*d; R.PC_MIN=0.5*d
t0=time.time(); print(f"=== delta_min={d} (delta_epi={2*d}) ===")
for scn in SC:
    c={o:0 for o in R.OUT}
    for i in range(N):
        o,_=R.classify(scn,100_000+37*i); c[o]+=1
    print(f"{scn:22s} "+" ".join(f"{o[:5]}={100*c[o]/N:5.1f}" for o in R.OUT),flush=True)
print(f"[{time.time()-t0:.0f}s]",flush=True)
