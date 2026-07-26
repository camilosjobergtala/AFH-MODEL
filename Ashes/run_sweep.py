# -*- coding: utf-8 -*-
"""Sample-size sweep of the CONJUNCTION under weak latent confounding with no true
return through B. Reports the rate of false return-compatible attribution by trials."""
import numpy as np, sys, time
from sim_return import DIM, wilson
from chain_return import estimand_A1_to_B, estimand_B_to_A2, chain_episodic

def gen_weak(n, seed, kappa=0.28):
    r=np.random.default_rng(seed)
    Xc=r.integers(0,4,n); Xk=r.standard_normal(n)
    dr=lambda: r.standard_normal((4,DIM))[Xc]+np.outer(Xk,r.standard_normal(DIM))
    U=r.standard_normal((n,3)); L=lambda: r.standard_normal((3,DIM))*kappa
    e=lambda: r.standard_normal((n,DIM))
    A1=dr()+U@L()+e()
    B =dr()+A1@(r.standard_normal((DIM,DIM))*0.7)+U@L()+e()
    A2=dr()+U@L()+e()                      # NO return through B
    return A1,B,A2,Xc,Xk

if __name__=="__main__":
    sizes=[int(v) for v in sys.argv[1].split(",")]; R=int(sys.argv[2]) if len(sys.argv)>2 else 200
    t0=time.time()
    print(f"{'trials':>7} | {'conjunction positive':>24} | {'C1':>6} {'C2':>6} {'C3':>6}")
    for n in sizes:
        c1=c2=c3=cc=0
        for i in range(R):
            A1,B,A2,Xc,Xk = gen_weak(n, 5000+i)
            _,p1 = estimand_A1_to_B(A1,B,Xc,Xk,nperm=100,seed=6000+i)
            _,p2 = estimand_B_to_A2(A1,B,A2,Xc,Xk,nperm=100,seed=6100+i)
            _,pe1,_,pe2,both = chain_episodic(A1,B,A2,Xc,Xk,nperm=50,seed=6200+i)
            s1,s2 = p1<0.05, p2<0.05
            c1+=s1; c2+=s2; c3+=both; cc+=(s1 and s2 and both)
        lo,hi=wilson(cc,R)
        print(f"{n:7d} | {100*cc/R:6.1f}% [95% CI {lo:4.1f},{hi:5.1f}] | "
              f"{100*c1/R:5.1f} {100*c2/R:5.1f} {100*c3/R:5.1f}", flush=True)
    print(f"[{time.time()-t0:.0f}s]", flush=True)
