import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

labels = ["recurrent\nreturn","nonlinear\nreturn","confounder +\nweak return","local\npersistence",
          "feedforward\nchain","observed\ncommon cause","parallel\npathways","latent\nconfounder",
          "latent ff\nstate","shared\nlow-rank","coarse\ndependence","token\ndisruption"]
ret = [1,1,1,0,0,0,0,0,0,0,0,0]
seg1 = [100.0,100.0,100.0,100.0,100.0,4.5,4.5,100.0,100.0,100.0,4.5,97.5]
seg2 = [100.0,100.0,100.0,4.0,3.5,3.0,4.5,100.0,100.0,100.0,7.5,99.5]
chain= [100.0,99.5,99.0,2.0,6.5,0.0,0.0,86.5,100.0,49.0,0.0,30.0]
conj = [100.0,99.5,99.0,0.0,0.5,0.0,0.0,86.5,100.0,49.0,0.0,30.0]

def wil(p,n=200,z=1.96):
    k=round(p*n/100); ph=k/n; d=1+z*z/n; c=(ph+z*z/(2*n))/d
    h=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return (100*max(0,c-h),100*min(1,c+h))
def errs(v): 
    lo,hi=zip(*[wil(x) for x in v]); return [np.array(v)-np.array(lo), np.array(hi)-np.array(v)]

x=np.arange(len(labels)); w=0.2
fig,ax=plt.subplots(figsize=(15,5.2))
specs=[(seg1,"#cfe0f5","",r"$\Delta R^2_{A_1\to B}$ (first segment)"),
       (seg2,"#5b9bd5","///",r"$\Delta R^2_{B\to A_2\mid A_1,X}$ (incremental role of B)"),
       (chain,"#9dc3e6","...","chain-level episodic specificity"),
       (conj,"#1f4e79","xx","conjunction: return-compatible")]
for i,(v,c,h,lab) in enumerate(specs):
    ax.bar(x+(i-1.5)*w, v, w, label=lab, color=c, hatch=h, edgecolor="#123",
           linewidth=0.6, yerr=errs(v), capsize=1.8, error_kw=dict(lw=0.8))
ax.axhline(5, ls=":", lw=1.0, color="#c00", zorder=0)
ax.set_ylabel("positive rate (%)", fontsize=12); ax.set_ylim(0,108)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
for i,t in enumerate(ax.get_xticklabels()):
    t.set_color("#1a7f37" if ret[i] else "#a33")
ax.tick_params(axis='y', labelsize=11)
for i in range(len(labels)):
    ax.text(i, -13, "RETURN" if ret[i] else "no return", ha="center", fontsize=8.2,
            color="#1a7f37" if ret[i] else "#a33", weight="bold" if ret[i] else "normal")
ax.legend(fontsize=10, ncol=2, loc="upper center", bbox_to_anchor=(0.5,1.16), frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("figs/Figure3_component_rates.png", dpi=300, bbox_inches="tight")
print("ok")
