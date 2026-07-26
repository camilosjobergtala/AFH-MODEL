import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
# ---------------- Figure 4: four-outcome heat map
rows=["absent return,\nadequate power","absent return,\nlow power (6 subj)","effect within\nequivalence zone",
      "effect within zone,\nhigh power (40x400)","effect near\nthe boundary","effect clearly\nabove $\\delta_{min}$",
      "return present","high but quantified\nmeasurement error","unreliable B\nrecording",
      "misspecified\ncovariates","latent\nconfounding","failed positive\ncontrol"]
M=np.array([[100.0,0,0,0],[72.5,0,27.5,0],[95.0,0,5.0,0],[100.0,0,0,0],
            [4.0,9.5,86.5,0],[0,100.0,0,0],[0,100.0,0,0],[2.5,14.0,83.5,0],
            [0,0,0,100.0],[0,100.0,0,0],[0,100.0,0,0],[2.0,0,0,98.0]])
cols=["contradicted","survived","inconclusive","not evaluable"]
fig,ax=plt.subplots(figsize=(9.2,9.6))
im=ax.imshow(M,cmap="Blues",vmin=0,vmax=100,aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(cols,fontsize=12)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=10.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]
        ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=11,
                color="white" if v>55 else "#222",weight="bold" if v>55 else "normal")
ax.axvline(2.5,color="#c00",lw=2.4)
ax.set_title("Four-outcome classification of the conjunction\n(% of 200 simulated studies per scenario)",
             fontsize=13,pad=26)
ax.text(0.85,-0.85,"decision under a valid design",ha="center",fontsize=11,color="#2c6fbb")
ax.text(3.0,-0.85,"validity failure",ha="center",fontsize=11,color="#c00")
fig.colorbar(im,ax=ax,shrink=0.6,label="% of studies")
plt.tight_layout()
plt.savefig("figs/Figure4_four_outcomes.png",dpi=300,bbox_inches="tight"); plt.close()
print("ok")
