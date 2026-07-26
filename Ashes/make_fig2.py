import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

# ---------------- Figure 2: six architectures, token disruption replaces coarse dep.
def node(ax,x,y,lab,grey=False):
    ax.add_patch(Ellipse((x,y),0.46,0.30,fc="#d9d9d9" if grey else "white",
                 ec="black",lw=1.6,zorder=3))
    ax.text(x,y,lab,ha="center",va="center",fontsize=12,zorder=4)
def arrow(ax,p,q,color="black",dashed=False,lw=2.0):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=15,lw=lw,
                 color=color,shrinkA=15,shrinkB=15,zorder=2,
                 linestyle=(0,(4,3)) if dashed else "solid"))

panels=[("(a) genuine return","return to source A"),
        ("(b) local persistence","no return through B"),
        ("(c) feedforward termination","terminates elsewhere"),
        ("(d) observed common cause","observed driver"),
        ("(e) latent common cause","unobserved driver"),
        ("(f) token disruption","coarse links, identity lost")]

fig,axes=plt.subplots(2,3,figsize=(14,7.4))
for ax,(title,sub) in zip(axes.ravel(),panels):
    ax.set_xlim(0,3); ax.set_ylim(0,2.2); ax.axis("off")
    ax.set_title(title,fontsize=13,pad=8)
    ax.text(1.5,0.06,sub,ha="center",fontsize=10.5,color="#555")
    if title.startswith("(a)"):
        node(ax,0.5,0.75,"$A_1$"); node(ax,1.5,1.65,"B"); node(ax,2.5,0.75,"$A_2$")
        arrow(ax,(0.5,0.75),(1.5,1.65),"#c0392b"); arrow(ax,(1.5,1.65),(2.5,0.75),"#c0392b")
    elif title.startswith("(b)"):
        node(ax,0.5,0.75,"$A_1$"); node(ax,1.5,1.65,"B"); node(ax,2.5,0.75,"$A_2$")
        arrow(ax,(0.5,0.75),(1.5,1.65)); arrow(ax,(0.5,0.75),(2.5,0.75))
    elif title.startswith("(c)"):
        node(ax,0.5,0.75,"$A_1$"); node(ax,1.5,1.65,"B"); node(ax,2.5,0.75,"C")
        arrow(ax,(0.5,0.75),(1.5,1.65)); arrow(ax,(1.5,1.65),(2.5,0.75))
    elif title.startswith("(d)"):
        node(ax,1.5,1.65,"X"); node(ax,0.6,0.7,"$A_1$"); node(ax,2.4,0.7,"$A_2$")
        arrow(ax,(1.5,1.65),(0.6,0.7)); arrow(ax,(1.5,1.65),(2.4,0.7))
    elif title.startswith("(e)"):
        node(ax,1.5,1.65,"U",grey=True); node(ax,0.6,0.7,"$A_1$"); node(ax,2.4,0.7,"$A_2$")
        arrow(ax,(1.5,1.65),(0.6,0.7)); arrow(ax,(1.5,1.65),(2.4,0.7))
    else:
        node(ax,0.5,0.75,"$A_1$"); node(ax,1.5,1.65,"B"); node(ax,2.5,0.75,"$A_2$")
        arrow(ax,(0.5,0.75),(1.5,1.65),"#7f7f7f",dashed=True)
        arrow(ax,(1.5,1.65),(2.5,0.75),"#7f7f7f",dashed=True)
fig.text(0.5,0.015,"Only panel (a) contains episode-specific return to the source population. "
  "$A_1$ and $A_2$ denote the same population in an early and a late window; B is the intermediate population.\n"
  "Red marks the return path; grey fill marks an unobserved variable; dashed grey marks a link that "
  "transmits only a coarse summary, so episode identity is not preserved.",
  ha="center",fontsize=10)
plt.tight_layout(rect=[0,0.075,1,1])
plt.savefig("figs/Figure2_architectures.png",dpi=300,bbox_inches="tight"); plt.close()

