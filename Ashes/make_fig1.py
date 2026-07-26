import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BE,BF="#2c6fbb","#eaf1fa"; GE,GF="#1a7f37","#e6f4ea"
RE_,RF="#b22222","#fdecec"; NE,NF="#6b6b6b","#efefef"
fig,ax=plt.subplots(figsize=(13.5,9.6))
def box(x,y,w,h,txt,ec,fc,fs=9.2,bold=False,dash=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",
        lw=1.7,ec=ec,fc=fc,zorder=3,linestyle="--" if dash else "-"))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,zorder=4,
            weight="bold" if bold else "normal")
def arr(p,q,c="#444",lw=1.6,dash=False,ms=13):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=c,
        shrinkA=0,shrinkB=0,zorder=2,linestyle=(0,(3.5,2.5)) if dash else "solid"))

gates=["domain\nmembership","presence\nevidence ($\\varphi$)","observation\nmodel","sufficient\nreliability",
       "sufficient\nsensitivity","positive control\npassed","minimal estimability\n& design adequacy"]
gw,gg=1.58,0.20
for i,g in enumerate(gates):
    x=i*(gw+gg); box(x,8.50,gw,0.86,g,BE,BF,fs=7.6)
    if i<6: arr((x+gw,8.93),(x+gw+gg,8.93),lw=1.0,ms=9)
    arr((x+gw/2,8.50),(x+gw/2,7.92),RE_,1.2,dash=True,ms=10)
R=6*(gw+gg)+gw
ax.text(R/2,9.62,"1.  Seven hierarchical validity gates, applied in fixed order",
        ha="center",fontsize=11.6,weight="bold")
box(0,7.26,R,0.64,"any gate fails   \u2192   NOT EVALUABLE",RE_,RF,fs=11.5,bold=True)

ax.text(R/2,6.82,"2.  Component tests, evaluated only if every gate passes",
        ha="center",fontsize=11.6,weight="bold")
cw=2.55; cg=(R-4*cw)/3
comps=[("Component 1\n$\\Delta R^2_{A_1\\to B}$\n\npredictive gain\nfrom source to\nintermediate",BE,BF,False),
       ("Component 2\n$\\Delta R^2_{B\\to A_2\\mid A_1,X}$\n\nincremental gain\nfrom intermediate\nto late source",BE,BF,False),
       ("Component 3\n$\\theta_{chain}$\n\nchain-level\nepisode\nspecificity",BE,BF,False),
       ("Component 4\nsame source\npopulation\n\ndesign requirement,\nnot an estimated\nstatistic",NE,NF,True)]
ctr=[]
for i,(t,ec,fc,dash) in enumerate(comps):
    x=i*(cw+cg); ctr.append(x+cw/2); box(x,4.62,cw,1.94,t,ec,fc,fs=8.4,dash=dash)
ax.text(ctr[3],4.50,"preregistered, frozen before analysis",ha="center",
        fontsize=7.4,style="italic",color=NE)

# conjunction box spans widely; four arrows converge on distinct points of its top edge
CX0,CX1,CY,CH = 0.13*R, 0.87*R, 3.16, 0.74
span=CX1-CX0
entry=[CX0+span*f for f in (0.16,0.39,0.61,0.84)]
for i,(cx,ex) in enumerate(zip(ctr,entry)):
    col = NE if i==3 else "#444"
    arr((cx,4.62),(ex,CY+CH),c=col,lw=1.5,dash=(i==3))
box(CX0,CY,span,CH,"3.  Conjunction-level aggregation   (all four components required)",
    GE,GF,fs=11.0,bold=True)

ow=3.1; og=(R-3*ow)/2
outs=["CONTRADICTED\n\nany component\ncontradicted",
      "SURVIVED TESTING\n\nall components\nsurvived",
      "INCONCLUSIVE\n\nnone contradicted,\nat least one\ninconclusive"]
for i,t in enumerate(outs):
    x=i*(ow+og); box(x,1.06,ow,1.40,t,GE,GF,fs=8.8)
    arr((R/2,CY),(x+ow/2,2.46),c=GE,lw=1.3)
ax.text(R/2,0.58,"Survival confirms neither constitution nor identity; the conjunction is observational "
        "and does not establish causal return.",ha="center",fontsize=9.4,style="italic",color="#333")
ax.set_xlim(-0.3,R+0.3); ax.set_ylim(0.22,9.95); ax.axis("off")
plt.tight_layout()
plt.savefig("figs/Figure1_decision_flow.png",dpi=300,bbox_inches="tight")
plt.savefig("figs/Figure1_decision_flow.pdf",bbox_inches="tight")
print("ok")
