"""Gap-filling benchmark on the 24 campaign runs: per-file model vs flow_params prior."""
import sys, os, json, glob, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from detect2 import analyse, reconstruct
EX=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); ROOT=os.path.expanduser("~/LNCMIG-Data/records")
FP={"1":json.load(open(f"{EX}/M9_M23072801_C2_ramps_top_field/M23072801-flow_params.json")),"2":json.load(open(f"{EX}/M9_M23072801_C2_ramps_top_field/M9Bitters_18MW-flow_params.json"))}
rng=np.random.default_rng(0); rows=[]
for p in sorted(glob.glob(f"{EX}/M9_M23072801_C*/provenance.json")):
    for key,r in json.load(open(p))["runs"].items():
        df=pd.read_csv(os.path.join(ROOT,r["pupitre"]),sep=r"\s+",skiprows=1); df=df.loc[:,~df.columns.str.startswith("Unnamed")]
        for g in "12":
            if f"GR{g}" not in r["active_groups"]: continue
            truth=df[f"Flow{g}"].to_numpy(float); rpm=df[f"Rpm{g}"].to_numpy(float)
            run=np.where((rpm>900)&(truth>1))[0]
            dr=np.abs(np.gradient(rpm))
            for L in (5,20,90,300):
                for _ in range(15):
                    if len(run)<L+20: continue
                    a=int(rng.choice(run[:-L])); b=a+L
                    dd=df.copy(); dd.loc[a:b-1,f"Flow{g}"]=0.0
                    res=analyse(dd,g,FP[g],taus=(10,) if g=="1" else (5,))
                    if not res["calibrated"]: continue
                    gap=np.zeros(len(df),bool); gap[a:b]=True
                    row=dict(tag=r["tag"],grp=g,L=L,ramp=dr[a:b].mean()>0.5,detected=res["dropout"][a:b].mean())
                    t=np.arange(len(df))
                    y=truth.copy(); y[gap]=np.interp(t[gap],t[~gap],truth[~gap]); row["interp"]=np.sqrt(np.mean((y[gap]-truth[gap])**2))
                    for name,model in (("file_model",res["f_rpm"]),("prior",res["f_prior"]),("head",res["f_hp"])):
                        row[name+"_raw"]=np.sqrt(np.mean((model[gap]-truth[gap])**2))
                        y=reconstruct(dd[f"Flow{g}"].to_numpy(float),gap,model)
                        row[name+"+offset"]=np.sqrt(np.mean((y[gap]-truth[gap])**2))
                    rows.append(row)
b=pd.DataFrame(rows); b.to_csv("bench_m9.csv",index=False); pd.set_option("display.width",250)
cols=["interp","file_model_raw","file_model+offset","prior_raw","prior+offset","head+offset"]
print("n",len(b)," detected:",b.detected.mean().round(3))
print("median RMS [l/s]"); print(b.groupby(["L","ramp"])[cols].median().round(2))
print("p90 RMS [l/s]"); print(b.groupby(["L","ramp"])[cols].quantile(.9).round(2))
print("by group, all gaps, median:"); print(b.groupby("grp")[cols].median().round(2))
