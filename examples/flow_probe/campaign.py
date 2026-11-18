"""Run detector v2 on the 24 M9_M23072801 campaign pupitre files."""
import sys, os, json, glob, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from detect2 import analyse, events
EX=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT=os.path.expanduser("~/LNCMIG-Data/records")
FP={"1":json.load(open(f"{EX}/M9_M23072801/M23072801-flow_params.json")) if os.path.exists(f"{EX}/M9_M23072801/M23072801-flow_params.json") else json.load(open(f"{EX}/M9_M23072801_C2_ramps_top_field/M23072801-flow_params.json")),
    "2":json.load(open(f"{EX}/M9_M23072801_C2_ramps_top_field/M9Bitters_18MW-flow_params.json"))}
rows, evs = [], []
for p in sorted(glob.glob(f"{EX}/M9_M23072801_C*/provenance.json")):
    d=json.load(open(p))
    for key,r in d["runs"].items():
        df=pd.read_csv(os.path.join(ROOT,r["pupitre"]),sep=r"\s+",skiprows=1)
        df=df.loc[:,~df.columns.str.startswith("Unnamed")]
        for g in "12":
            res=analyse(df,g,FP[g],taus=(10,) if g=="1" else (5,))
            lr=res.get("rpm_law_res")
            on=res["on"]
            row=dict(group=d["group"],tag=r["tag"],grp=f"GR{g}",active=f"GR{g}" in r["active_groups"],n=len(df),n_on=int(on.sum()),
                     calibrated=res["calibrated"],rpm_law_rms=float(np.sqrt(np.mean(lr[on]**2))) if lr is not None and on.any() else np.nan,
                     rpm_law_max=float(np.abs(lr[on]).max()) if lr is not None and on.any() else np.nan,
                     rpm_law_bad_s=int((np.abs(lr)>20)[on].sum()) if lr is not None else 0)
            if res["calibrated"]:
                row.update(tau=res["tau"],tau_m=res["tau_m"],a=res["a"],b=res["b"],mad=res["s"],thr=res["thr"],n_hp=res["n_hp"],mad_hp=res["sh"],
                           a_prior=res["a_prior"],b_prior=res["b_prior"],bias_prior=res["bias_prior"])
            for kind in ("dropout","stuck","probe","stale","hydraulic","pump_off"):
                row[kind+"_s"]=int(res[kind].sum()) if kind in res else 0
            rows.append(row)
            for kind,a,b in events(res):
                fl=df[f"Flow{g}"].to_numpy(float)[a:b]
                evs.append(dict(group=d["group"],tag=r["tag"],grp=f"GR{g}",kind=kind,start=a,stop=b,len=b-a,
                                flow_med=float(np.median(fl)),pred=float(np.median(res["f_rpm"][a:b])) if "f_rpm" in res else np.nan,
                                head=float(np.median(res["head"][a:b]))))
pd.DataFrame(rows).to_csv("campaign_runs.csv",index=False); pd.DataFrame(evs).to_csv("campaign_events.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
R=pd.DataFrame(rows)
print(R[["group","tag","grp","active","n_on","tau","a","b","a_prior","b_prior","bias_prior","mad","n_hp","mad_hp","rpm_law_rms","rpm_law_max","rpm_law_bad_s"]].round(3).to_string())
print(R[["group","tag","grp"]+[k+"_s" for k in ("dropout","stuck","probe","stale","hydraulic","pump_off")]].to_string())
E=pd.DataFrame(evs); print(E.groupby(["kind"]).agg(n=("len","size"),runs=("tag","nunique"),total=("len","sum"),max=("len","max")))
