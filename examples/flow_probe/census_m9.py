"""Detector v2 over the whole M9 srv-data-install archive."""
import sys, os, glob, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from detect2 import analyse, events
warnings.filterwarnings("ignore")
D=os.path.expanduser("~/LNCMIG-Data/records/srv-data-install/M9")
def one(f):
    try:
        df=pd.read_csv(f,sep=r"\s+",skiprows=1,on_bad_lines="skip")
    except Exception:
        return [], []
    df=df.loc[:,~df.columns.str.startswith("Unnamed")]
    rows, evs = [], []
    for g in "12":
        cols=[f"Flow{g}",f"Rpm{g}",f"HP{g}","BP"]
        if not set(cols)<=set(df.columns): continue
        d=df[cols].apply(pd.to_numeric,errors="coerce").ffill().fillna(0)
        try: r=analyse(d,g)
        except Exception: continue
        on=int(r["on"].sum())
        if on<100: continue
        name=os.path.basename(f)
        row=dict(file=name,grp=g,n=len(d),on=on,calibrated=r["calibrated"],
                 running=int((r["on"]&(r["head"]>0.5)).sum()))
        if r["calibrated"]:
            row.update(tau=r["tau"],tau_m=r["tau_m"],b=r["b"],mad=r["s"])
        for k in ("dropout","stuck","probe","stale","hydraulic","pump_off"):
            row[k+"_s"]=int(r[k].sum()) if k in r else 0
        rows.append(row)
        fl=d[f"Flow{g}"].to_numpy()
        for kind,a,b in events(r):
            evs.append(dict(file=name,grp=g,kind=kind,start=a,len=b-a,flow_med=float(np.median(fl[a:b])),
                            flow_before=float(np.median(fl[max(0,a-10):a])) if a else np.nan,head=float(np.median(r["head"][a:b]))))
    return rows, evs
if __name__=="__main__":
    files=sorted(glob.glob(D+"/*.txt"))
    print(len(files))
    with Pool(16) as p: res=p.map(one,files,chunksize=20)
    pd.DataFrame([x for r,_ in res for x in r]).to_csv("m9_runs.csv",index=False)
    pd.DataFrame([x for _,e in res for x in e]).to_csv("m9_events.csv",index=False)
