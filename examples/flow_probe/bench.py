import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
from multiprocessing import Pool
from detect import analyse, reconstruct, runs
warnings.filterwarnings("ignore")
D=os.path.expanduser("~/github/python_magnetrun/srvdata/")
s=pd.read_csv("scan2.csv")
def one(args):
    f,g,seed=args
    rng=np.random.default_rng(seed)
    df=pd.read_csv(D+f,sep=r"\s+",skiprows=1)
    cols=[f"Flow{g}",f"Rpm{g}",f"HP{g}","BP"]
    d=df[cols].apply(pd.to_numeric,errors="coerce").ffill().fillna(0)
    truth=d[f"Flow{g}"].values.astype(float).copy()
    on=np.where(d[f"Rpm{g}"].values>900)[0]
    rpm=d[f"Rpm{g}"].values; dr=np.abs(np.gradient(rpm))
    out=[]
    for L in (5,20,90,300):
        for trial in range(4):
            if len(on)<L+50: continue
            a=int(rng.choice(on[:-L])); b=a+L
            if not (rpm[a:b]>900).all(): continue
            ramp = dr[a:b].mean()>0.5
            dd=d.copy(); dd.loc[a:b-1,f"Flow{g}"]=0.0
            r=analyse(dd,g)
            if "sensor" not in r: continue
            bad=r["sensor"]
            det=bad[a:b].mean()
            fa=int((bad & ~np.isin(np.arange(len(bad)),np.arange(a,b))).sum())
            row=dict(file=f,grp=g,L=L,ramp=ramp,detected=det,false=fa)
            gap=np.zeros(len(bad),bool); gap[a:b]=True
            for m in ("interp","rpm","rpm_bias","rpm_hp"):
                y=reconstruct(dd,g,gap,r,m)
                row[m]=np.sqrt(np.mean((y[a:b]-truth[a:b])**2))
            out.append(row)
    return out
if __name__=="__main__":
    sub=s.sample(400,random_state=0)
    args=[(r.file,str(int(r.grp)),i) for i,r in enumerate(sub.itertuples())]
    with Pool(16) as p: res=p.map(one,args,chunksize=4)
    b=pd.DataFrame([x for xx in res for x in xx]); b.to_csv("bench.csv",index=False)
    pd.set_option("display.width",200)
    print("n",len(b))
    print(b.groupby(["L","ramp"])[["detected","interp","rpm","rpm_bias","rpm_hp"]].median().round(2))
    print(b.groupby(["L","ramp"])[["interp","rpm","rpm_bias","rpm_hp"]].quantile(0.9).round(2))
    print("false-alarm samples per trial: median",b.false.median()," mean",b.false.mean().round(2)," trials with >0:",(b.false>0).mean().round(3))
