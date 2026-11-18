import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
from multiprocessing import Pool
from detect import runs
warnings.filterwarnings("ignore")
D=os.path.expanduser("~/github/python_magnetrun/srvdata/")
def one(a):
    f,g=a
    df=pd.read_csv(D+f,sep=r"\s+",skiprows=1)
    d=df[[f"Flow{g}",f"Rpm{g}",f"HP{g}","BP"]].apply(pd.to_numeric,errors="coerce").ffill().fillna(0)
    flow=d[f"Flow{g}"].values; dp=(d[f"HP{g}"]-d.BP).values; on=d[f"Rpm{g}"].values>900
    z=on&(flow<=0.5)
    out=[]
    for s,e in runs(z):
        h=np.median(dp[s:e])
        out.append(dict(file=f,grp=g,start=int(s),len=int(e-s),head=float(h),
                        head_before=float(np.median(dp[max(0,s-10):s])) if s>0 else np.nan,
                        flow_before=float(np.median(flow[max(0,s-10):s])) if s>0 else np.nan))
    return out
if __name__=="__main__":
    s=pd.read_csv("scan.csv"); s=s[s.n.notna()&(s.nzero>0)&(s.nzero<0.95*s.n)]
    args=[(r.file,str(int(r.grp))) for r in s.itertuples()]
    with Pool(16) as p: res=p.map(one,args,chunksize=8)
    pd.DataFrame([x for xx in res for x in xx]).to_csv("zero2.csv",index=False)
