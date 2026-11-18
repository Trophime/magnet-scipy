import sys, os, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
from multiprocessing import Pool
from detect import analyse, runs
warnings.filterwarnings("ignore")
D=os.path.expanduser("~/github/python_magnetrun/srvdata")
def one(f):
    try: df=pd.read_csv(f,sep=r"\s+",skiprows=1)
    except Exception: return []
    out=[]
    for g in "12":
        cols=[f"Flow{g}",f"Rpm{g}",f"HP{g}","BP"]
        if not set(cols)<=set(df.columns): continue
        d=df[cols].apply(pd.to_numeric,errors="coerce").ffill().fillna(0)
        try: r=analyse(d,g)
        except Exception as e: continue
        on=r["on"].sum()
        if on<100: continue
        if "sensor" not in r:
            out.append(dict(file=os.path.basename(f),grp=g,on=on,kind="dead" if r["zero"].sum()>0.95*on else "fewgood",n=int(r["zero"].sum()),len=int(r["zero"].sum()),hp_dp=float(np.median((d[f"HP{g}"]-d.BP)[r["on"]]))))
            continue
        for k in ("sensor","hydraul"):
            for a,b in runs(r[k]):
                if b-a<3: continue
                seg=d[f"Flow{g}"].values[a:b]
                out.append(dict(file=os.path.basename(f),grp=g,on=on,kind=k,start=int(a),len=int(b-a),
                    zero=bool((seg<=0.5).all()),flow_med=float(np.median(seg)),pred=float(np.median(r["f_rpm"][a:b])),
                    flow_std=float(seg.std()),tau=r["tau"],mad=r["s"]))
    return out
if __name__=="__main__":
    files=sorted(glob.glob(D+"/*.txt"))
    with Pool(16) as p: res=p.map(one,files,chunksize=20)
    pd.DataFrame([x for xx in res for x in xx]).to_csv(sys.argv[1],index=False)
