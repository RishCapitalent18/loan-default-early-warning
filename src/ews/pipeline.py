"""
Consumer loan delinquency early-warning pipeline.

Rebuilds the damaged origination export into a clean, typed, model-ready
table, trains a gradient-boosted classifier on a time-ordered split, and
analyses where the risk concentrates. Everything is reproducible from the
committed data/raw_loans.csv.
"""
import json, re
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

RAW="data/raw_loans.csv"

# ----------------------------- repair layer -----------------------------
def money(v):
    if v is None: return np.nan
    s=str(v).strip().replace("$","").replace(",","")
    if s in ("","N/A","n/a"): return np.nan
    try: x=float(s)
    except ValueError: return np.nan
    return np.nan if x<=0 else x        # non-positive income is not usable

def pct(v):
    if v is None: return np.nan
    s=str(v).strip().replace("%","")
    if s in ("","n/a","N/A"): return np.nan
    try: return float(s)
    except ValueError: return np.nan

def emp(v):
    if v is None: return np.nan
    s=str(v).strip().lower()
    if s in ("","n/a","none"): return np.nan
    if s.startswith("< 1"): return 0.0
    if s.startswith("10+"): return 10.0
    m=re.search(r"\d+",s)
    return float(m.group()) if m else np.nan

def fico(v):
    x=pd.to_numeric(v,errors="coerce")
    return np.nan if (pd.isna(x) or x>850 or x<300) else x   # 9999 sentinel dropped

def repair(df):
    out=pd.DataFrame()
    out["loan_id"]=df["loan_id"]
    out["issue_month"]=df["issue_month"].astype(int)
    out["fico"]=df["fico"].map(fico)
    out["annual_inc"]=df["annual_inc"].map(money)
    out["dti"]=pd.to_numeric(df["dti"],errors="coerce")
    out["revol_util"]=df["revol_util"].map(pct)
    out["emp_length_yrs"]=df["emp_length_yrs"].map(emp)
    out["loan_amnt"]=pd.to_numeric(df["loan_amnt"],errors="coerce")
    out["term"]=df["term"].str.extract(r"(\d+)").astype(int)
    out["int_rate"]=df["int_rate"].map(pct)
    out["home_ownership"]=df["home_ownership"].str.upper()
    out["purpose"]=df["purpose"]
    out["default_60d"]=df["default_60d"].astype(int)
    return out

def main():
    raw=pd.read_csv(RAW)
    n_raw=len(raw)

    # a strict validation load would reject rows with unusable values
    reject_mask=(
        raw["annual_inc"].isin(["0","-1","N/A",""]) |
        (pd.to_numeric(raw["fico"],errors="coerce")>850) |
        raw["dti"].isna() |
        (raw["revol_util"].astype(str).str.strip()=="") |
        raw["emp_length_yrs"].isna() |
        raw["emp_length_yrs"].astype(str).str.strip().isin(["","n/a"])
    )
    rescued=int(reject_mask.sum())

    df=repair(raw)
    # median impute numeric, mode/UNK for categoricals (the rebuild step)
    for c in ["fico","annual_inc","dti","revol_util","emp_length_yrs","int_rate"]:
        df[c]=df[c].fillna(df[c].median())
    df["home_ownership"]=df["home_ownership"].replace({"": "UNK"}).fillna("UNK")

    df["loan_to_income"]=df["loan_amnt"]/df["annual_inc"]
    feats=["fico","annual_inc","dti","revol_util","emp_length_yrs","loan_amnt",
           "term","int_rate","loan_to_income","home_ownership","purpose"]
    X=pd.get_dummies(df[feats],columns=["home_ownership","purpose"])
    y=df["default_60d"].values

    # time-ordered split: earlier vintages train, latest test
    train=df["issue_month"]<=18
    Xtr,Xte=X[train.values],X[~train.values]
    ytr,yte=y[train.values],y[~train.values]

    clf=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,
        max_iter=350,l2_regularization=1.0,random_state=42,
        validation_fraction=0.15,early_stopping=True)
    clf.fit(Xtr,ytr)
    prob=clf.predict_proba(Xte)[:,1]

    auc=roc_auc_score(yte,prob)
    ap=average_precision_score(yte,prob)
    base=yte.mean()

    # rank test loans by risk; capture within top deciles
    order=np.argsort(-prob)
    yte_sorted=yte[order]
    n=len(yte); tp=yte.sum()
    def capture(frac): 
        k=int(n*frac); return yte_sorted[:k].sum()/tp
    cap10=capture(0.10); cap20=capture(0.20)

    # operating point: flag top 20% for review
    k=int(n*0.20); thr=np.sort(prob)[::-1][k-1]
    flag=(prob>=thr).astype(int)
    prec=precision_score(yte,flag); rec=recall_score(yte,flag)

    # rules baseline a lender might already run
    te=df[~train.values].reset_index(drop=True)
    rules=((te["fico"]<640)|(te["dti"]>28)|(te["int_rate"]>19)).astype(int).values
    rules_rec=recall_score(yte,rules); rules_prec=precision_score(yte,rules)

    # segment lift table
    seg=(df[~train.values].assign(prob=prob)
         .groupby("purpose")
         .agg(loans=("default_60d","size"),
              actual_default=("default_60d","mean"),
              mean_risk=("prob","mean")).reset_index()
         .sort_values("actual_default",ascending=False))

    report={
      "rows_total":n_raw,
      "records_rescued_by_repair":rescued,
      "records_rescued_pct":round(100*rescued/n_raw,2),
      "test_loans":int(n),
      "test_default_rate":round(float(base),4),
      "roc_auc":round(float(auc),4),
      "pr_auc":round(float(ap),4),
      "capture_top_decile_pct":round(float(cap10)*100,2),
      "capture_top_two_deciles_pct":round(float(cap20)*100,2),
      "review_top20_precision":round(float(prec),4),
      "review_top20_recall":round(float(rec),4),
      "rules_baseline_recall":round(float(rules_rec),4),
      "rules_baseline_precision":round(float(rules_prec),4),
    }
    with open("reports/metrics.json","w") as f: json.dump(report,f,indent=2)
    seg.to_csv("reports/segment_lift.csv",index=False)

    for k_,v in report.items(): print(f"{k_:32}: {v}")
    print("\nsegment lift (by purpose):")
    print(seg.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":
    main()
