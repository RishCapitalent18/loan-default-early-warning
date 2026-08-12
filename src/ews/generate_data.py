"""
Synthesizes a deliberately messy consumer loan book, the kind a lender's
origination systems actually emit: money fields stored as strings with dollar
signs and commas, employment length as free text, rates and utilisation with
percent signs, FICO sentinels (9999) for missing pulls, inconsistent home
ownership casing, and missing values throughout. A latent 60-day delinquency
signal is embedded so a model can learn it after the data is rebuilt.

Committed to data/raw_loans.csv so the pipeline is reproducible. Seed fixed.
"""
import numpy as np, pandas as pd, os

SEED=42; N=40000
rng=np.random.default_rng(SEED)

fico   = np.clip(rng.normal(690,55,N),520,850).round()
log_inc= rng.normal(11.0,0.5,N)
income = np.exp(log_inc)
dti    = np.clip(rng.normal(18,8,N),0,45)
util   = np.clip(rng.normal(52,25,N),0,130)
emp_yrs= np.clip(rng.normal(6,4,N),0,25)
loan   = np.clip(rng.normal(15000,8000,N),1000,40000).round(-2)
term   = rng.choice([36,60],N,p=[0.7,0.3])
purpose= rng.choice(["debt_consolidation","credit_card","home_improvement",
                     "major_purchase","small_business","medical","other"],
                    N,p=[0.42,0.22,0.11,0.08,0.06,0.05,0.06])
# int rate rises as fico falls, plus term premium and noise
z=lambda x:(x-x.mean())/x.std()
int_rate=np.clip(12.5 - 3.2*z(fico) + 1.1*(term==60) + rng.normal(0,1.6,N),5,30)

purpose_risk=pd.Series({"debt_consolidation":0.1,"credit_card":0.15,
    "home_improvement":-0.15,"major_purchase":0.0,"small_business":0.55,
    "medical":0.25,"other":0.1})
logit=(-2.55
       -0.80*z(fico) +0.42*z(dti) +0.50*z(int_rate) +0.32*z(util)
       -0.16*z(emp_yrs) -0.22*(log_inc-log_inc.mean())
       +purpose_risk.reindex(purpose).values
       +rng.normal(0,0.88,N))
p=1/(1+np.exp(-logit))
default=(rng.random(N)<p).astype(int)

df=pd.DataFrame({
    "loan_id":[f"L{100000+i}" for i in range(N)],
    "issue_month":rng.integers(1,25,N),  # months since program start (temporal split)
    "fico":fico.astype(int),
    "annual_inc":income.round(2),
    "dti":dti.round(2),
    "revol_util":util.round(1),
    "emp_length_yrs":emp_yrs.round(1),
    "loan_amnt":loan.astype(int),
    "term":term,
    "int_rate":int_rate.round(2),
    "home_ownership":rng.choice(["RENT","MORTGAGE","OWN"],N,p=[0.4,0.45,0.15]),
    "purpose":purpose,
    "default_60d":default,
})

# ---- now damage it the way real origination exports are damaged ----
def dirty_money(v):
    if rng.random()<0.5: return f"${v:,.2f}"
    if rng.random()<0.3: return f"{v:,.0f}"
    return str(v)
df["annual_inc"]=df["annual_inc"].map(dirty_money)
# income sentinels / bad values
bad_inc=rng.random(N)<0.04
df.loc[bad_inc,"annual_inc"]=rng.choice(["0","-1","N/A",""],bad_inc.sum())

# emp length as free text
def emp_text(y):
    if y<1: return "< 1 year"
    if y>=10: return "10+ years"
    return f"{int(round(y))} years"
df["emp_length_yrs"]=df["emp_length_yrs"].map(emp_text)
emp_missing=rng.random(N)<0.06
df.loc[emp_missing,"emp_length_yrs"]=rng.choice(["n/a","",None],emp_missing.sum())

df["int_rate"]=df["int_rate"].map(lambda v:f"{v}%")
util_missing=rng.random(N)<0.08
df["revol_util"]=df["revol_util"].map(lambda v:f"{v}%")
df.loc[util_missing,"revol_util"]=""
# FICO sentinel for missing pulls
fico_missing=rng.random(N)<0.05
df.loc[fico_missing,"fico"]=9999
df["term"]=df["term"].map(lambda t:f" {t} months")
df["home_ownership"]=df["home_ownership"].map(
    lambda h: rng.choice([h,h.lower(),h.title()]))
# scatter a few fully blank dti
dti_missing=rng.random(N)<0.05
df.loc[dti_missing,"dti"]=np.nan

os.makedirs("data",exist_ok=True)
df.to_csv("data/raw_loans.csv",index=False)
print(f"rows={len(df)} default_rate={df['default_60d'].mean():.4f} "
      f"bad_income={int(bad_inc.sum())} fico_missing={int(fico_missing.sum())}")
