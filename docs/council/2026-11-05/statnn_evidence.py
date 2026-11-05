"""CPU evidence: run from worktree after sourcing scripts/env.sh."""
import json,csv,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wsp.predictor.data import FeatureDataset,read_targets
from wsp.predictor.baselines import _make_model
from wsp.features.pool import TYPES
from wsp.features.stats import DELTA_STAT_NAMES
warnings.filterwarnings('ignore')
ds=FeatureDataset('data/features/qwen3.5-0.8b','data/evals/qwen3.5-0.8b',targets=['mmlu_pro','hellaswag'])
raw=read_targets('data/evals/qwen3.5-0.8b')
names=['mmlu_pro','hellaswag','kl_to_base','wikitext_ppl_log']
y=np.full((len(ds),4),np.nan); y[:,:2]=ds.targets.numpy()[:,[ds.names.index(t) for t in names[:2]]]
for i,r in enumerate(ds.rows):
 ref=raw.reference(r['id'],ds.successors[r['id']] or 'base'); obs=raw[r['id']]
 for j,k in [(2,'kl_to_base'),(3,'wikitext_ppl')]:
  a,b=obs.get(k,np.nan),ref.get(k,np.nan)
  if np.isfinite(a) and np.isfinite(b) and (j!=3 or min(a,b)>0): y[i,j]=np.log(a)-np.log(b) if j==3 else a-b
X=np.stack([ds.entry(r['id'])['stats'].numpy() for r in ds.rows]); D=np.stack([ds.entry(r['id'])['descriptor'].numpy() for r in ds.rows]); N=np.array([ds.metadata[r['id']]['norms']['fro_effective'] for r in ds.rows]); fam=np.array([r['family'] for r in ds.rows])
features=[('global','all',s) for s in DELTA_STAT_NAMES]+[(t,str(b),s) for t in TYPES for b in range(4) for s in DELTA_STAT_NAMES]
assert len(features)==X.shape[1]
rows=[]; lines=['# StatNN evidence — seed 0, trajectory split','', 'CPU refits using `FeatureDataset`, `FeatureDataset.folds("trajectory", 5)` and `_make_model("statnn", 0)`; no parameter selection. StatNN receives raw pooled stats, exactly as `src/wsp/predictor/audit.py`. Descriptor-only is StandardScaler + Ridge(alpha=1); delta_norm is IsotonicRegression(increasing="auto", out_of_bounds="clip") on effective Frobenius norm. R² is pooled out-of-fold R², not a causal variance decomposition. SD uses ddof=1.','', 'Sources: `src/wsp/predictor/{data,baselines,audit,splits}.py`, `src/wsp/features/{pool,stats}.py`, `data/features/qwen3.5-0.8b/manifest.json`, `data/evals/qwen3.5-0.8b`, `data/zoo/qwen3.5-0.8b/manifest.jsonl`. Manifest has version/layout, not names; names below are decoded from pool.py and DELTA_STAT_NAMES. Bucket b is floor(layer*4/24); non-layer tensors use bucket 0.','', f'Matched rate dataset n={len(ds)}; zoo manifest SHA256 `{ds.snapshot_info["sha256"]}`. KL/PPL are explicitly excluded by FeatureDataset rate checks: these columns extend its rows with the same `read_targets` and `TargetRecords.reference` matching, subtracting matched predecessor/base KL and log(PPL). Missing/nonfinite targets are excluded per fold. No separate unbounded neural head is fitted.','', '## Pooled out-of-fold performance','', '| target | n | model | R² | Spearman |','|---|---:|---|---:|---:|']
folds=ds.folds('trajectory',5); models={}; scores={}; tops={}; within=[]; allpred={}
for j,t in enumerate(names):
 pred={k:np.full(len(ds),np.nan) for k in ['statnn','descriptor_only','delta_norm']}; gains=np.zeros(X.shape[1]); fitted=[]
 for tr,te in folds:
  tr=tr[np.isfinite(y[tr,j])]
  if len(tr)<2: continue
  est=_make_model('statnn',0).fit(X[tr],y[tr,j]); pred['statnn'][te]=est.predict(X[te]); gains+=est.booster_.feature_importance(importance_type='gain'); fitted.append((est,tr,te))
  pred['descriptor_only'][te]=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(D[tr],y[tr,j]).predict(D[te])
  pred['delta_norm'][te]=IsotonicRegression(increasing='auto',out_of_bounds='clip').fit(N[tr],y[tr,j]).predict(N[te])
 for k,p in pred.items():
  ok=np.isfinite(y[:,j])&np.isfinite(p); n=int(ok.sum()); rho=float(spearmanr(y[ok,j],p[ok]).statistic) if n>1 else np.nan; r2=r2_score(y[ok,j],p[ok]) if n>1 else np.nan
  scores[t,k]=(n,rho,r2); rows.append(dict(section='performance',target=t,n=n,model=k,spearman=rho,r2=r2)); lines.append(f'| {t} | {n} | {k} | {r2:.6f} | {rho:.6f} |')
 allpred[t]=pred; models[t]=fitted; tops[t]=np.argsort(-gains)[:20]; total=gains.sum(); n=int(np.isfinite(y[:,j]).sum())
 for rank,ix in enumerate(tops[t],1):
  role,bucket,stat=features[ix]; rows.append(dict(section='importance',target=t,n=n,rank=rank,feature=f'{role}.bucket{bucket}.{stat}',role=role,bucket=bucket,statistic=stat,gain=gains[ix],share=gains[ix]/total if total else np.nan))
 print(t,scores[t,'statnn'],flush=True)
 if t=='mmlu_pro': saved_pred=pred['statnn']
# Verify exact current suite labels/folds/predictions against archived seed-0 output.
archive=pd.concat([pd.read_parquet(f'data/runs/p4/d3v2-m1-trajectory/seed-0/fold-{f}.parquet') for f in range(5)])
lines+=['','## Suite reproduction','']
for j,t in enumerate(names[:2]):
 a=archive[archive.target==t].set_index('id'); ids=[r['id'] for r in ds.rows]; same=set(ids)==set(a.index); assert same; aa=a.loc[ids]; assert np.allclose(y[:,j],aa.observed,atol=1e-8); assert np.allclose(allpred[t]['statnn'],aa.statnn,atol=1e-8); rho=spearmanr(a.observed,a.statnn).statistic
 lines.append(f'{t}: current n={len(ids)}, archived n={len(a)}, same row IDs={same}; archived seed-0 StatNN Spearman={rho:.6f}, refit={scores[t,"statnn"][1]:.6f}. Brief ≈0.57/0.65 refers to three-seed means, not exact seed-0 values.')
lines+=['','## Gain importances','', 'Sum raw split gains over outer-fold fits, then normalize once per target. Correlated coordinates can share/substitute gain; these are fitted-model allocations. All top-20 entries follow (role, depth bucket, statistic).']
for t in names:
 rr=[r for r in rows if r['section']=='importance' and r['target']==t]; n=scores[t,'statnn'][0]
 lines += ['',f'### {t}, n={n}', '',f'Top 5 share={sum(r["share"] for r in rr[:5]):.6f}; top 20 share={sum(r["share"] for r in rr):.6f}.','', '| rank | feature | role | bucket | statistic | gain | share |','|---:|---|---|---|---|---:|---:|']
 for r in rr: lines.append(f'| {r["rank"]} | {r["feature"]} | {r["role"]} | {r["bucket"]} | {r["statistic"]} | {r["gain"]:.6g} | {r["share"]:.6f} |')
lines+=['','## Family dynamic range','', '| target | family | n | mean Δy | SD Δy | min | max |','|---|---|---:|---:|---:|---:|---:|']
for j,t in enumerate(names[:2]):
 for f in sorted(set(fam)):
  v=y[(fam==f)&np.isfinite(y[:,j]),j]; n=len(v)
  if not n:continue
  r=dict(section='family_distribution',target=t,family=f,n=n,mean=v.mean(),sd=v.std(ddof=1),minimum=v.min(),maximum=v.max());rows.append(r);lines.append(f'| {t} | {f} | {n} | {r["mean"]:.6f} | {r["sd"]:.6f} | {r["minimum"]:.6f} | {r["maximum"]:.6f} |')
lines+=['','## Within-family refits','', 'Same helper and five trajectory-grouped folds on each family subset; family membership is constant. NaN denotes undefined Spearman when observed or predicted values are constant. Families with fewer than 30 matched rows are excluded.','', '| target | family | n | Spearman |','|---|---|---:|---:|']
for f in sorted(set(fam)):
 idx=np.where(fam==f)[0]
 if len(idx)<30:continue
 ff=ds.folds('trajectory',5,indices=idx)
 for j,t in enumerate(names):
  p=np.full(len(ds),np.nan)
  for tr,te in ff:
   tr=tr[np.isfinite(y[tr,j])]
   if len(tr)<2:continue
   p[te]=_make_model('statnn',0).fit(X[tr],y[tr,j]).predict(X[te])
  ok=np.isfinite(p)&np.isfinite(y[:,j]); n=int(ok.sum()); rho=spearmanr(y[ok,j],p[ok]).statistic if n>1 else np.nan
  within.append((t,f,n,rho));rows.append(dict(section='within_family',target=t,family=f,n=n,spearman=rho));lines.append(f'| {t} | {f} | {n} | {rho:.6f} |')
lines+=['','## mmlu_pro partial dependence','', 'For each fold model, replace one coordinate in its held-out rows with each pooled empirical decile, average predictions over those rows, then pool by held-out row count. This is marginal model behavior under coordinate replacement, including combinations outside joint feature support; it does not establish a causal effect. Repeated quantiles are collapsed.','', '| target | n | feature | grid value | mean predicted Δy |','|---|---:|---|---:|---:|']
for ix in tops['mmlu_pro'][:3]:
 grid=np.unique(np.quantile(X[:,ix],np.linspace(0,1,11))); vals=[]; feature='.'.join(features[ix])
 for g in grid:
  ps=[]
  for est,tr,te in models['mmlu_pro']:
   xx=X[te].copy();xx[:,ix]=g;ps.extend(est.predict(xx))
  v=float(np.mean(ps));vals.append(v);rows.append(dict(section='partial_dependence',target='mmlu_pro',n=len(ps),feature=feature,grid=g,prediction=v));lines.append(f'| mmlu_pro | {len(ps)} | {feature} | {g:.8g} | {v:.8g} |')
 diff=np.diff(vals); shape='nondecreasing' if np.all(diff>=-1e-10) else 'nonincreasing' if np.all(diff<=1e-10) else 'nonmonotone'
 k=int(np.argmax(np.abs(diff))) if len(diff) else 0
 lines.append(f'\n{feature}, target=mmlu_pro, n={len(ds)}: {shape} on sampled grid; largest adjacent change={diff[k]:.6g} between {grid[k]:.6g} and {grid[k+1]:.6g}. Grid alone does not locate a unique threshold.\n')
lines+=['## Five-line summary','', 'Branch: `council-evidence`.']
lines.append('Reproduced suite: '+ '; '.join(f'{t} n={scores[t,"statnn"][0]} yes={abs(scores[t,"statnn"][1]-spearmanr(archive[archive.target==t].observed,archive[archive.target==t].statnn).statistic)<1e-6}, ρ={scores[t,"statnn"][1]:.6f}' for t in names[:2])+'.')
lines.append('Top 5: '+'; '.join(t+f' (n={scores[t,"statnn"][0]}): '+', '.join('.'.join(features[i]) for i in tops[t][:5]) for t in names[:2])+'.')
lines.append('Within-family StatNN Spearman range (finite): '+ '; '.join(f'{t}: {min(v[3] for v in within if v[0]==t and np.isfinite(v[3])):.6f} to {max(v[3] for v in within if v[0]==t and np.isfinite(v[3])):.6f} (family n: '+', '.join(f'{f}={n}' for tt,f,n,rho in within if tt==t)+')' for t in names)+'.')
# lines.append('Within-family StatNN Spearman: '+'; '.join(f'{t}: '+', '.join(f'{f} n={n} ρ={rho:.6f}' for tt,f,n,rho in within if tt==t) for t in names)+'.')
lines.append('StatNN − descriptor-only Spearman gap: '+'; '.join(f'{t} n={scores[t,"statnn"][0]}: {scores[t,"statnn"][1]-scores[t,"descriptor_only"][1]:.6f}' for t in names)+'.')
out=Path('docs/council/2026-11-05');(out/'evidence-statnn-importances.md').write_text('\n'.join(lines)+'\n');pd.DataFrame(rows).to_csv(out/'evidence-statnn-importances.csv',index=False)
