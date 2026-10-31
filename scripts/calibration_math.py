#!/usr/bin/env python3
"""Numerical helpers for continuous composition calibration."""
import hashlib, json, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye, block_diag

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def jsave(path,record):Path(path).write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def effective_n(weight):
    return float(weight.sum()**2/np.dot(weight,weight)) if np.isfinite(weight).all() and np.dot(weight,weight)>0 else np.nan


def weighted_ks(a,wa,b,wb):
    if not(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(wa).all() and np.isfinite(wb).all()) or min(wa.sum(),wb.sum())<=0:return np.nan
    ia=np.argsort(a,kind='stable');ib=np.argsort(b,kind='stable');a=a[ia];b=b[ib];wa=wa[ia]/wa.sum();wb=wb[ib]/wb.sum();grid=np.unique(np.r_[a,b]);ca=np.r_[0,np.cumsum(wa)];cb=np.r_[0,np.cumsum(wb)];return float(np.max(abs(ca[np.searchsorted(a,grid,side='right')]-cb[np.searchsorted(b,grid,side='right')])))


def program(features,blocks):
    counts=[len(x) for x in features];N=sum(counts);P=features[0].shape[1];offset=np.r_[0,np.cumsum(counts)]
    uniform=np.concatenate([np.full(n,1/n) for n in counts]);prior=np.mean([x.mean(axis=0) for x in features],axis=0)
    objective=np.r_[np.zeros(N+P),np.full(N,.01),np.ones(P)]
    sum_rows=csr_matrix((np.ones(N),(np.repeat(np.arange(4),counts),np.arange(N))),shape=(4,N))
    feature_eq=block_diag([csr_matrix(x.T) for x in features],format='csr');target_eq=vstack([-eye(P,format='csr')]*4,format='csr')
    Aeq=vstack([hstack([sum_rows,csr_matrix((4,P+N+P))]),hstack([feature_eq,target_eq,csr_matrix((4*P,N+P))])],format='csr');beq=np.r_[np.ones(4),np.zeros(4*P)]
    I=eye(N,format='csr');J=eye(P,format='csr');zNP=csr_matrix((N,P));zPN=csr_matrix((P,N))
    Aub=[hstack([I,zNP,-I,zNP]),hstack([-I,zNP,-I,zNP]),hstack([zPN,J,zPN,-J]),hstack([zPN,-J,zPN,-J])];bub=[uniform,-uniform,prior,-prior]
    rr=[];cc=[];nr=0
    for r,block in enumerate(blocks):
        for label in sorted(np.unique(block)):
            local=np.flatnonzero(block==label)+offset[r];rr.extend([nr]*len(local));cc.extend(local.tolist());nr+=1
    mass=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(nr,N));Aub.append(hstack([mass,csr_matrix((nr,P+N+P))]));bub.append(np.full(nr,1/3))
    Aub=vstack(Aub,format='csr');bub=np.concatenate(bub)
    assert Aeq.shape[1]==Aub.shape[1]==len(objective)==2*N+2*P
    bounds=[(0,.05)]*N+[(None,None)]*P+[(0,None)]*(N+P)
    t0=time.monotonic();error=None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:result=linprog(objective,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs',options={'threads':2,'time_limit':120,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        except Exception as exc:result=None;error=type(exc).__name__+': '+str(exc)
    warnings_list=[str(x.message) for x in caught]
    if result is None or result.x is None:
        return dict(weights=[np.full(n,np.nan) for n in counts],target=np.full(P,np.nan),prior=prior,status=int(result.status) if result is not None else -1,success=False,message=str(result.message) if result is not None else error,objective=np.nan,objective_check=np.nan,solver_seconds=time.monotonic()-t0,n_variables=len(objective),n_equalities=len(beq),n_inequalities=len(bub),iterations=int(result.nit) if result is not None and result.nit is not None else 0,warnings=warnings_list,eq_violation=np.nan,ineq_violation=np.nan)
    vector=result.x;w=vector[:N];target=vector[N:N+P];recomputed=float(np.sum(abs(target-prior))+.01*np.sum(abs(w-uniform)))
    assert np.isclose(result.fun,recomputed,atol=1e-6,rtol=1e-7),'L1 objective mismatch'
    return dict(weights=[w[offset[i]:offset[i+1]] for i in range(4)],target=target,prior=prior,status=int(result.status),success=bool(result.success),message=str(result.message),objective=float(result.fun),objective_check=recomputed,solver_seconds=time.monotonic()-t0,n_variables=len(objective),n_equalities=len(beq),n_inequalities=len(bub),iterations=int(result.nit),warnings=warnings_list,eq_violation=float(np.max(abs(Aeq@vector-beq))),ineq_violation=float(np.max(np.maximum(Aub@vector-bub,0))))

