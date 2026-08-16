#!/usr/bin/env python3
"""Conservatively repair isolated SMPL tracking spikes in place."""
from pathlib import Path
import argparse, shutil
import numpy as np

def repair(a, threshold):
    out=a.copy(); changed=0
    for _ in range(3):
        for i in range(1,len(out)-1):
            left=np.linalg.norm(out[i]-out[i-1]); right=np.linalg.norm(out[i+1]-out[i]); bridge=np.linalg.norm(out[i+1]-out[i-1])
            if left>threshold and right>threshold and bridge < min(left,right)*0.45:
                out[i]=(out[i-1]+out[i+1])/2; changed+=1
    return out,changed

def main():
 p=argparse.ArgumentParser();p.add_argument('path',type=Path);a=p.parse_args(); path=a.path
 with np.load(path) as d: data={k:d[k] for k in d.files}
 t=data['transl']; before=float(np.linalg.norm(np.diff(t,axis=0),axis=1).max())
 median=float(np.median(np.linalg.norm(np.diff(t,axis=0),axis=1))); threshold=max(1.0,median*8)
 new_t,n=repair(t,threshold)
 if n==0: print(f'[unchanged] {path.parent}'); return
 after=float(np.linalg.norm(np.diff(new_t,axis=0),axis=1).max())
 if after >= before: print(f'[unchanged] {path.parent}'); return
 backup=path.with_name('smpl_params.before_stabilization.npz')
 if not backup.exists(): shutil.copy2(path,backup)
 data['transl']=new_t.astype(np.float32)
 np.savez_compressed(path,**data)
 print(f'[improved] {path.parent} spikes={n} max_jump={before:.3f}->{after:.3f}m')
if __name__=='__main__': main()
