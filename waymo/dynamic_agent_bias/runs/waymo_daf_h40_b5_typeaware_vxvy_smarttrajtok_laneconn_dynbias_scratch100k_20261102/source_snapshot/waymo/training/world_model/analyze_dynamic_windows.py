#!/usr/bin/env python3
"""Audit H15 window motion, matching training's focus validity and fallback."""
import argparse, csv, gzip, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from waymo.training.world_model.direct_action_flow import agents_to_bntf, physical_transition_valid, wrap_angle_rad

class AgentsOnly(Dataset):
    def __init__(self, root): self.paths=sorted(Path(root).glob('*.npz'))
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with np.load(self.paths[i]) as z:
            return torch.from_numpy(z['agents']).float(),torch.from_numpy(z['agent_mask']).bool(),i

def window_stats(a,mask,horizon=15,history=11,threshold=.5):
    a=agents_to_bntf(a,mask)
    v=(a[...,5]>.5)&mask[:,:,None]&torch.isfinite(a[...,:2]).all(-1)&torch.isfinite(a[...,6])
    delta=a[:,:,1:,:2]-a[:,:,:-1,:2]
    yaw=wrap_angle_rad(a[:,:,1:,6]-a[:,:,:-1,6])
    valid=v[:,:,1:]&v[:,:,:-1]&physical_transition_valid(delta,yaw,max_displacement_m=5.,max_yaw_delta_rad=.75)
    prefix=valid.unfold(-1,horizon,1)[:,:,history-1:].long().cumprod(-1)
    count=prefix.sum(-1)
    dist=delta.norm(dim=-1).unfold(-1,horizon,1)[:,:,history-1:]
    speed=torch.where(prefix.bool(),dist,0.).sum(-1)/(count.clamp_min(1)*.1)
    legal=count[:,0]==horizon
    fallback=~legal.any(-1)
    selected=legal.clone()
    selected[fallback]=False
    selected[fallback,count[fallback,0].argmax(-1)]=True
    # all: exactly the training eligibility. robust: >= 10 valid transitions.
    eligible=count>0; robust=count>=10
    dynamic=speed>threshold
    n=eligible.sum(1); nr=robust.sum(1)
    ratio=(dynamic&eligible).sum(1)/n.clamp_min(1)
    robust_ratio=(dynamic&robust).sum(1)/nr.clamp_min(1)
    return [x.cpu().numpy() for x in (selected,fallback,n,ratio,nr,robust_ratio,dynamic[:,0])]

def run(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    report={'horizon':15,'history':11,'speed_threshold_mps':.5,'dt':.1,
            'ratio_definition':'dynamic agents / agents with >=1 valid contiguous transition; robust version requires >=10',
            'sampling':'uniform files, uniform complete focus windows; longest prefix fallback if none',
            'splits':{}}
    for split in ('train','val'):
        ds=AgentsOnly(Path(args.data_dir)/split)
        if args.max_files: ds.paths=ds.paths[:args.max_files]
        loader=DataLoader(ds,batch_size=16,num_workers=2,pin_memory=True)
        hist=np.zeros(10,dtype=np.int64); scene_means=[]; ranges=[]; robust_means=[]; fallback_count=0;windows=0
        start=time.time()
        with gzip.open(out/f'{split}_windows.csv.gz','wt') as stream:
            writer=csv.writer(stream);writer.writerow(['dataset_index','filename','anchor','fallback','valid_agents','dynamic_ratio','agents_ge10steps','dynamic_ratio_ge10steps','focus_dynamic'])
            for bi,(a,m,ids) in enumerate(loader):
                result=window_stats(a.to('cuda'),m.to('cuda'))
                sel,fb,n,r,nr,rr,fd=result
                for i,idx in enumerate(ids.tolist()):
                    js=np.flatnonzero(sel[i]);values=r[i,js]
                    fallback_count+=int(fb[i]);windows+=len(js)
                    hist+=np.histogram(values,bins=np.linspace(0,1,11))[0]
                    scene_means.append(float(values.mean()));ranges.append(float(np.ptp(values)))
                    robust_values=rr[i,js][nr[i,js]>0]
                    if len(robust_values):robust_means.append(float(robust_values.mean()))
                    for j in js:
                        writer.writerow([idx,ds.paths[idx].name,int(j)+10,int(fb[i]),int(n[i,j]),float(r[i,j]),int(nr[i,j]),float(rr[i,j]) if nr[i,j] else '',int(fd[i,j])])
                if bi%50==0:
                    print(f'{split}: {min((bi+1)*16,len(ds))}/{len(ds)} files, {windows} windows, {time.time()-start:.1f}s',flush=True)
                    stream.flush()
        report['splits'][split]={'files':len(ds),'selected_windows':windows,'fallback_files':fallback_count,
            'window_dynamic_ratio_histogram_edges':np.linspace(0,1,11).tolist(),'window_histogram_counts':hist.tolist(),
            'scene_uniform_mean_dynamic_ratio':float(np.mean(scene_means)),
            'scene_mean_dynamic_ratio_quantiles':np.quantile(scene_means,[0,.1,.25,.5,.75,.9,1]).tolist(),
            'within_scene_ratio_range_quantiles':np.quantile(ranges,[0,.1,.25,.5,.75,.9,1]).tolist(),
            'fraction_scenes_ratio_range_lt_005':float(np.mean(np.array(ranges)<.05)),
            'scene_uniform_mean_robust_dynamic_ratio':float(np.mean(robust_means)) if robust_means else None}
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['splits'][split]),flush=True)
    print('DONE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data_dir',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--max_files',type=int,default=0)
    with torch.inference_mode():run(p.parse_args())
