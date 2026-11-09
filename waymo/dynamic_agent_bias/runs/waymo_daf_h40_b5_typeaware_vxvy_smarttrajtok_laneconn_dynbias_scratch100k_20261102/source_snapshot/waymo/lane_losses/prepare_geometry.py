#!/usr/bin/env python3
"""Road corridor widths from annotated boundaries and measured vehicle widths.

Independent sidecars. No fabricated width at unannotated intersections/lanes.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import os
from pathlib import Path
import struct
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from waymo.adaptations.scenario_reader import scenario_class
from waymo.adaptations.prepare_map_cache import rebase_manifest_rows
VERSION=1


def project_boundary(xy,boundary):
    """Closest projection to raw boundary polyline, vectorized in bounded chunks."""
    if len(boundary)<2:return None
    start=boundary[:-1];delta=np.diff(boundary,axis=0);square=(delta*delta).sum(-1)
    output=[]
    for first in range(0,len(xy),128):
        q=xy[first:first+128]
        frac=((q[:,None]-start[None])*delta[None]).sum(-1)/np.maximum(square[None],1e-12)
        projected=start[None]+np.clip(frac,0,1)[...,None]*delta[None]
        dist=((q[:,None]-projected)**2).sum(-1)
        indices=dist.argmin(-1);output.append(projected[np.arange(len(q)),indices])
    return np.concatenate(output)


def lane_profile(lane,features,count):
    raw=np.asarray([[p.x,p.y] for p in lane.polyline],float).reshape(-1,2)
    if len(raw)<2:return np.zeros((count,2),np.float32),np.zeros(count,bool)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=-1))]
    intervals=np.stack([np.linspace(min(i*5.,arc[-1]),min((i+1)*5.,arc[-1]),6) for i in range(count)])
    flat=intervals.flatten()
    xy=np.stack([np.interp(flat,arc,raw[:,i]) for i in range(2)],-1)
    direction=np.gradient(xy.reshape(count,6,2),axis=1).reshape(-1,2)
    norm=np.linalg.norm(direction,axis=-1,keepdims=True)
    tangent=np.divide(direction,norm,out=np.zeros_like(direction),where=norm>1e-8)
    widths=np.full((len(flat),2),np.nan)
    for side,name in enumerate(['left_boundaries','right_boundaries']):
        for annotation in getattr(lane,name):
            a,b=annotation.lane_start_index,annotation.lane_end_index
            if not (0<=a<=b<len(arc)):raise ValueError('Invalid road boundary range')
            region=(flat>=arc[a]-1e-4)&(flat<=arc[b]+1e-4)&(norm[:,0]>1e-8)
            if not region.any():continue
            feature=features.get(int(annotation.boundary_feature_id))
            if feature is None:continue
            field='road_line' if feature.HasField('road_line') else 'road_edge' if feature.HasField('road_edge') else None
            if field is None:continue
            boundary=np.asarray([[p.x,p.y] for p in getattr(feature,field).polyline],float).reshape(-1,2)
            projection=project_boundary(xy[region],boundary)
            if projection is None:continue
            offset=projection-xy[region];t=tangent[region]
            signed=t[:,0]*offset[:,1]-t[:,1]*offset[:,0]
            width=signed if side==0 else -signed
            forward=np.abs((offset*t).sum(-1))
            good=(width>=.5)&(width<=10.)&(forward<=3.)
            slots=np.flatnonzero(region)[good];values=width[good]
            before=widths[slots,side]
            widths[slots,side]=np.where(np.isfinite(before),np.minimum(before,values),values)
    valid=np.isfinite(widths).all(-1).reshape(count,6).all(-1)
    # Both sides must support all six sampled points of the whole segment.
    width=np.nan_to_num(widths.reshape(count,6,2),nan=0.).mean(1).astype(np.float32)
    width[~valid]=0.
    if lane.type not in (1,2):valid[:]=False;width[:]=0.
    return width,valid


def process_shard(job):
    shard,rows,map_root,out_root=job
    grouped=defaultdict(list)
    for row in rows:grouped[(row['scenario_id'],row['scenario_offset'])].append(row)
    cls=scenario_class();files=known=total=measured=0
    with open(shard,'rb') as reader:
        for (sid,offset),group in grouped.items():
            reader.seek(offset);header=reader.read(12)
            if len(header)!=12:raise IOError('Missing Scenario record')
            size=struct.unpack('<Q',header[:8])[0];payload=reader.read(size)
            if len(payload)!=size or len(reader.read(4))!=4:raise IOError('Truncated Scenario')
            scene=cls();scene.ParseFromString(payload)
            if scene.scenario_id!=sid:raise ValueError('Scenario identity mismatch')
            features={int(f.id):f for f in scene.map_features};tracks={int(t.id):t for t in scene.tracks};profiles={}
            for row in group:
                name=Path(row['npz_path']).name;target=Path(out_root)/row['split']/name
                target.parent.mkdir(parents=True,exist_ok=True)
                with np.load(Path(map_root)/row['split']/name,allow_pickle=False) as d:
                    ids=d['map_ids'];lane=d['map_is_lane']
                if target.exists():
                    with np.load(target,allow_pickle=False) as d:
                        if int(d['geometry_version'])!=VERSION or str(d['scenario_id'])!=sid or not np.array_equal(ids,d['map_ids']):raise ValueError('Stale loss geometry')
                        known+=int(d['lane_width_mask'].sum());measured+=int(d['agent_width_mask'].sum())
                else:
                    widths=np.zeros((len(ids),2),np.float32);mask=np.zeros(len(ids),bool)
                    for fid in np.unique(ids[lane]):
                        indices=np.flatnonzero((ids==fid)&lane);key=(int(fid),len(indices))
                        if key not in profiles:profiles[key]=lane_profile(features[int(fid)].lane,features,len(indices))
                        widths[indices],mask[indices]=profiles[key]
                    with np.load(row['npz_path'],allow_pickle=False) as d:
                        agent_ids=d['agent_ids'];agent_mask=d['agent_mask'];lengths=d['agent_lengths']
                    vehicle_width=np.zeros(len(agent_ids),np.float32);vehicle_mask=np.zeros(len(agent_ids),bool)
                    for slot in np.flatnonzero(agent_mask):
                        track=tracks[int(agent_ids[slot])]
                        positive=[s.width for s in track.states if s.valid and np.isfinite(s.width) and s.width>0]
                        if positive and np.isfinite(lengths[slot]) and lengths[slot]>0:
                            vehicle_width[slot]=positive[0];vehicle_mask[slot]=True
                    tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
                    with tmp.open('wb') as writer:
                        np.savez_compressed(writer,lane_half_widths=widths,lane_width_mask=mask,
                            agent_widths=vehicle_width,agent_width_mask=vehicle_mask,agent_ids=agent_ids,map_ids=ids,
                            scenario_id=np.asarray(sid),geometry_version=np.asarray(VERSION))
                    os.replace(tmp,target);known+=int(mask.sum());measured+=int(vehicle_mask.sum())
                total+=int(lane.sum());files+=1
    return dict(shard=str(shard),files=files,known_road_segments=known,lane_segments=total,measured_agent_widths=measured)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['data_root','map_cache_dir','index_dir','output_dir']:p.add_argument('--'+arg,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max_files',type=int,default=0)
    args=p.parse_args()
    with (Path(args.data_root)/'manifest.csv').open() as f:rows=list(csv.DictReader(f))
    if args.max_files:rows=rows[:args.max_files]
    rows=rebase_manifest_rows(rows,args.data_root);wanted={r['scenario_id'] for r in rows};locations={}
    for index in sorted(Path(args.index_dir).glob('*.jsonl')):
        with index.open() as f:
            meta=json.loads(next(f));stat=Path(meta['shard']).stat()
            if stat.st_size!=meta['size'] or stat.st_mtime_ns!=meta['mtime_ns']:raise ValueError('Stale Scenario index')
            for line in f:
                loc=json.loads(line)
                if loc['scenario_id'] in wanted:
                    if loc['scenario_id'] in locations:raise ValueError('Duplicate Scenario ID')
                    locations[loc['scenario_id']]=(meta['shard'],loc['offset'])
    if wanted-set(locations):raise ValueError('Incomplete Scenario index')
    jobs=defaultdict(list)
    for row in rows:
        shard,offset=locations[row['scenario_id']];row['scenario_offset']=offset;jobs[shard].append(row)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);reports=[];completed=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(process_shard,(s,r,args.map_cache_dir,args.output_dir)) for s,r in jobs.items()]
        for future in as_completed(futures):
            report=future.result();reports.append(report);completed+=report['files']
            print(json.dumps(dict(event='loss_geometry_progress',completed=completed,total=len(rows),**report)),flush=True)
    summary=dict(version=VERSION,files=completed,complete=not bool(args.max_files),
       map_cache_dir=str(Path(args.map_cache_dir).resolve()),data_root=str(Path(args.data_root).resolve()),
       known_road_segments=sum(r['known_road_segments'] for r in reports),lane_segments=sum(r['lane_segments'] for r in reports))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
