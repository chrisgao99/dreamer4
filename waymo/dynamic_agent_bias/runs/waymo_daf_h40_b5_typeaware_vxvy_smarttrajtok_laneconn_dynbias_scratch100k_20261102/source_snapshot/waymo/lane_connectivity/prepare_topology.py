#!/usr/bin/env python3
"""Independent segment graph sidecars; reuse baseline 5m maps verbatim.

Directed edge types: 1 same-lane hold/forward, 2 legal successor,
3 permitted left change, 4 permitted right change. Lane changes require the
whole source segment to have a known broken marking and a valid neighbor span.
Solid, mixed, unknown and road-edge boundaries never create a change edge.
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
BROKEN_TYPES={1,4,5}


def build_graph(scene,map_ids,is_lane):
    lanes={int(f.id):f.lane for f in scene.map_features if f.HasField('lane')}
    groups={int(fid):np.flatnonzero((map_ids==fid)&is_lane) for fid in np.unique(map_ids[is_lane])}
    edges=[{} for _ in map_ids]
    counts=dict(continuation=0,successor=0,left_change=0,right_change=0,excluded_unknown_or_solid=0)
    def add(src,dst,kind):
        edges[int(src)][int(dst)]=kind
        if src!=dst:counts[['','continuation','successor','left_change','right_change'][kind]]+=1
    def arc(lane):
        pts=np.array([[p.x,p.y] for p in lane.polyline],float).reshape(-1,2)
        return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=-1))]
    for fid,indices in groups.items():
        if fid not in lanes:raise ValueError(f'Lane ID absent from raw Scenario: {fid}')
        lane=lanes[fid];dist=arc(lane)
        total=dist[-1]
        # Sidecars follow baseline order: whole lane, fixed 5m arc segments.
        expected=max(1,int(np.ceil(total/5.)))
        if len(indices)!=expected and not (abs(len(indices)-expected)==1 and abs(total-5.*min(len(indices),expected))<1e-3):
            raise ValueError(f'Lane segment count changed: {fid} {len(indices)} != {expected}')
        for i,src in enumerate(indices):
            add(src,src,1)
            if i+1<len(indices):add(src,indices[i+1],1)
        for successor in lane.exit_lanes:
            if int(successor) in groups:add(indices[-1],groups[int(successor)][0],2)
        for name,kind in [('left_neighbors',3),('right_neighbors',4)]:
            for neighbor in getattr(lane,name):
                other=int(neighbor.feature_id)
                if other not in groups:continue
                other_dist=arc(lanes[other]);dest=groups[other]
                if not len(dist) or not len(other_dist):continue
                a,b=neighbor.self_start_index,neighbor.self_end_index
                c,d=neighbor.neighbor_start_index,neighbor.neighbor_end_index
                if not (0<=a<=b<len(dist) and 0<=c<=d<len(other_dist)):
                    raise ValueError('Invalid Waymo neighbor span')
                for i,src in enumerate(indices):
                    lo=min(i*5.,total);hi=min((i+1)*5.,total)
                    if hi-lo<1e-6:continue
                    if lo<dist[a]-1e-4 or hi>dist[b]+1e-4:continue
                    boundary=getattr(lane,'left_boundaries' if kind==3 else 'right_boundaries')
                    candidates=list(neighbor.boundaries) or list(boundary)
                    intervals=[];blocked=False
                    for mark in candidates:
                        x,y=mark.lane_start_index,mark.lane_end_index
                        if not (0<=x<=y<len(dist)):raise ValueError('Invalid boundary span')
                        left=max(lo,dist[x]);right=min(hi,dist[y])
                        if right-left<=1e-6:continue
                        if mark.boundary_type not in BROKEN_TYPES:blocked=True
                        else:intervals.append((left,right))
                    # Union coverage must span the entire source segment.
                    covered=lo
                    for left,right in sorted(intervals):
                        if left>covered+1e-4:break
                        covered=max(covered,right)
                    if blocked or covered<hi-1e-4:
                        counts['excluded_unknown_or_solid']+=1;continue
                    mid=(lo+hi)*.5
                    fraction=(mid-dist[a])/max(dist[b]-dist[a],1e-6)
                    target_arc=other_dist[c]+fraction*(other_dist[d]-other_dist[c])
                    target_idx=min(len(dest)-1,max(0,int(target_arc//5.)))
                    add(src,dest[target_idx],kind)
    width=max(1,max(map(len,edges)))
    out=np.full((len(map_ids),width),-1,np.int64);types=np.zeros_like(out)
    for src,adj in enumerate(edges):
        for j,(dst,kind) in enumerate(sorted(adj.items())):out[src,j]=dst;types[src,j]=kind
    return out,types,counts


def process_shard(job):
    shard,rows,map_root,out_root=job
    grouped=defaultdict(list)
    for row in rows:grouped[(row['scenario_id'],row['scenario_offset'])].append(row)
    cls=scenario_class();counts=defaultdict(int);files=0
    with open(shard,'rb') as reader:
        for (sid,offset),group in grouped.items():
            reader.seek(offset);header=reader.read(12)
            if len(header)!=12:raise IOError('Missing Scenario header')
            size=struct.unpack('<Q',header[:8])[0];payload=reader.read(size)
            if len(payload)!=size or len(reader.read(4))!=4:raise IOError('Truncated Scenario')
            scene=cls();scene.ParseFromString(payload)
            if scene.scenario_id!=sid:raise ValueError('Scenario ID/index mismatch')
            for row in group:
                name=Path(row['npz_path']).name;source=Path(map_root)/row['split']/name
                target=Path(out_root)/row['split']/name;target.parent.mkdir(parents=True,exist_ok=True)
                with np.load(source,allow_pickle=False) as d:
                    ids=d['map_ids'];lane=d['map_is_lane']
                    if str(d['scenario_id'])!=sid or float(d['segment_length_m'])!=5.:raise ValueError('Baseline map cache mismatch')
                if target.exists():
                    with np.load(target,allow_pickle=False) as d:
                        if int(d['topology_version'])!=VERSION or str(d['scenario_id'])!=sid or not np.array_equal(d['map_ids'],ids):
                            raise ValueError('Stale topology cache')
                        stats=json.loads(str(d['counts_json']))
                else:
                    edges,types,stats=build_graph(scene,ids,lane)
                    tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
                    with tmp.open('wb') as writer:
                        np.savez_compressed(writer,lane_edges=edges,lane_edge_types=types,map_ids=ids,
                            scenario_id=np.asarray(sid),topology_version=np.asarray(VERSION),counts_json=np.asarray(json.dumps(stats)))
                    os.replace(tmp,target)
                for key,value in stats.items():counts[key]+=value
                files+=1
    return dict(shard=str(shard),files=files,**counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root',required=True);p.add_argument('--map_cache_dir',required=True)
    p.add_argument('--index_dir',required=True);p.add_argument('--output_dir',required=True)
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
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    reports=[];total=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(process_shard,(s,r,args.map_cache_dir,args.output_dir)) for s,r in jobs.items()]
        for future in as_completed(futures):
            report=future.result();reports.append(report);total+=report['files']
            print(json.dumps(dict(event='topology_progress',completed=total,total=len(rows),**report)),flush=True)
    summary=dict(version=VERSION,files=total,complete=not bool(args.max_files),
                 data_root=str(Path(args.data_root).resolve()),map_cache_dir=str(Path(args.map_cache_dir).resolve()),
                 shards=len(reports),edge_counts={key:sum(r.get(key,0) for r in reports) for key in
                    ['continuation','successor','left_change','right_change','excluded_unknown_or_solid']})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(event='topology_complete',**summary)),flush=True)

if __name__=='__main__':main()
