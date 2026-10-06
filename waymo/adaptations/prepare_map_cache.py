#!/usr/bin/env python3
"""Rebuild uncapped ~5 m maps and exact lane controls from raw Scenario records.

Keep agents/splits/statistics in the original NPZs untouched. Write one sidecar
per focus NPZ. Resume is supported; identity and preparation settings are checked.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from waymo.adaptations.scenario_reader import scenario_class
import struct
from waymo.core.waymo_vector_filter import _rotate_xy

VERSION = 1


def metric_segments(xy, map_type, length=5., points=6):
    """Arc-length interpolation, shared endpoints, preserve final short segment."""
    xy = np.asarray(xy, np.float64)
    if len(xy) == 0: return []
    if not np.isfinite(xy).all(): raise ValueError('Nonfinite map geometry')
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    xy = xy[keep]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    total = arc[-1]
    output = []
    for start in np.arange(0., max(total, 1e-6), length):
        end = min(start + length, total)
        samples = np.linspace(start, end, points)
        coords = np.stack([np.interp(samples, arc, xy[:, i]) for i in range(2)], -1)
        # Preserve interior vertices as samples when the polyline bends: the
        # resampled representation approximates curves at <=1m for 5m/6 points.
        delta = np.gradient(coords, axis=0)
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        direction = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-8)
        feat = np.zeros((points, 6), np.float32)
        feat[:, :2], feat[:, 2:4], feat[:, 4], feat[:, 5] = coords, direction, map_type, 1.
        output.append(feat)
    return output


def geometry(scene):
    result = []
    stops = defaultdict(list)
    for f in scene.map_features:
        if f.HasField('stop_sign'):
            for lane in f.stop_sign.lane:
                stops[lane].append([f.stop_sign.position.x, f.stop_sign.position.y])
        for name, offset in [('lane',0),('road_line',5),('road_edge',14),
                             ('crosswalk',18),('speed_bump',19),('driveway',20)]:
            if not f.HasField(name): continue
            value = getattr(f, name)
            polygon = name in ('crosswalk','speed_bump','driveway')
            pts = value.polygon if polygon else value.polyline
            xy = np.array([[p.x,p.y] for p in pts], np.float64).reshape(-1,2)
            if polygon and len(xy): xy = np.concatenate((xy,xy[:1]),0)
            kind = offset if polygon else offset + value.type
            result.append((int(f.id), kind, xy, name == 'lane'))
    return result, stops


def build_cache(scene, source, length, points):
    with np.load(source, allow_pickle=False) as d:
        if str(d['scenario_id']) != scene.scenario_id: raise ValueError('Scenario ID mismatch')
        cfg = json.loads(str(d['config_json']))
        if not cfg['normalize_to_ego']: raise ValueError('Expected normalized scene coordinates')
        origin = d['ego_origin_xy']; heading = float(d['ego_heading'])
        agents = d['agents']; mask = d['agent_mask']; ids = d['agent_ids']
        if agents.shape[0] != len(mask): agents = agents.transpose(1,0,2)
        if len(scene.timestamps_seconds) != agents.shape[1] or scene.current_time_index != 10:
            raise ValueError('Scenario/NPZ temporal alignment differs')
        # Verify track IDs and world positions at the original current frame.
        tracks = {t.id:t for t in scene.tracks}
        source_indices=d['agent_src_indices']
        for slot in np.flatnonzero(mask):
            if scene.tracks[int(source_indices[slot])].id != int(ids[slot]):
                raise ValueError('Scenario/NPZ source track ordering differs')
        for slot in np.flatnonzero(mask & (agents[:,10,5] > .5)):
            state = tracks[int(ids[slot])].states[10]
            expected = _rotate_xy(np.array([[state.center_x,state.center_y]])-origin,heading)[0]
            if not state.valid or not np.allclose(expected,agents[slot,10,:2],atol=.05):
                raise ValueError('Scenario track/frame transform mismatch')
        if len(scene.dynamic_map_states) != len(scene.timestamps_seconds):
            raise ValueError('Incomplete dynamic map time axis')
        crop_indices = d['map_crop_src_indices'].tolist()
        crop = []
        for idx in crop_indices:
            track = scene.tracks[idx]
            crop.extend([[s.center_x,s.center_y] for s in track.states if s.valid])
        crop = np.asarray(crop).reshape(-1,2)
        if not len(crop): raise ValueError('Empty map crop trajectory')
        required_ids = set(d['map_ids'][d['map_mask'].any(-1)].tolist())
        threshold = cfg['map_distance_threshold']
    features, stops = geometry(scene)
    segments=[]; map_ids=[]; flags=[]; stop_positions=[]; lane_flags=[]
    included_ids = set()
    for fid,kind,xy,is_lane in features:
        if not len(xy): continue
        close = False
        for start in range(0,len(xy),256):
            if (np.linalg.norm(xy[start:start+256,None]-crop[None],axis=-1) <= threshold).any():
                close = True; break
        if not close and fid not in required_ids: continue
        included_ids.add(fid)
        local = _rotate_xy(xy-origin,heading)
        sign_xy = _rotate_xy(np.asarray(stops.get(fid, [[0.,0.]]))-origin,heading)
        for seg in metric_segments(local,kind,length,points):
            # Multiple signs can control a lane; retain nearest sign to each segment.
            index = np.linalg.norm(sign_xy-seg[:,:2].mean(0),axis=-1).argmin()
            segments.append(seg); map_ids.append(fid); lane_flags.append(is_lane)
            flags.append(is_lane and fid in stops)
            stop_positions.append(sign_xy[index] if flags[-1] else [0.,0.])
    missing = required_ids - included_ids
    # Standalone stop-sign tokens are replaced by lane attributes, so exempt them.
    sign_ids = {int(f.id) for f in scene.map_features if f.HasField('stop_sign')}
    if missing - sign_ids: raise ValueError(f'Original map features missing: {missing-sign_ids}')
    if not segments:
        segments=[np.zeros((points,6),np.float32)];map_ids=[-1];flags=[False];stop_positions=[[0.,0.]];lane_flags=[False]
    lanes = sorted({int(s.lane) for frame in scene.dynamic_map_states for s in frame.lane_states})
    # Keep a masked dummy slot for scenes with no observed traffic lights.
    light_ids = np.full((len(scene.timestamps_seconds),max(1,len(lanes))),-1,np.int64)
    lights = np.zeros((*light_ids.shape,4),np.float32); light_mask = np.zeros(light_ids.shape,bool)
    lookup = {lane:i for i,lane in enumerate(lanes)}
    for t,frame in enumerate(scene.dynamic_map_states):
        for s in frame.lane_states:
            i = lookup[int(s.lane)]
            xy = _rotate_xy(np.array([[s.stop_point.x,s.stop_point.y]])-origin,heading)[0]
            light_ids[t,i]=s.lane; lights[t,i]=[*xy,s.state,1.]; light_mask[t,i]=True
    polylines = np.stack(segments)
    return dict(map_polylines=polylines,map_mask=polylines[...,5]>.5,
                map_ids=np.array(map_ids,np.int64),map_is_lane=np.array(lane_flags,bool),
                map_stop_sign=np.array(flags,bool),map_stop_point=np.asarray(stop_positions,np.float32),
                lights=lights,light_mask=light_mask,light_ids=light_ids,
                scenario_id=np.asarray(scene.scenario_id),cache_version=np.asarray(VERSION),
                segment_length_m=np.asarray(length),points_per_segment=np.asarray(points),
                source_npz=np.asarray(str(Path(source).resolve())))


def process_shard(job):
    shard, rows, output, length, points = job
    wanted = defaultdict(list)
    for row in rows: wanted[row['scenario_id']].append(row)
    found=set(); count=0; max_segments=0
    cls = scenario_class()
    selected = {}
    for row in rows: selected[int(row['scenario_offset'])] = row['scenario_id']
    stream = open(shard, 'rb')
    for offset, expected_id in sorted(selected.items()):
        stream.seek(offset)
        header = stream.read(12)
        if len(header) != 12: raise IOError('Truncated Scenario header')
        size = struct.unpack('<Q', header[:8])[0]
        payload = stream.read(size)
        if len(payload) != size or len(stream.read(4)) != 4: raise IOError('Truncated Scenario')
        scene = cls(); scene.ParseFromString(payload)
        if scene.scenario_id != expected_id: raise ValueError('Stale Scenario offset index')
        if scene.scenario_id not in wanted: continue
        for row in wanted[scene.scenario_id]:
            source=Path(row['npz_path']); target=Path(output)/row['split']/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                with np.load(target,allow_pickle=False) as d:
                    if (int(d['cache_version']) != VERSION or float(d['segment_length_m']) != length
                        or int(d['points_per_segment']) != points or str(d['scenario_id']) != scene.scenario_id
                        or str(d['source_npz']) != str(source.resolve())):
                        raise ValueError(f'Stale map cache: {target}')
                    n=len(d['map_ids'])
            else:
                data=build_cache(scene,source,length,points); n=len(data['map_ids'])
                tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
                with tmp.open('wb') as writer: np.savez_compressed(writer,**data)
                os.replace(tmp,target)
            count+=1; max_segments=max(max_segments,n)
        found.add(scene.scenario_id)
        if len(found)==len(wanted): break
    stream.close()
    if set(wanted)-found: raise ValueError(f'{shard}: missing scenarios {set(wanted)-found}')
    return dict(shard=str(shard),files=count,max_segments=max_segments)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root',required=True);p.add_argument('--output_dir',required=True)
    p.add_argument('--scenario_root',default='/p/liverobotics/waymo_open_dataset_motion/scenario')
    p.add_argument('--index_dir',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max_files',type=int,default=0)
    p.add_argument('--segment_length_m',type=float,default=5.);p.add_argument('--points_per_segment',type=int,default=6)
    args=p.parse_args()
    if args.segment_length_m<=0 or args.points_per_segment<2: raise ValueError('Invalid segment geometry settings')
    with (Path(args.data_root)/'manifest.csv').open() as stream: rows=list(csv.DictReader(stream))
    if args.max_files: rows=rows[:args.max_files]
    wanted={row['scenario_id'] for row in rows}
    locations={}
    for index in sorted(Path(args.index_dir).glob('*.jsonl')):
        with index.open() as stream:
            meta=json.loads(next(stream)); shard=Path(meta['shard'])
            stat=shard.stat()
            if stat.st_size != meta['size'] or stat.st_mtime_ns != meta['mtime_ns']:
                raise ValueError(f'Stale Scenario index: {index}')
            for line in stream:
                loc=json.loads(line); sid=loc['scenario_id']
                if sid in wanted:
                    if sid in locations: raise ValueError(f'Duplicate Scenario ID: {sid}')
                    locations[sid]=(str(shard),loc['offset'])
    if wanted-set(locations):
        raise ValueError(f"Missing {len(wanted-set(locations))} scenarios in index; run index_scenarios.py first")
    jobs=defaultdict(list)
    for row in rows:
        shard,offset=locations[row['scenario_id']]
        row['scenario_offset']=offset
        jobs[shard].append(row)
    tasks=[(s,r,args.output_dir,args.segment_length_m,args.points_per_segment) for s,r in jobs.items()]
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    reports=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report in pool.map(process_shard,tasks):
            reports.append(report); print(json.dumps(report),flush=True)
    summary=dict(version=VERSION,files=sum(r['files'] for r in reports),shards=len(reports),
                 max_segments=max(r['max_segments'] for r in reports),data_root=str(Path(args.data_root).resolve()),
                 scenario_root=args.scenario_root,segment_length_m=args.segment_length_m,
                 points_per_segment=args.points_per_segment,complete=(not args.max_files))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
