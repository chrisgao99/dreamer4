#!/usr/bin/env python3
"""Repair missing current-frame static lengths using each track's first valid measurement.

Only the already-enriched dataset is updated, atomically. Original source NPZs
are untouched, and their compressed members are copied verbatim. This handles
tracks absent at the original current frame but present at later training anchors.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import csv
from functools import partial
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from waymo.data_prep.backfill_waymo_agent_lengths import atomic_json,enrich_one
from waymo.core.waymo_vector_filter import _get_tfexample_class,_feature_to_numpy


def find_missing(row, *, first_required_frame=0):
    with np.load(row['npz_path'],allow_pickle=False) as d:
        a=d['agents'];mask=d['agent_mask'].astype(bool);length=d['agent_lengths']
        if a.shape[0]!=len(mask):a=a.transpose(1,0,2)
        wheels=np.isin(a[:,10,7].round(),[1,3])
        valid=(a[:,first_required_frame:,5]>.5).any(-1)
        missing=mask&wheels&valid&((length<=0)|~np.isfinite(length))
        if not missing.any():return None
        return dict(row,missing_slots=np.flatnonzero(missing).tolist())


def scan(root,workers):
    with (root/'manifest.csv').open() as f:rows=list(csv.DictReader(f))
    missing=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0,len(rows),256):
            missing.extend(x for x in pool.map(find_missing,rows[offset:offset+256]) if x is not None)
            if offset%2048==0:
                print(json.dumps(dict(event='scan',files=min(offset+256,len(rows)),total=len(rows),affected_files=len(missing))),flush=True)
    return missing,len(rows)


def first_valid_lengths(features):
    lengths=[];valid=[]
    for part,steps in [('past',10),('current',1),('future',80)]:
        lengths.append(np.asarray(_feature_to_numpy(features[f'state/{part}/length']),np.float32).reshape(128,steps))
        valid.append(np.asarray(_feature_to_numpy(features[f'state/{part}/valid'])).reshape(128,steps)>0)
    length=np.concatenate(lengths,axis=1);good=np.concatenate(valid,axis=1)&np.isfinite(length)&(length>0)
    first=good.argmax(1)
    result=np.where(good.any(1),length[np.arange(128),first],np.nan)
    return result,first


def repair_shard(shard,rows,first_required_frame=0):
    by_record=defaultdict(list)
    for r in rows:by_record[int(r['record_index'])].append(r)
    repaired=[]
    with open(shard,'rb') as f:
        size=os.fstat(f.fileno()).st_size
        for index in range(max(by_record)+1):
            header=f.read(12)
            if len(header)!=12:raise IOError(f'Truncated TFRecord {shard}:{index}')
            length=struct.unpack('<Q',header[:8])[0]
            if f.tell()+length+4>size:raise IOError('Truncated TFRecord payload')
            if index not in by_record:
                f.seek(length+4,1);continue
            payload=f.read(length);f.read(4)
            example=_get_tfexample_class()();example.ParseFromString(payload)
            feat=example.features.feature
            scenario=_feature_to_numpy(feat['scenario/id'])
            ids=np.asarray(_feature_to_numpy(feat['state/id'])).astype(np.int64)
            measured,first=first_valid_lengths(feat)
            for row in by_record[index]:
                target=Path(row['npz_path']);source=Path(row['source_npz_path'])
                with np.load(target,allow_pickle=False) as d:
                    if str(d['scenario_id'])!=scenario or scenario!=row['scenario_id']:raise ValueError('Scenario mismatch')
                    slots=np.asarray(row['missing_slots']);indices=d['agent_src_indices'][slots]
                    if not np.array_equal(d['agent_ids'][slots],ids[indices]):raise ValueError('Track identity mismatch')
                    lengths=d['agent_lengths'].copy();old=lengths[slots].copy()
                    new=measured[indices]
                    available=np.isfinite(new)&(new>0)
                    agents=d['agents']
                    if agents.shape[0]!=len(d['agent_mask']):agents=agents.transpose(1,0,2)
                    required=(agents[slots,first_required_frame:,5]>.5).any(-1)
                    unresolved=~available & (~np.isfinite(old)|(old<=0))
                    bad=unresolved & required
                    if bad.any():
                        raise ValueError(f'No valid measured length for active tracks: {target}; '
                            f'slots={slots[bad].tolist()}, track_ids={d["agent_ids"][slots[bad]].tolist()}, '
                            f'first_required_frame={first_required_frame}')
                    skipped=[dict(slot=int(slot),track_id=int(d['agent_ids'][slot]),
                        source_index=int(d['agent_src_indices'][slot]),
                        reason='no positive measurement; inactive from first_required_frame onward')
                        for slot in slots[unresolved & ~required]]
                    # Keep existing positive dimensions even when retrying an old audit.
                    fill=available & (~np.isfinite(old)|(old<=0))
                    lengths[slots[fill]]=new[fill]
                tmp=target.with_name(target.name+f'.repair.{os.getpid()}')
                try:
                    enrich_one(source,tmp,lengths,resume=False)
                    os.replace(tmp,target)
                finally:tmp.unlink(missing_ok=True)
                repaired.append(dict(npz_path=str(target),slots=slots.tolist(),old_lengths=old.tolist(),
                    new_lengths=lengths[slots].tolist(),first_valid_frame=np.where(available,first[indices],-1).tolist(),
                    first_required_frame=first_required_frame,unresolved_inactive_tracks=skipped))
    return repaired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--history_length',type=int,default=1,help='Earliest required frame is history_length-1; later active tracks remain strict')
    p.add_argument('--audit_only',action='store_true');p.add_argument('--audit_path',required=True)
    args=p.parse_args();root=Path(args.data_root).resolve();audit=Path(args.audit_path)
    if args.workers<1:raise ValueError('workers must be positive')
    if not 1<=args.history_length<=91:raise ValueError('history_length must be in [1,91]')
    first_required_frame=args.history_length-1
    started=time.time()
    if audit.exists():
        report=json.loads(audit.read_text())
        if report['data_root']!=str(root):raise ValueError('Audit source mismatch')
        missing=report['missing'];total=report['total_files']
    else:
        missing,total=scan(root,args.workers)
        report=dict(data_root=str(root),total_files=total,missing=missing,affected_files=len(missing),
                    missing_tracks=sum(len(r['missing_slots']) for r in missing))
        audit.parent.mkdir(parents=True,exist_ok=True);atomic_json(audit,report)
    print(json.dumps({k:v for k,v in report.items() if k!='missing'}),flush=True)
    if args.audit_only:return
    groups=defaultdict(list)
    for row in missing:groups[row['tfrecord_path']].append(row)
    journal=root/'length_supplement_journal.jsonl'
    done=[]
    with journal.open('a') as log,ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(repair_shard,s,rs,first_required_frame):s for s,rs in groups.items()}
        for future in as_completed(futures):
            try:entries=future.result()
            except BaseException:
                for queued in futures:queued.cancel()
                raise
            done.extend(entries)
            for entry in entries:log.write(json.dumps(entry)+'\n')
            log.flush()
            print(json.dumps(dict(event='repair',files=len(done),total=len(missing),elapsed_seconds=round(time.time()-started,1))),flush=True)
    # Recheck only affected files; all others were checked by the audit scan.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if any(x is not None for x in pool.map(partial(find_missing,first_required_frame=first_required_frame),missing)):
            raise RuntimeError('Missing dimensions remain')
    atomic_json(root/'length_supplement_summary.json',dict(status='complete',total_files=total,
        repaired_files=len(done),repaired_tracks=sum(len(r['slots'])-len(r['unresolved_inactive_tracks']) for r in done),
        first_required_frame=first_required_frame,
        unresolved_inactive_tracks=sum(len(r['unresolved_inactive_tracks']) for r in done),
        method='missing current length -> first positive measured length at a valid track frame',
        source_manifest_preserved=True,original_npzs_untouched=True,elapsed_seconds=time.time()-started))
    print('Length supplementation complete',flush=True)


if __name__=='__main__':main()
