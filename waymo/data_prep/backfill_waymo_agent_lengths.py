#!/usr/bin/env python3
"""Add measured lengths to existing NPZs without recomputing or recompressing features.

Reads only selected TFRecord payloads, once per scenario, using the existing
manifest's shard/record index. Original NPZ ZIP members are copied verbatim;
only agent_lengths.npy is appended. Inputs are never modified.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import zipfile

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from waymo.core.waymo_vector_filter import _get_tfexample_class, _feature_to_numpy


RAW_FIELDS = ('scenario/id', 'state/id', 'state/type', 'state/current/valid', 'state/current/length')


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def selected_records(path, indices):
    """Seek over unwanted payloads; no TF/protobuf parsing for skipped scenes."""
    wanted = set(indices)
    if not wanted:
        return
    if min(wanted) < 0:
        raise ValueError('Negative TFRecord index')
    with open(path, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        for index in range(max(wanted) + 1):
            header = f.read(12)
            if len(header) != 12:
                raise IOError(f'{path}: missing/truncated record header at index {index}')
            length = struct.unpack('<Q', header[:8])[0]
            if f.tell() + length + 4 > size:
                raise IOError(f'{path}: truncated record payload at index {index}')
            if index not in wanted:
                f.seek(length + 4, os.SEEK_CUR)
                continue
            payload = f.read(length)
            footer = f.read(4)
            if len(payload) != length or len(footer) != 4:
                raise IOError(f'{path}: truncated selected record {index}')
            example = _get_tfexample_class()()
            example.ParseFromString(payload)
            features = example.features.feature
            missing = set(RAW_FIELDS) - set(features)
            if missing:
                raise KeyError(f'{path}:{index}: missing raw features {sorted(missing)}')
            yield index, {key: _feature_to_numpy(features[key]) for key in RAW_FIELDS}


def aligned_lengths(source, row, raw):
    """Cross-check both source index and track ID; never rely on slot order alone."""
    with np.load(source, allow_pickle=False) as data:
        scenario = str(data['scenario_id'])
        if scenario != row['scenario_id'] or scenario != raw['scenario/id']:
            raise ValueError(f'{source}: scenario identity mismatch')
        if 'source_record_index' in data and int(data['source_record_index']) != int(row['record_index']):
            raise ValueError(f'{source}: record index mismatch')
        if 'source_tfrecord_path' in data and Path(str(data['source_tfrecord_path'])).resolve() != Path(row['tfrecord_path']).resolve():
            raise ValueError(f'{source}: TFRecord path mismatch')
        mask = np.asarray(data['agent_mask'], dtype=bool)
        src = np.asarray(data['agent_src_indices'], dtype=np.int64)
        ids = np.asarray(data['agent_ids'], dtype=np.int64)
        raw_ids = np.asarray(raw['state/id']).reshape(-1).astype(np.int64)
        lengths = np.asarray(raw['state/current/length'], dtype=np.float32).reshape(-1)
        types = np.asarray(raw['state/type']).reshape(-1).astype(np.int64)
        current_valid = np.asarray(raw['state/current/valid']).reshape(-1) > 0
        if not (src.shape == ids.shape == mask.shape) or mask.ndim != 1:
            raise ValueError(f'{source}: incompatible agent metadata shapes')
        if not (raw_ids.shape == lengths.shape == types.shape == current_valid.shape):
            raise ValueError(f'{source}: incompatible raw metadata shapes')
        if np.any((src[mask] < 0) | (src[mask] >= len(lengths))):
            raise ValueError(f'{source}: selected agent source index out of range')
        if not np.array_equal(ids[mask], raw_ids[src[mask]]):
            raise ValueError(f'{source}: agent ID/source index mismatch')
        agents = data['agents']
        if agents.ndim != 3:
            raise ValueError(f'{source}: invalid agents shape')
        if agents.shape[0] == len(mask):
            current = agents[:, 10]
        elif agents.shape[1] == len(mask):
            current = agents[10]
        else:
            raise ValueError(f'{source}: cannot identify agent axis')
        if not np.array_equal(current[mask, 7].round().astype(np.int64), types[src[mask]]):
            raise ValueError(f'{source}: agent type mismatch')
        if not np.array_equal(current[mask, 5] > .5, current_valid[src[mask]]):
            raise ValueError(f'{source}: current validity mismatch')
        result = np.zeros(mask.shape, dtype=np.float32)
        result[mask] = lengths[src[mask]]
        wheeled = mask & (current[:, 5] > .5) & np.isin(current[:, 7].round(), [1, 3])
        if np.any(wheeled & (~np.isfinite(result) | (result <= 0))):
            raise ValueError(f'{source}: active vehicle/cyclist missing a positive measured length')
        if 'agent_lengths' in data and not np.array_equal(data['agent_lengths'], result, equal_nan=True):
            raise ValueError(f'{source}: existing agent_lengths disagree with raw data')
        return result


def verify_output(source, output, lengths):
    """Verify preserved ZIP directory entries and the new NPY payload."""
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output) as enriched:
        old = {i.filename: i for i in original.infolist()}
        new = {i.filename: i for i in enriched.infolist()}
        if len(new) != len(enriched.infolist()) or len(old) != len(original.infolist()):
            raise ValueError(f'{output}: duplicate ZIP members')
        if set(new) != set(old) | {'agent_lengths.npy'}:
            raise ValueError(f'{output}: unexpected or missing NPZ fields')
        for name, before in old.items():
            after = new[name]
            if (before.CRC, before.file_size, before.compress_size, before.compress_type) != (
                    after.CRC, after.file_size, after.compress_size, after.compress_type):
                raise ValueError(f'{output}: original ZIP member changed: {name}')
        actual = np.load(io.BytesIO(enriched.read('agent_lengths.npy')), allow_pickle=False)
        if actual.dtype != np.float32 or not np.array_equal(actual, lengths, equal_nan=True):
            raise ValueError(f'{output}: incorrect agent_lengths')


def enrich_one(source, output, lengths, *, resume):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Refusing to modify original NPZ')
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        verify_output(source, output, lengths)
        return 'verified_existing'
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f'.{os.getpid()}.tmp')
    try:
        # Preserve existing compressed feature bytes. np.savez would recompress
        # everything; ZIP append only creates the new NPY and directory entries.
        shutil.copyfile(source, tmp)
        with zipfile.ZipFile(tmp, 'a', compression=zipfile.ZIP_DEFLATED) as archive:
            if 'agent_lengths.npy' not in archive.namelist():
                buf = io.BytesIO()
                np.save(buf, lengths, allow_pickle=False)
                archive.writestr('agent_lengths.npy', buf.getvalue())
        verify_output(source, tmp, lengths)
        # Link publishes atomically and refuses to overwrite an existing path.
        os.link(tmp, output)
        return 'written'
    finally:
        tmp.unlink(missing_ok=True)


def process_shard(shard, rows, output_dir, resume):
    by_record = defaultdict(list)
    for row in rows:
        by_record[int(row['record_index'])].append(row)
    counts = Counter()
    for index, raw in selected_records(shard, by_record):
        for row in by_record[index]:
            source = Path(row['npz_path'])
            output = Path(output_dir) / row['split'] / source.name
            lengths = aligned_lengths(source, row, raw)
            result = enrich_one(source, output, lengths, resume=resume)
            counts[result] += 1
    if sum(counts.values()) != len(rows):
        raise RuntimeError(f'{shard}: did not process every requested NPZ')
    return dict(counts)


def plan(args):
    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Use a separate sibling output directory, not the input tree')
    manifest = source / 'manifest.csv'
    with manifest.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    by_shard = defaultdict(list)
    destinations = set()
    expected_paths = set()
    for row in rows:
        if row['split'] not in ('train', 'val'):
            raise ValueError('Unsupported split: ' + row['split'])
        path = Path(os.path.abspath(row['npz_path']))
        if path.parent != source / row['split']:
            raise ValueError(f'Manifest NPZ missing or outside source split: {path}')
        key = (row['split'], path.name)
        if key in destinations:
            raise ValueError('Duplicate NPZ in manifest: ' + str(path))
        destinations.add(key)
        expected_paths.add(path)
        by_shard[row['tfrecord_path']].append(row)
    actual_paths = {p for split in ('train', 'val') for p in (source/split).glob('*.npz')}
    if expected_paths != actual_paths:
        raise ValueError('Manifest and train/val directory membership differ')
    if not rows:
        raise ValueError('Empty manifest')
    shards = sorted(by_shard)
    if args.max_shards:
        shards = shards[:args.max_shards]
    for shard in shards:
        if not Path(shard).is_file():
            raise FileNotFoundError(shard)
    selected_shards = set(shards)
    chosen = [r for r in rows if r['tfrecord_path'] in selected_shards]
    config = dict(version=1, source_dir=str(source), output_dir=str(output),
        source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        selected_shards=len(shards), expected_files=len(chosen),
        expected_split_counts=dict(Counter(r['split'] for r in chosen)),
        max_shards=args.max_shards, original_feature_storage='verbatim ZIP copy, append agent_lengths.npy',
        length_source='state/current/length; selected with agent_src_indices and verified with state/id')
    return output, fields, chosen, [(s,by_shard[s]) for s in shards], config


def run(args):
    if args.workers < 1 or args.max_shards < 0:
        raise ValueError('workers must be positive and max_shards nonnegative')
    output, fields, rows, jobs, config = plan(args)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.backfill.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output/'enrichment_config.json'
        if config_path.exists():
            if not args.resume or json.loads(config_path.read_text()) != config:
                raise ValueError('Existing output requires --resume with identical source/configuration')
        else:
            if any(p.name != '.backfill.lock' for p in output.iterdir()):
                raise ValueError('New output directory is not empty')
            atomic_json(config_path, config)
        started = time.time()
        status = dict(status='running', pid=os.getpid(), started_unix=started, workers=args.workers,
                      expected_files=len(rows), expected_shards=len(jobs), completed_shards=0,
                      written=0, verified_existing=0, elapsed_seconds=0.)
        atomic_json(output/'enrichment_status.json', status)
        print(json.dumps(dict(event='start', **config, workers=args.workers)), flush=True)
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(process_shard, shard, samples, str(output), args.resume):shard
                           for shard,samples in jobs}
                for future in as_completed(pending):
                    try:
                        result = future.result()
                    except BaseException:
                        for queued in pending:
                            queued.cancel()
                        raise
                    for key,value in result.items():status[key] += value
                    status['completed_shards'] += 1
                    status['elapsed_seconds'] = round(time.time()-started, 2)
                    atomic_json(output/'enrichment_status.json', status)
                    print(json.dumps(dict(event='progress', shard=pending[future], **status)), flush=True)
            # Publish dataset manifest only once every source row succeeded.
            manifest_tmp = output/'manifest.csv.tmp'
            with manifest_tmp.open('w', newline='') as f:
                extra = [] if 'source_npz_path' in fields else ['source_npz_path']
                writer = csv.DictWriter(f,fieldnames=fields+extra)
                writer.writeheader()
                for row in rows:
                    new = dict(row,source_npz_path=row['npz_path'],
                               npz_path=str(output/row['split']/Path(row['npz_path']).name))
                    writer.writerow(new)
            os.replace(manifest_tmp,output/'manifest.csv')
            actual_counts = {split:len(list((output/split).glob('*.npz'))) for split in config['expected_split_counts']}
            if actual_counts != config['expected_split_counts']:
                raise RuntimeError(f'Final split membership count mismatch: {actual_counts}')
            status.update(status='complete',elapsed_seconds=round(time.time()-started,2),split_counts=actual_counts)
            atomic_json(output/'enrichment_status.json',status)
            print(json.dumps(dict(event='complete',**status)),flush=True)
        except BaseException as exc:
            status.update(status='failed',error=repr(exc),elapsed_seconds=round(time.time()-started,2))
            atomic_json(output/'enrichment_status.json',status)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max_shards',type=int,default=0,help='Smoke test only: limit to first K shards; 0 = full dataset')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
