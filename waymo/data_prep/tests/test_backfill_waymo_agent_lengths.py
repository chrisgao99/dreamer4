import csv
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from waymo.core.waymo_vector_filter import _get_tfexample_class
from waymo.data_prep.backfill_waymo_agent_lengths import (
    selected_records,aligned_lengths,enrich_one,verify_output,run,plan,
)


def fixture_dataset(tmp_path):
    root=tmp_path/'source';(root/'train').mkdir(parents=True);(root/'val').mkdir()
    shard=tmp_path/'raw.tfrecord'
    raw={'scenario/id':'scene','state/id':np.array([10.,20.,30.]),'state/type':np.array([1.,2.,3.]),
         'state/current/valid':np.ones(3,dtype=np.int64),'state/current/length':np.array([4.5,.8,2.],np.float32)}
    cls=_get_tfexample_class()
    with shard.open('wb') as f:
        for scenario in ('unused','scene'):
            example=cls()
            for name,value in dict(raw,**{'scenario/id':scenario}).items():
                if isinstance(value,str):example.features.feature[name].bytes_list.value.append(value.encode())
                elif name.endswith('/valid'):example.features.feature[name].int64_list.value.extend(value)
                else:example.features.feature[name].float_list.value.extend(value)
            payload=example.SerializeToString()
            f.write(struct.pack('<Q',len(payload))+b'\0'*4+payload+b'\0'*4)
    rows=[]
    for split,src in [('train',np.array([2,0,-1])),('val',np.array([0,2,-1]))]:
        agents=np.zeros((3,91,8),np.float32)
        agents[:2,:,5]=1;agents[:2,:,7]=raw['state/type'][src[:2],None]
        p=root/split/'sample.npz'
        np.savez_compressed(p,scenario_id='scene',agent_src_indices=src,agent_mask=np.array([1,1,0],bool),
            agent_ids=np.array([raw['state/id'][src[0]],raw['state/id'][src[1]],-1],np.int64),
            agents=agents,map_polylines=np.arange(100,dtype=np.float32).reshape(10,10),
            source_record_index=1,source_tfrecord_path=str(shard))
        rows.append(dict(split=split,scenario_id='scene',tfrecord_path=str(shard),record_index='1',npz_path=str(p)))
    with (root/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return root,shard,raw,rows


def args_for(root,tmp_path,**kw):
    return SimpleNamespace(source_dir=str(root),output_dir=str(tmp_path/'output'),workers=1,resume=False,max_shards=0,**kw)


def test_indexed_read_and_id_alignment(tmp_path):
    root,shard,raw,rows=fixture_dataset(tmp_path)
    records=list(selected_records(shard,[1]))
    assert len(records)==1 and records[0][1]['scenario/id']=='scene'
    length=aligned_lengths(rows[0]['npz_path'],rows[0],records[0][1])
    np.testing.assert_array_equal(length,np.array([2.,4.5,0.],np.float32))
    bad=dict(raw);bad['state/id']=np.array([20,10,30])
    with pytest.raises(ValueError,match='ID/source index'):
        aligned_lengths(rows[0]['npz_path'],rows[0],bad)
    with pytest.raises(ValueError,match='scenario'):
        aligned_lengths(rows[0]['npz_path'],dict(rows[0],scenario_id='wrong'),raw)


def test_copy_preserves_every_existing_npy_byte_and_input_file(tmp_path):
    _,_,raw,rows=fixture_dataset(tmp_path)
    source=Path(rows[0]['npz_path']);before=hashlib.sha256(source.read_bytes()).hexdigest()
    lengths=aligned_lengths(source,rows[0],raw);out=tmp_path/'new.npz'
    assert enrich_one(source,out,lengths,resume=False)=='written'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    with zipfile.ZipFile(source) as a,zipfile.ZipFile(out) as b:
        assert set(b.namelist())==set(a.namelist())|{'agent_lengths.npy'}
        for key in a.namelist():assert a.read(key)==b.read(key)
    assert enrich_one(source,out,lengths,resume=True)=='verified_existing'
    with pytest.raises(FileExistsError):enrich_one(source,out,lengths,resume=False)
    with pytest.raises(ValueError,match='incorrect agent_lengths'):
        verify_output(source,out,lengths+1)
    with pytest.raises(ValueError,match='original NPZ'):
        enrich_one(source,source,lengths,resume=True)


def test_complete_manifest_resume_and_source_membership(tmp_path):
    root,_,_,rows=fixture_dataset(tmp_path);args=args_for(root,tmp_path)
    run(args)
    out=Path(args.output_dir);status=json.loads((out/'enrichment_status.json').read_text())
    assert status['status']=='complete' and status['split_counts']=={'train':1,'val':1}
    with (out/'manifest.csv').open() as f: new=list(csv.DictReader(f))
    assert [r['source_npz_path'] for r in new]==[r['npz_path'] for r in rows]
    assert [r['split'] for r in new]==['train','val']
    args.resume=True;run(args)
    assert json.loads((out/'enrichment_status.json').read_text())['verified_existing']==2
    np.savez(root/'train'/'unexpected.npz',a=np.zeros(1))
    with pytest.raises(ValueError,match='membership'):plan(args)


def test_truncated_record_and_bad_length_fail(tmp_path):
    _,shard,raw,rows=fixture_dataset(tmp_path)
    with shard.open('r+b') as f:f.truncate(shard.stat().st_size-1)
    with pytest.raises(IOError,match='truncated'):list(selected_records(shard,[1]))
    raw['state/current/length'][2]=0
    with pytest.raises(ValueError,match='positive measured length'):
        aligned_lengths(rows[0]['npz_path'],rows[0],raw)


def test_failed_run_reports_failure_and_no_manifest(tmp_path):
    root,_,_,rows=fixture_dataset(tmp_path);args=args_for(root,tmp_path)
    with (root/'manifest.csv').open() as f: text=f.read()
    (root/'manifest.csv').write_text(text.replace('scene','wrong'))
    with pytest.raises(ValueError,match='scenario'):run(args)
    out=Path(args.output_dir)
    assert json.loads((out/'enrichment_status.json').read_text())['status']=='failed'
    assert not (out/'manifest.csv').exists()
