import struct
import zipfile
import numpy as np
import pytest
from waymo.core.waymo_vector_filter import _get_tfexample_class
from waymo.data_prep.backfill_waymo_agent_lengths import enrich_one
from waymo.data_prep.supplement_waymo_agent_lengths import first_valid_lengths,find_missing,repair_shard


def make_example():
    example=_get_tfexample_class()()
    feat=example.features.feature
    feat['scenario/id'].bytes_list.value.append(b'synthetic')
    feat['state/id'].float_list.value.extend(np.arange(128)+100)
    for part,n in [('past',10),('current',1),('future',80)]:
        lengths=np.full((128,n),-1.,np.float32);valid=np.zeros((128,n),np.int64)
        if part=='past':lengths[3,2:]=4.3;valid[3,2:]=1
        if part=='future':
            lengths[7,9:]=2.2;valid[7,9:]=1
            lengths[7,20:]=2.9  # Use FIRST observation, not future median.
        feat[f'state/{part}/length'].float_list.value.extend(lengths.ravel())
        feat[f'state/{part}/valid'].int64_list.value.extend(valid.ravel())
    return example


def test_first_valid_measurement_not_invalid_or_later_size():
    lengths,indices=first_valid_lengths(make_example().features.feature)
    assert lengths[3]==pytest.approx(4.3) and indices[3]==2
    assert lengths[7]==pytest.approx(2.2) and indices[7]==20
    assert np.isnan(lengths[0])


def test_repair_preserves_existing_dimensions_and_feature_bytes(tmp_path):
    example=make_example();raw=example.SerializeToString();shard=tmp_path/'raw'
    shard.write_bytes(struct.pack('<Q',len(raw))+b'\0'*4+raw+b'\0'*4)
    agents=np.zeros((3,91,8),np.float32);agents[0,:,7]=1;agents[1,:,7]=3
    agents[0,:,5]=1;agents[1,20:,5]=1
    source=tmp_path/'source.npz';target=tmp_path/'enriched.npz'
    np.savez_compressed(source,scenario_id='synthetic',agents=agents,agent_mask=np.array([1,1,0],bool),
        agent_ids=np.array([103,107,-1]),agent_src_indices=np.array([3,7,-1]),map_polylines=np.arange(100))
    enrich_one(source,target,np.array([4.8,-1,0],np.float32),resume=False)
    row=dict(npz_path=str(target),source_npz_path=str(source),scenario_id='synthetic',record_index='0',tfrecord_path=str(shard))
    missing=find_missing(row);assert missing['missing_slots']==[1]
    journal=repair_shard(str(shard),[missing]);assert len(journal)==1
    assert find_missing(row) is None
    with np.load(target) as d:np.testing.assert_allclose(d['agent_lengths'],[4.8,2.2,0])
    with zipfile.ZipFile(source) as a,zipfile.ZipFile(target) as b:
        for key in a.namelist():assert a.read(key)==b.read(key)
    # Retrying a staged audit is idempotent.
    repair_shard(str(shard),[missing])
    assert find_missing(row) is None


@pytest.mark.parametrize('active_frame,should_fail',[(1,False),(10,True),(70,True)])
def test_invalid_raw_length_only_allowed_before_required_frames(tmp_path,active_frame,should_fail):
    example=make_example();feat=example.features.feature
    # Reproduce a valid observation with a negative physical length.
    part,offset=('past',active_frame) if active_frame<10 else (('current',0) if active_frame==10 else ('future',active_frame-11))
    feat[f'state/{part}/valid'].int64_list.value[offset]=1
    feat[f'state/{part}/length'].float_list.value[offset]=-.52411866
    raw=example.SerializeToString();shard=tmp_path/'raw'
    shard.write_bytes(struct.pack('<Q',len(raw))+b'\0'*4+raw+b'\0'*4)
    agents=np.zeros((1,91,8),np.float32);agents[0,:,7]=1;agents[0,active_frame,5]=1
    source=tmp_path/'source.npz';target=tmp_path/'enriched.npz'
    np.savez_compressed(source,scenario_id='synthetic',agents=agents,agent_mask=np.array([True]),
        agent_ids=np.array([100]),agent_src_indices=np.array([0]))
    enrich_one(source,target,np.array([-1],np.float32),resume=False)
    row=dict(npz_path=str(target),source_npz_path=str(source),scenario_id='synthetic',record_index='0')
    missing=find_missing(row)
    if should_fail:
        with pytest.raises(ValueError,match='active tracks'):
            repair_shard(str(shard),[missing],first_required_frame=10)
    else:
        entries=repair_shard(str(shard),[missing],first_required_frame=10)
        assert entries[0]['unresolved_inactive_tracks'][0]['track_id']==100
        assert entries[0]['first_valid_frame']==[-1]
        assert find_missing(row,first_required_frame=10) is None
        with np.load(target) as d:
            assert d['agent_lengths'][0]==-1
            assert d['agent_mask'][0]
            np.testing.assert_array_equal(d['agents'],agents)
