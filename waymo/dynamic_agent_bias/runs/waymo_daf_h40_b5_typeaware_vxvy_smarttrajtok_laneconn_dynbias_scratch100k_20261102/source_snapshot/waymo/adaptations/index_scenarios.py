#!/usr/bin/env python3
"""Index scenario_id -> shard/byte offset, resumable one JSONL per shard.

Scenario and tf.Example shards are independently shuffled. Never pair by shard
number or record index. Uses a tiny protobuf reader which skips nested payloads.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import struct
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def id_class():
    proto=descriptor_pb2.FileDescriptorProto(name='scenario_id.proto',package='index',syntax='proto2')
    msg=proto.message_type.add(name='ScenarioID')
    msg.field.add(name='scenario_id',number=5,type=9,label=1)
    pool=descriptor_pool.DescriptorPool();pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('index.ScenarioID'))


def index_shard(task):
    path,output=map(Path,task)
    path=path.resolve();stat=path.stat()
    if output.exists():
        with output.open() as f:meta=json.loads(next(f))
        if meta != dict(shard=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns):
            raise ValueError(f'Stale Scenario index: {output}')
        return str(output)
    cls=id_class();tmp=output.with_name(output.name+f'.{os.getpid()}.tmp')
    output.parent.mkdir(parents=True,exist_ok=True)
    with path.open('rb') as stream,tmp.open('w') as writer:
        writer.write(json.dumps(dict(shard=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns))+'\n')
        index=0
        while True:
            offset=stream.tell();header=stream.read(12)
            if not header:break
            if len(header)!=12:raise IOError('Truncated TFRecord header')
            length=struct.unpack('<Q',header[:8])[0];payload=stream.read(length)
            if len(payload)!=length or len(stream.read(4))!=4:raise IOError('Truncated TFRecord')
            scene=cls();scene.ParseFromString(payload)
            if not scene.scenario_id:raise ValueError(f'Not a Scenario TFRecord: {path}')
            writer.write(json.dumps(dict(scenario_id=scene.scenario_id,offset=offset,index=index))+'\n')
            index+=1
    os.replace(tmp,output)
    return str(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario_root',required=True);p.add_argument('--output_dir',required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=Path(args.scenario_root);out=Path(args.output_dir)
    # OOI dataset train/val examples originate in the raw training partition.
    shards=sorted((root/'training').glob('*.tfrecord-*'))
    if not shards:raise FileNotFoundError(root/'training')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(index_shard,[(s,out/(s.name+'.jsonl')) for s in shards]):
            print(result,flush=True)

if __name__=='__main__':main()
