from pathlib import Path
import numpy as np
import torch
from waymo.lane_connectivity.dataset import ConnectivityDataset,collate_connectivity
FIELDS=('lane_half_widths','lane_width_mask','agent_widths','agent_width_mask')


class LaneLossDataset(ConnectivityDataset):
    def __init__(self,roots,map_root,topology_root,geometry_root):
        super().__init__(roots,map_root,topology_root);self.geometry_root=Path(geometry_root)

    def __getitem__(self,index):
        item=super().__getitem__(index);source=Path(self.paths[index])
        with np.load(self.geometry_root/source.parent.name/source.name,allow_pickle=False) as d:
            if int(d['geometry_version'])!=1 or str(d['scenario_id'])!=item['scenario_id']:raise ValueError('Loss geometry identity mismatch')
            if not np.array_equal(d['map_ids'],item['map_ids'].numpy()) or not np.array_equal(d['agent_ids'],item['agent_ids'].numpy()):raise ValueError('Loss geometry IDs/order mismatch')
            for key in FIELDS:item[key]=torch.from_numpy(d[key].copy())
        return item


def collate_losses(items):
    result=collate_connectivity([{k:v for k,v in item.items() if k not in FIELDS} for item in items])
    m=result['map_ids'].shape[1]
    for key in FIELDS:
        if key.startswith('lane_'):
            shape=(len(items),m,*items[0][key].shape[1:]);out=items[0][key].new_zeros(shape)
            for i,item in enumerate(items):out[i,:item[key].shape[0]]=item[key]
            result[key]=out
        else:result[key]=torch.stack([item[key] for item in items])
    return result
