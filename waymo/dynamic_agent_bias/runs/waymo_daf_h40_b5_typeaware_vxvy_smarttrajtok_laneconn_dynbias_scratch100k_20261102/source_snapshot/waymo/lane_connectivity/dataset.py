"""Independent graph overlays over the unchanged baseline map cache."""
from pathlib import Path
import numpy as np
import torch
from waymo.adaptations.map_dataset import MapAdaptationDataset,collate_maps


class ConnectivityDataset(MapAdaptationDataset):
    def __init__(self,roots,cache_root,topology_root):
        super().__init__(roots,cache_root);self.topology_root=Path(topology_root)

    def __getitem__(self,index):
        item=super().__getitem__(index);source=Path(self.paths[index])
        path=self.topology_root/source.parent.name/source.name
        with np.load(path,allow_pickle=False) as d:
            if int(d['topology_version'])!=1 or str(d['scenario_id'])!=item['scenario_id']:
                raise ValueError('Topology cache identity mismatch')
            if not np.array_equal(d['map_ids'],item['map_ids'].numpy()):raise ValueError('Topology/map segment order mismatch')
            for key in ['lane_edges','lane_edge_types']:item[key]=torch.from_numpy(d[key].copy()).long()
        return item


def collate_connectivity(items):
    base=collate_maps([{k:v for k,v in item.items() if k not in ['lane_edges','lane_edge_types']} for item in items])
    m=base['map_ids'].shape[1];e=max(item['lane_edges'].shape[1] for item in items)
    for key in ['lane_edges','lane_edge_types']:
        out=items[0][key].new_full((len(items),m,e),-1 if key=='lane_edges' else 0)
        for i,item in enumerate(items):
            values=item[key];out[i,:values.shape[0],:values.shape[1]]=values
        base[key]=out
    return base
