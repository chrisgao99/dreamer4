"""Load uncapped map sidecars while reusing the original agent NPZ files."""
from pathlib import Path
import numpy as np
import torch
from waymo.core.waymo_vector_dataset import WaymoVectorDataset

FIELDS = ('map_polylines','map_mask','map_ids','map_is_lane','map_stop_sign','map_stop_point',
          'lights','light_mask','light_ids')


class MapAdaptationDataset(WaymoVectorDataset):
    def __init__(self, roots, cache_root):
        super().__init__(roots)
        self.cache_root=Path(cache_root)

    def __getitem__(self, index):
        item=super().__getitem__(index)
        source=Path(self.paths[index]);sidecar=self.cache_root/source.parent.name/source.name
        with np.load(sidecar,allow_pickle=False) as d:
            if str(d['scenario_id']) != item['scenario_id'] or str(d['source_npz']) != str(source.resolve()):
                raise ValueError(f'Map cache source identity mismatch: {sidecar}')
            if int(d['cache_version']) != 1: raise ValueError('Unsupported map cache version')
            for key in FIELDS: item[key]=torch.from_numpy(d[key].copy())
        item['map_adaptation']=torch.tensor(True)
        return item


def collate_maps(items):
    """Pad only within each minibatch; never truncate spatial coverage."""
    output={}
    for key,value in items[0].items():
        if not torch.is_tensor(value): continue
        if key in FIELDS:
            axis=1 if key.startswith('light') else 0
            size=max(item[key].shape[axis] for item in items)
            shape=list(value.shape);shape[axis]=size
            fill=-1 if key.endswith('_ids') else 0
            padded=[]
            for item in items:
                target=value.new_full(shape,fill)
                slices=[slice(None)]*value.ndim;slices[axis]=slice(0,item[key].shape[axis])
                target[tuple(slices)]=item[key];padded.append(target)
            output[key]=torch.stack(padded)
        else:
            output[key]=torch.stack([item[key] for item in items])
    return output
