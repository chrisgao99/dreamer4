#!/usr/bin/env python3
"""Run deterministic CPU regression checks without a pytest dependency."""
import importlib
import inspect
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    torch.set_num_threads(1)
    modules=['waymo.adaptations.tests.test_map_adaptation',
             'waymo.training.world_model.tests.test_agent_velocity_input',
             'waymo.training.world_model.tests.test_direct_action_flow']
    count=0
    for name in modules:
        module=importlib.import_module(name)
        for key,test in inspect.getmembers(module,inspect.isfunction):
            if key.startswith('test_'):
                test();count+=1;print('PASS',key,flush=True)
    print(f'{count} checks passed',flush=True)

if __name__=='__main__':main()
