"""Shared settings: the laneconn stage-one baseline arguments plus the dynamic agent bias."""
import torch
import torch.distributed as dist

from waymo.lane_connectivity.run_stage1 import training_args

REPO = '/p/yufeng/tri30/dreamer4'
DATA_ROOT = f'{REPO}/data/waymo_vector_dataset_ooi_centered_50k_with_lengths'
MAP_CACHE_DIR = f'{REPO}/data/waymo_map_smarttrajtok_5m_v2'
TOPOLOGY_DIR = f'{REPO}/data/waymo_lane_connectivity_5m_v1'
GLOBAL_BATCH = 16  # baseline: batch 4 x grad_accum 4 on one GPU


def stage1_args(run_name):
    """Exactly the waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k run, plus the new flag."""
    args = training_args(TOPOLOGY_DIR, run_name, DATA_ROOT, MAP_CACHE_DIR)
    args.ckpt_dir = f'{REPO}/waymo/checkpoints/{run_name}'
    args.dynamic_agent_bias = True
    return args


def rank_batch_sizes(world_size, total=GLOBAL_BATCH):
    """Split one optimizer step's global batch as evenly as possible (16 on 3 GPUs: 6,5,5)."""
    return [total // world_size + (r < total % world_size) for r in range(world_size)]


def sum_gradients(params):
    """All-reduce SUM of gradients in one flat buffer; missing grads count as zero."""
    params = [p for p in params if p.requires_grad]
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        p.grad = flat[offset:offset + n].view_as(p).to(p.dtype)
        offset += n


@torch.no_grad()
def broadcast_parameters(module, src=0):
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src)


@torch.no_grad()
def max_parameter_divergence(module):
    """Max |param - rank0 param| across ranks (should stay 0)."""
    flat = torch.cat([p.detach().reshape(-1).float() for p in module.parameters()])
    reference = flat.clone()
    dist.broadcast(reference, 0)
    value = (flat - reference).abs().max()
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)
