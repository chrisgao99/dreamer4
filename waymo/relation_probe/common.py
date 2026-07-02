"""Shared utilities for current-state relation probes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = WAYMO_ROOT / "core"
for path in (REPO_ROOT, CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from vector_tokenizer_decoder import VectorBlockCausalTokenizerDecoder
    from vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from waymo_vector_dataset import WaymoVectorDataset
except ModuleNotFoundError:
    from waymo.core.vector_tokenizer_decoder import VectorBlockCausalTokenizerDecoder
    from waymo.core.vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset


def seed_everything(seed: int) -> None:
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def json_dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def slice_context_window(batch: Dict[str, Any], context_steps: int) -> Dict[str, Any]:
    if context_steps <= 0:
        return batch
    out = dict(batch)
    k = int(batch["agent_mask"].shape[-1])
    if batch["agents"].shape[1] == k:
        out["agents"] = batch["agents"][:, :, :context_steps]
    else:
        out["agents"] = batch["agents"][:, :context_steps]
    out["lights"] = batch["lights"][:, :context_steps]
    out["light_mask"] = batch["light_mask"][:, :context_steps]
    return out


def _as_args_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, argparse.Namespace):
        return vars(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _arg(args: Dict[str, Any], name: str, default: Any) -> Any:
    return args[name] if name in args and args[name] is not None else default


def _infer_tokenizer_shapes(state: Dict[str, torch.Tensor]) -> tuple[int, int]:
    try:
        n_agents = int(state["decoder.agent_queries"].shape[0])
        n_lights = int(state["decoder.light_queries"].shape[0])
    except KeyError as exc:
        raise KeyError("Checkpoint does not look like a Waymo vector tokenizer checkpoint.") from exc
    return n_agents, n_lights


class FrozenWaymoVectorTokenizer(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def build_waymo_vector_tokenizer_from_args(args: Dict[str, Any], n_agents: int, n_lights: int) -> FrozenWaymoVectorTokenizer:
    encoder_kwargs = dict(
        d_model=int(_arg(args, "d_model", 256)),
        n_heads=int(_arg(args, "n_heads", 4)),
        depth=int(_arg(args, "depth", 4)),
        n_latents=int(_arg(args, "n_latents", 64)),
        d_bottleneck=int(_arg(args, "d_bottleneck", 64)),
        hidden_dim=int(_arg(args, "hidden_dim", 128)),
        dropout=float(_arg(args, "dropout", 0.0)),
        mlp_ratio=float(_arg(args, "mlp_ratio", 4.0)),
        time_every=int(_arg(args, "time_every", 1)),
        scale_pos_embeds=bool(_arg(args, "scale_pos_embeds", True)),
        bottleneck_output=str(_arg(args, "bottleneck_output", "tanh")),
    )
    encoder_variant = str(_arg(args, "encoder_variant", "repeat_map"))
    if encoder_variant == "repeat_map":
        encoder = VectorBlockCausalEncoder(**encoder_kwargs)
    elif encoder_variant == "static_map_query":
        encoder = VectorStaticMapQueryEncoder(
            **encoder_kwargs,
            map_depth=int(_arg(args, "map_depth", 2)),
            map_cross_every=int(_arg(args, "map_cross_every", 1)),
            map_query_tokens=str(_arg(args, "map_query_tokens", "latent_agent")),
        )
    else:
        raise ValueError(f"Unknown encoder_variant in tokenizer checkpoint: {encoder_variant!r}")

    agent_xy_loss = str(_arg(args, "agent_xy_loss", "smooth_l1"))
    decoder = VectorBlockCausalTokenizerDecoder(
        d_bottleneck=int(_arg(args, "d_bottleneck", 64)),
        d_model=int(_arg(args, "d_model", 256)),
        n_heads=int(_arg(args, "n_heads", 4)),
        depth=int(_arg(args, "decoder_depth", _arg(args, "depth", 4))),
        n_latents=int(_arg(args, "n_latents", 64)),
        n_agents=n_agents,
        n_lights=n_lights,
        dropout=float(_arg(args, "dropout", 0.0)),
        mlp_ratio=float(_arg(args, "mlp_ratio", 4.0)),
        time_every=int(_arg(args, "time_every", 1)),
        scale_pos_embeds=bool(_arg(args, "scale_pos_embeds", True)),
        use_agent_tokens=bool(_arg(args, "decoder_use_agent_tokens", False)),
        agent_token_mode=str(_arg(args, "decoder_agent_token_mode", "none")),
        predict_agent_xy_gmm=agent_xy_loss == "gmm",
    )
    return FrozenWaymoVectorTokenizer(encoder=encoder, decoder=decoder)


@torch.no_grad()
def load_frozen_waymo_vector_tokenizer(ckpt_path: str, device: torch.device) -> tuple[FrozenWaymoVectorTokenizer, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok_args = _as_args_dict(ckpt.get("args", {}))
    state = ckpt["model"]
    n_agents, n_lights = _infer_tokenizer_shapes(state)
    tokenizer = build_waymo_vector_tokenizer_from_args(tok_args, n_agents=n_agents, n_lights=n_lights)
    tokenizer.load_state_dict(state, strict=True)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    return tokenizer, tok_args


@torch.no_grad()
def encode_current_z(tokenizer: FrozenWaymoVectorTokenizer, batch: Dict[str, Any], query_step: int) -> torch.Tensor:
    out = tokenizer.encoder(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    return out.z[:, query_step]


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed_everything(info.seed)

