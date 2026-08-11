"""Score saved oracle-focus rollouts with Waymo's Sim Agents metrics.

This is a deliberately local variant of WOSAC.  Before scoring, the original
Scenario is reduced to the NPZ's selected agents that were valid at the WOMD
current frame.  The focus track is treated as the local SDC and follows the
logged trajectory exactly.  Thus collisions and other interactions are only
computed inside the selected current-valid set; this output must not be
reported as an official challenge submission score.
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics


CHALLENGE_TYPE = submission_specs.ChallengeType.SIM_AGENTS
CURRENT_TIME_INDEX = 10
FUTURE_STEPS = 80
NUM_ROLLOUTS = 32


def load_rollout_manifest(path: Path, max_views: int) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if max_views > 0:
        records = records[:max_views]
    if not records:
        raise ValueError(f"No rollout records in {path}")
    for record in records:
        rollout_path = Path(record["rollout_path"])
        if not rollout_path.is_file():
            raise FileNotFoundError(f"Missing generated rollout: {rollout_path}")
    return records


def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    config_dir = Path(metrics.__file__).resolve().parent
    candidates = (
        config_dir / "challenge_2025_sim_agents_config.textproto",
        config_dir / "challenge_2024_config.textproto",
    )
    for path in candidates:
        if path.is_file():
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(path.read_text(), config)
            print(f"wosac_metrics_config={path}", flush=True)
            return config
    raise FileNotFoundError(f"No Sim Agents metric config found under {config_dir}")


def load_scenario(path: Path, expected_id: str) -> scenario_pb2.Scenario:
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(path.read_bytes())
    if scenario.scenario_id != expected_id:
        raise ValueError(
            f"Scenario ID mismatch for {path}: proto={scenario.scenario_id!r}, expected={expected_id!r}"
        )
    return scenario


def build_local_scenario(
    original: scenario_pb2.Scenario,
    selected_current_valid_ids: list[int],
    focus_track_id: int,
) -> tuple[scenario_pb2.Scenario, list[Any]]:
    """Keep only selected current-valid tracks and mark all as evaluated."""
    track_by_id = {int(track.id): track for track in original.tracks}
    missing = [track_id for track_id in selected_current_valid_ids if track_id not in track_by_id]
    if missing:
        raise KeyError(f"Selected IDs missing from Scenario {original.scenario_id}: {missing}")
    if focus_track_id not in selected_current_valid_ids:
        raise ValueError(f"Focus {focus_track_id} is not selected/current-valid")

    original_ttp_difficulty: dict[int, int] = {}
    for prediction in original.tracks_to_predict:
        track_id = int(original.tracks[prediction.track_index].id)
        original_ttp_difficulty[track_id] = int(prediction.difficulty)
    # Unlike tracks_to_predict, objects_of_interest stores object IDs rather
    # than indices into Scenario.tracks.
    original_ooi_ids = {int(track_id) for track_id in original.objects_of_interest}

    local = scenario_pb2.Scenario()
    local.CopyFrom(original)
    del local.tracks[:]
    del local.tracks_to_predict[:]
    del local.objects_of_interest[:]

    ordered_tracks = []
    for track_id in selected_current_valid_ids:
        source = track_by_id[track_id]
        if len(source.states) <= CURRENT_TIME_INDEX or not source.states[CURRENT_TIME_INDEX].valid:
            raise ValueError(f"Selected track {track_id} is not Scenario-current-valid")
        copied = local.tracks.add()
        copied.CopyFrom(source)
        ordered_tracks.append(copied)

    focus_index = selected_current_valid_ids.index(focus_track_id)
    local.sdc_track_index = focus_index
    for index, track_id in enumerate(selected_current_valid_ids):
        if index != focus_index:
            prediction = local.tracks_to_predict.add()
            prediction.track_index = index
            prediction.difficulty = original_ttp_difficulty.get(track_id, 0)
        if track_id in original_ooi_ids:
            local.objects_of_interest.append(track_id)

    sim_ids = list(submission_specs.get_sim_agent_ids(local, CHALLENGE_TYPE))
    eval_ids = list(submission_specs.get_evaluation_sim_agent_ids(local, CHALLENGE_TYPE))
    if set(sim_ids) != set(selected_current_valid_ids):
        raise ValueError(f"Local sim ID mismatch: {sim_ids} vs {selected_current_valid_ids}")
    if set(eval_ids) != set(selected_current_valid_ids):
        raise ValueError(f"Local evaluation ID mismatch: {eval_ids} vs {selected_current_valid_ids}")
    return local, ordered_tracks


def build_scenario_rollouts(
    scenario: scenario_pb2.Scenario,
    center_x: np.ndarray,
    center_y: np.ndarray,
    heading: np.ndarray,
    selected_slots: list[int],
    selected_ids: list[int],
    ordered_tracks: list[Any],
) -> sim_agents_submission_pb2.ScenarioRollouts:
    expected_shape = (NUM_ROLLOUTS, FUTURE_STEPS)
    if center_x.shape[:2] != expected_shape or center_y.shape[:2] != expected_shape:
        raise ValueError(f"Expected rollout prefix shape {expected_shape}, got {center_x.shape}")
    if heading.shape != center_x.shape:
        raise ValueError(f"Heading shape {heading.shape} does not match center_x {center_x.shape}")

    bundle = sim_agents_submission_pb2.ScenarioRollouts(scenario_id=scenario.scenario_id)
    for rollout_index in range(NUM_ROLLOUTS):
        joint_scene = bundle.joint_scenes.add()
        for slot, track_id, track in zip(selected_slots, selected_ids, ordered_tracks):
            # The vector world model is planar and does not predict elevation.
            # Reuse logged z so the local metric measures its modeled x/y/yaw
            # rather than penalizing a nonexistent vertical output head.
            future_z = [float(state.center_z) for state in track.states[11:91]]
            if len(future_z) != FUTURE_STEPS:
                raise ValueError(f"Track {track_id} has {len(future_z)} future z states")
            trajectory = joint_scene.simulated_trajectories.add()
            trajectory.object_id = int(track_id)
            trajectory.center_x.extend(center_x[rollout_index, :, slot].astype(float).tolist())
            trajectory.center_y.extend(center_y[rollout_index, :, slot].astype(float).tolist())
            trajectory.center_z.extend(future_z)
            trajectory.heading.extend(heading[rollout_index, :, slot].astype(float).tolist())
    submission_specs.validate_scenario_rollouts(bundle, scenario, CHALLENGE_TYPE)
    return bundle


def proto_scalars(message: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for field, _ in message.ListFields():
        if field.name == "scenario_id":
            continue
        value = getattr(message, field.name)
        if np.isscalar(value):
            values[field.name] = float(value)
    return values


def metrics_message_from_output_record(output_record: dict[str, Any]) -> sim_agents_metrics_pb2.SimAgentMetrics:
    """Rebuild a metrics proto from a durable per-view JSONL record."""
    message = sim_agents_metrics_pb2.SimAgentMetrics(scenario_id=str(output_record["scenario_id"]))
    known_fields = {field.name for field in message.DESCRIPTOR.fields}
    for name, value in output_record["metrics"].items():
        if name not in known_fields or name == "scenario_id":
            raise ValueError(f"Unknown SimAgentMetrics field in resume record: {name!r}")
        setattr(message, name, float(value))
    return message


def _record_identity(record: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(record.get("view_index", -1)),
        str(record["scenario_id"]),
        int(record["focus_track_id"]),
    )


def load_existing_per_view_results(
    path: Path,
    manifest_records: list[dict[str, Any]],
    *,
    repair_truncated_tail: bool,
) -> list[dict[str, Any]]:
    """Load and validate a contiguous completed prefix from the per-view JSONL.

    A process abort can interrupt only the final JSON line.  When requested, a
    malformed final non-empty line is discarded and the valid prefix is
    rewritten atomically.  Any corruption before the tail remains a hard error.
    """
    if not path.exists():
        return []
    raw_lines = path.read_text().splitlines()
    parsed: list[dict[str, Any]] = []
    valid_lines: list[str] = []
    for line_index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            output_record = json.loads(line)
        except json.JSONDecodeError:
            is_last_nonempty = not any(item.strip() for item in raw_lines[line_index + 1 :])
            if not (repair_truncated_tail and is_last_nonempty):
                raise ValueError(f"Malformed resume JSONL at line {line_index + 1}: {path}")
            print(f"resume_repair=discard_truncated_tail line={line_index + 1} path={path}", flush=True)
            break
        parsed.append(output_record)
        valid_lines.append(json.dumps(output_record, sort_keys=True))

    if len(parsed) > len(manifest_records):
        raise ValueError(f"Resume file has {len(parsed)} rows but manifest selection has {len(manifest_records)}")
    for expected_index, output_record in enumerate(parsed):
        expected = _record_identity(manifest_records[expected_index])
        actual = _record_identity(output_record)
        if actual != expected:
            raise ValueError(
                f"Resume prefix mismatch at row {expected_index}: output={actual}, manifest={expected}"
            )
        # Validate that all scalar fields can reconstruct the official proto.
        metrics_message_from_output_record(output_record)

    if len(valid_lines) != sum(bool(line.strip()) for line in raw_lines):
        temporary = path.with_suffix(path.suffix + ".repair.tmp")
        temporary.write_text("".join(line + "\n" for line in valid_lines))
        os.replace(temporary, path)
    return parsed


def write_progress_json(
    path: Path,
    *,
    completed_views: int,
    total_views: int,
    new_views: int,
    scoring_device: str,
    per_view_path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "completed" if completed_views == total_views else "partial",
                "completed_views": completed_views,
                "remaining_views": total_views - completed_views,
                "total_views": total_views,
                "new_views_this_invocation": new_views,
                "scoring_device": scoring_device,
                "per_view_metrics": str(per_view_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, path)


def score_one_record(
    record: dict[str, Any],
    config_bytes: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """Worker-safe scorer for one focus-view."""
    view_start = time.perf_counter()
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    config.ParseFromString(config_bytes)
    rollout_path = Path(record["rollout_path"])
    with np.load(rollout_path, allow_pickle=False) as data:
        scenario_id = str(data["scenario_id"])
        scenario_pb_path = Path(str(data["scenario_pb_path"]))
        focus_track_id = int(data["focus_track_id"])
        agent_ids = np.asarray(data["agent_ids"], dtype=np.int64)
        current_valid = np.asarray(data["current_valid"], dtype=bool)
        center_x = np.asarray(data["center_x"], dtype=np.float32)
        center_y = np.asarray(data["center_y"], dtype=np.float32)
        heading = np.asarray(data["heading"], dtype=np.float32)

    selected_slots = [int(index) for index in np.flatnonzero(current_valid)]
    selected_ids = [int(agent_ids[index]) for index in selected_slots]
    original = load_scenario(scenario_pb_path, scenario_id)
    local_scenario, ordered_tracks = build_local_scenario(
        original,
        selected_ids,
        focus_track_id,
    )
    bundle = build_scenario_rollouts(
        local_scenario,
        center_x,
        center_y,
        heading,
        selected_slots,
        selected_ids,
        ordered_tracks,
    )
    scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
        config,
        local_scenario,
        bundle,
        CHALLENGE_TYPE,
    )
    output_record = {
        "view_index": int(record.get("view_index", -1)),
        "scenario_id": scenario_id,
        "focus_track_id": focus_track_id,
        "num_selected_current_valid": len(selected_ids),
        "view_seconds": time.perf_counter() - view_start,
        "metrics": proto_scalars(scenario_metrics),
    }
    return scenario_metrics.SerializeToString(), output_record


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tf.config.set_visible_devices([], "GPU")
    visible_gpus = tf.config.list_physical_devices("GPU")
    for gpu in visible_gpus:
        # WOSAC does not need to reserve the entire accelerator.  This is
        # especially important on shared 80 GiB A100 nodes.
        tf.config.experimental.set_memory_growth(gpu, True)
    scoring_device = "gpu" if visible_gpus and args.device in {"auto", "gpu"} else "cpu"
    if args.device == "gpu" and not visible_gpus:
        raise RuntimeError("--device gpu requested, but TensorFlow cannot see a GPU")
    if scoring_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    num_workers = int(args.num_workers)
    if num_workers <= 0:
        num_workers = 1 if scoring_device == "gpu" else 4
    if scoring_device == "gpu" and num_workers != 1:
        raise ValueError("GPU WOSAC scoring requires --num_workers 1")
    manifest_path = Path(args.rollout_manifest).resolve()
    records = load_rollout_manifest(manifest_path, int(args.max_views))
    config = load_metrics_config()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_view_path = output_dir / "per_view_wosac_metrics.jsonl"
    progress_path = output_dir / "wosac_scoring_progress.json"
    if args.resume:
        existing_results = load_existing_per_view_results(
            per_view_path,
            records,
            repair_truncated_tail=True,
        )
    else:
        if per_view_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Per-view metrics already exist: {per_view_path}. "
                "Use --resume to continue or --overwrite to replace them."
            )
        existing_results = []
    completed_before = len(existing_results)
    records_to_score = records[completed_before:]
    if int(args.max_new_views) > 0:
        records_to_score = records_to_score[: int(args.max_new_views)]
    setup_seconds = time.perf_counter() - PROCESS_START
    print(
        f"scoring_protocol=oracle_focus_local_selected_current_valid views={len(records)} "
        f"official_submission_compliant=false device={scoring_device} "
        f"visible_tf_gpus={len(visible_gpus)} workers={num_workers} setup_seconds={setup_seconds:.2f} "
        f"resume={bool(args.resume)} completed_before={completed_before} "
        f"new_views_limit={int(args.max_new_views)} new_views_planned={len(records_to_score)}",
        flush=True,
    )

    scoring_start = time.perf_counter()
    all_output_records = list(existing_results)
    config_bytes = config.SerializeToString()
    if num_workers <= 1:
        result_iterator = (
            score_one_record(record, config_bytes)
            for record in records_to_score
        )
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = [
            executor.submit(score_one_record, record, config_bytes)
            for record in records_to_score
        ]
        # Preserve manifest order so the durable JSONL is always a contiguous
        # prefix and can be resumed without ambiguity.
        result_iterator = (
            future.result()
            for future in futures
        )
    output_mode = "a" if args.resume else "w"
    with per_view_path.open(output_mode) as output_handle:
        for local_index, (_, output_record) in enumerate(result_iterator, start=1):
            completed_views = completed_before + local_index
            output_handle.write(json.dumps(output_record, sort_keys=True) + "\n")
            output_handle.flush()
            if int(args.fsync_every) > 0 and local_index % int(args.fsync_every) == 0:
                os.fsync(output_handle.fileno())
            all_output_records.append(output_record)
            elapsed = time.perf_counter() - scoring_start
            mean_worker_seconds = sum(
                float(record.get("view_seconds", 0.0)) for record in all_output_records
            ) / len(all_output_records)
            print(
                f"wosac_progress={completed_views}/{len(records)} "
                f"worker_view_seconds={output_record['view_seconds']:.2f} "
                f"invocation_seconds={elapsed:.2f} "
                f"mean_seconds_per_view={mean_worker_seconds:.2f} "
                f"remaining_minutes={mean_worker_seconds * (len(records) - completed_views) / 60.0:.1f} "
                f"metametric={output_record['metrics'].get('metametric', float('nan')):.4f}",
                flush=True,
            )
        output_handle.flush()
        if int(args.fsync_every) > 0:
            os.fsync(output_handle.fileno())
    if executor is not None:
        executor.shutdown(wait=True)

    new_views = len(all_output_records) - completed_before
    completed_views = len(all_output_records)
    write_progress_json(
        progress_path,
        completed_views=completed_views,
        total_views=len(records),
        new_views=new_views,
        scoring_device=scoring_device,
        per_view_path=per_view_path,
    )
    if completed_views < len(records):
        partial = {
            "status": "partial",
            "completed_views": completed_views,
            "remaining_views": len(records) - completed_views,
            "total_views": len(records),
            "new_views_this_invocation": new_views,
            "scoring_device": scoring_device,
            "per_view_metrics": str(per_view_path),
            "progress_json": str(progress_path),
        }
        print(
            f"wosac_partial_complete completed={completed_views}/{len(records)} "
            f"new_views={new_views} progress_json={progress_path}",
            flush=True,
        )
        return partial

    per_view_messages = [metrics_message_from_output_record(record) for record in all_output_records]
    invocation_scoring_seconds = time.perf_counter() - scoring_start
    scoring_seconds = sum(float(record.get("view_seconds", 0.0)) for record in all_output_records)
    aggregated = metrics.aggregate_scenario_metrics(per_view_messages)
    bucketed = metrics.aggregate_metrics_to_buckets(config, aggregated)
    seconds_per_view = scoring_seconds / len(records)
    projected_seconds = setup_seconds + seconds_per_view * int(args.total_views_for_eta)
    summary = {
        "protocol": "oracle_focus_local_selected_current_valid",
        "official_wosac_submission_compliant": False,
        "missing_unselected_agents_filled": False,
        "focus_trajectory": "logged_oracle",
        "vertical_position": "logged_oracle_because_model_is_planar",
        "focus_included_in_metric_average": True,
        "interactions_scope": "selected_current_valid_agents",
        "views": len(records),
        "num_workers": num_workers,
        "scoring_device": scoring_device,
        "visible_tensorflow_gpus": len(visible_gpus),
        "tensorflow_gpu_memory_growth": bool(visible_gpus),
        "total_views_for_eta": int(args.total_views_for_eta),
        "setup_seconds": setup_seconds,
        "scoring_seconds": scoring_seconds,
        "final_invocation_scoring_seconds": invocation_scoring_seconds,
        "seconds_per_view": seconds_per_view,
        "projected_total_seconds": projected_seconds,
        "projected_total_hours": projected_seconds / 3600.0,
        "aggregate_metrics": proto_scalars(aggregated),
        "bucketed_metrics": proto_scalars(bucketed),
        "rollout_manifest": str(manifest_path),
        "per_view_metrics": str(per_view_path),
    }
    output_json = Path(args.output_json or (output_dir / "wosac_summary.json"))
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"wosac_complete views={len(records)} scoring_seconds={scoring_seconds:.2f} "
        f"seconds_per_view={seconds_per_view:.2f} "
        f"projected_{args.total_views_for_eta}_hours={projected_seconds / 3600.0:.2f}",
        flush=True,
    )
    print(f"aggregate_metrics={json.dumps(summary['aggregate_metrics'], sort_keys=True)}", flush=True)
    print(f"bucketed_metrics={json.dumps(summary['bucketed_metrics'], sort_keys=True)}", flush=True)
    print(f"wrote_wosac_summary={output_json}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score local oracle-focus WOSAC rollouts.")
    parser.add_argument("--rollout_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--total_views_for_eta", type=int, default=5000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse the contiguous prefix in per_view_wosac_metrics.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-view metrics. Mutually exclusive with --resume.",
    )
    parser.add_argument(
        "--max_new_views",
        type=int,
        default=0,
        help="Score at most this many new views, then exit cleanly so TensorFlow/CUDA can reset.",
    )
    parser.add_argument(
        "--fsync_every",
        type=int,
        default=1,
        help="Fsync the durable JSONL after this many newly completed views; <=0 disables fsync.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="0 selects 1 worker on TensorFlow GPU or 4 workers on CPU.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    return parser


if __name__ == "__main__":
    argument_parser = build_parser()
    args = argument_parser.parse_args()
    if args.resume and args.overwrite:
        argument_parser.error("--resume and --overwrite are mutually exclusive")
    if args.max_new_views < 0:
        argument_parser.error("--max_new_views must be non-negative")
    score(args)
