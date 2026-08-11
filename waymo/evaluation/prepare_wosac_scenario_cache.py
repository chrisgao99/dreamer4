"""Match OOI-centered NPZ validation views to raw WOMD Scenario protos.

The OOI-centered dataset was prepared from WOMD tf.Example shards.  WOSAC
metrics, however, consume the richer ``Scenario`` proto representation.  This
script uses the source shard and record index recorded in ``manifest.csv`` as
an efficient candidate lookup, then verifies the match by ``scenario_id``.

Outputs are deliberately independent of TensorFlow:

* ``scenarios/<scenario_id>.pb`` stores the original serialized Scenario.
* ``eval_manifest.csv`` has one row per focus view.
* ``summary.json`` records coverage and eligibility statistics.

The extraction is resumable.  Existing cached protos are parsed and verified
before they are reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from waymo_open_dataset.protos import scenario_pb2


CURRENT_TIME_INDEX = 10
POSITION_MATCH_TOLERANCE_METERS = 0.01
HEADING_MATCH_TOLERANCE_RADIANS = 1e-4
_TFEXAMPLE_SHARD_RE = re.compile(
    r"^(?P<split>.+)_tfexample\.tfrecord-(?P<shard>\d+)-of-(?P<count>\d+)$"
)


@dataclass(frozen=True)
class ScenarioRequest:
    scenario_id: str
    source_tfexample_path: Path
    source_tfexample_record_index: int
    source_scenario_path: Path
    source_scenario_record_index: int


def _parse_semicolon_ints(value: str) -> list[int]:
    return [int(part) for part in str(value).split(";") if part != ""]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_shard_path(tfexample_path: Path, scenario_root: Path) -> Path:
    match = _TFEXAMPLE_SHARD_RE.match(tfexample_path.name)
    if match is None:
        raise ValueError(f"Unrecognized tf.Example shard name: {tfexample_path.name}")
    filename = (
        f"{match.group('split')}.tfrecord-"
        f"{match.group('shard')}-of-{match.group('count')}"
    )
    return scenario_root / filename


def _load_view_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row.get("split") == split]
    if not rows:
        raise ValueError(f"No rows with split={split!r} in {manifest_path}")
    return rows


def _select_scenarios(
    rows: list[dict[str, str]], max_scenarios: int
) -> list[dict[str, str]]:
    if max_scenarios <= 0:
        return rows
    selected_ids: set[str] = set()
    for row in rows:
        selected_ids.add(row["scenario_id"])
        if len(selected_ids) >= max_scenarios:
            break
    return [row for row in rows if row["scenario_id"] in selected_ids]


def _load_scenario_locations(
    index_path: Path,
    wanted_ids: set[str],
    scenario_root: Path,
) -> dict[str, tuple[Path, int]]:
    with index_path.open() as handle:
        shard_ids = json.load(handle)
    locations: dict[str, tuple[Path, int]] = {}
    for shard_name, scenario_ids in shard_ids.items():
        shard_path = scenario_root / shard_name
        for record_index, scenario_id in enumerate(scenario_ids):
            if scenario_id not in wanted_ids:
                continue
            if scenario_id in locations:
                raise ValueError(f"Duplicate scenario ID in {index_path}: {scenario_id}")
            locations[scenario_id] = (shard_path, record_index)
    missing_ids = wanted_ids - set(locations)
    if missing_ids:
        raise KeyError(
            f"Scenario index {index_path} is missing {len(missing_ids)} requested IDs; "
            f"first IDs: {sorted(missing_ids)[:10]}"
        )
    return locations


def _build_requests(
    rows: Iterable[dict[str, str]],
    scenario_root: Path,
    scenario_index_path: Path,
) -> dict[str, ScenarioRequest]:
    rows = list(rows)
    wanted_ids = {row["scenario_id"] for row in rows}
    scenario_locations = _load_scenario_locations(
        scenario_index_path, wanted_ids, scenario_root
    )
    requests: dict[str, ScenarioRequest] = {}
    for row in rows:
        scenario_id = row["scenario_id"]
        tfexample_path = Path(row["tfrecord_path"])
        scenario_path, scenario_record_index = scenario_locations[scenario_id]
        request = ScenarioRequest(
            scenario_id=scenario_id,
            source_tfexample_path=tfexample_path,
            source_tfexample_record_index=int(row["record_index"]),
            source_scenario_path=scenario_path,
            source_scenario_record_index=scenario_record_index,
        )
        previous = requests.get(scenario_id)
        if previous is not None and previous != request:
            raise ValueError(
                f"Conflicting source locations for scenario {scenario_id}: "
                f"{previous} vs {request}"
            )
        requests[scenario_id] = request
    return requests


def _read_tfrecord_targets(
    path: Path, target_indices: set[int]
) -> Iterator[tuple[int, bytes]]:
    """Read selected uncompressed TFRecord payloads using seeks for other rows."""
    if not target_indices:
        return
    if min(target_indices) < 0:
        raise ValueError(f"Negative TFRecord index requested for {path}")
    max_index = max(target_indices)
    with path.open("rb") as handle:
        for record_index in range(max_index + 1):
            header = handle.read(12)
            if len(header) != 12:
                raise EOFError(
                    f"TFRecord {path} ended before requested record {max_index}; "
                    f"stopped at {record_index}"
                )
            payload_length = struct.unpack("<Q", header[:8])[0]
            if record_index in target_indices:
                payload = handle.read(payload_length)
                if len(payload) != payload_length:
                    raise EOFError(f"Truncated payload at {path}:{record_index}")
                footer = handle.read(4)
                if len(footer) != 4:
                    raise EOFError(f"Truncated footer at {path}:{record_index}")
                yield record_index, payload
            else:
                handle.seek(payload_length + 4, 1)


def _parse_scenario(payload: bytes, expected_id: str, source: str) -> scenario_pb2.Scenario:
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(payload)
    if scenario.scenario_id != expected_id:
        raise ValueError(
            f"Scenario ID mismatch at {source}: expected {expected_id}, "
            f"found {scenario.scenario_id}"
        )
    return scenario


@lru_cache(maxsize=8)
def _load_cached_scenario(path: Path, expected_id: str) -> scenario_pb2.Scenario:
    return _parse_scenario(path.read_bytes(), expected_id, str(path))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)


def _cache_scenarios(
    requests: dict[str, ScenarioRequest],
    scenario_dir: Path,
    num_workers: int,
) -> tuple[dict[str, Path], int, int]:
    cached_paths: dict[str, Path] = {}
    pending_by_shard: dict[Path, list[ScenarioRequest]] = defaultdict(list)
    reused = 0

    for scenario_id, request in requests.items():
        cache_path = scenario_dir / f"{scenario_id}.pb"
        if cache_path.exists():
            cached_paths[scenario_id] = cache_path
            reused += 1
        else:
            pending_by_shard[request.source_scenario_path].append(request)

    def extract_shard(
        item: tuple[Path, list[ScenarioRequest]],
    ) -> dict[str, Path]:
        shard_path, shard_requests = item
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing Scenario shard: {shard_path}")
        requests_by_index: dict[int, ScenarioRequest] = {}
        for request in shard_requests:
            previous = requests_by_index.get(request.source_scenario_record_index)
            if previous is not None and previous.scenario_id != request.scenario_id:
                raise ValueError(
                    f"Two scenario IDs request {shard_path}:"
                    f"{request.source_scenario_record_index}: "
                    f"{previous.scenario_id}, {request.scenario_id}"
                )
            requests_by_index[request.source_scenario_record_index] = request

        found_indices: set[int] = set()
        shard_cached_paths: dict[str, Path] = {}
        for record_index, payload in _read_tfrecord_targets(
            shard_path, set(requests_by_index)
        ):
            request = requests_by_index[record_index]
            _parse_scenario(
                payload,
                request.scenario_id,
                f"{shard_path}:{record_index}",
            )
            cache_path = scenario_dir / f"{request.scenario_id}.pb"
            _write_atomic(cache_path, payload)
            shard_cached_paths[request.scenario_id] = cache_path
            found_indices.add(record_index)

        missing_indices = set(requests_by_index) - found_indices
        if missing_indices:
            raise RuntimeError(
                f"Failed to extract indices {sorted(missing_indices)} from {shard_path}"
            )
        return shard_cached_paths

    extracted = 0
    shard_items = sorted(pending_by_shard.items(), key=lambda item: str(item[0]))
    total_shards = len(shard_items)
    workers = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for shard_number, shard_cached_paths in enumerate(
            executor.map(extract_shard, shard_items), start=1
        ):
            cached_paths.update(shard_cached_paths)
            extracted += len(shard_cached_paths)
            if (
                shard_number == 1
                or shard_number % 50 == 0
                or shard_number == total_shards
            ):
                print(
                    f"scenario extraction {shard_number}/{total_shards} shards; "
                    f"cached={len(cached_paths)}/{len(requests)}",
                    flush=True,
                )

    missing_ids = set(requests) - set(cached_paths)
    if missing_ids:
        raise RuntimeError(f"Missing cached scenarios: {sorted(missing_ids)}")
    return cached_paths, extracted, reused


def _wrapped_angle_difference(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _validate_view(
    row: dict[str, str],
    scenario_path: Path,
    request: ScenarioRequest,
) -> dict[str, object]:
    scenario = _load_cached_scenario(scenario_path, row["scenario_id"])
    tracks_by_id = {int(track.id): track for track in scenario.tracks}
    proto_ooi_ids = {int(value) for value in scenario.objects_of_interest}
    row_ooi_ids = _parse_semicolon_ints(row.get("ooi_track_ids", ""))
    focus_id = int(row["focus_track_id"])
    if focus_id not in row_ooi_ids or focus_id not in proto_ooi_ids:
        raise ValueError(
            f"{row['scenario_id']} focus {focus_id} is not in both OOI definitions: "
            f"manifest={row_ooi_ids}, proto={sorted(proto_ooi_ids)}"
        )
    if not set(row_ooi_ids).issubset(proto_ooi_ids):
        raise ValueError(
            f"{row['scenario_id']} manifest OOIs {row_ooi_ids} are not a subset of "
            f"Scenario OOIs {sorted(proto_ooi_ids)}"
        )

    npz_path = Path(row["npz_path"])
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing NPZ view: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        npz_scenario_id = str(np.asarray(data["scenario_id"]).item())
        npz_focus_id = int(np.asarray(data["focus_track_id"]).item())
        agent_ids_all = np.asarray(data["agent_ids"], dtype=np.int64).reshape(-1)
        agent_mask = np.asarray(data["agent_mask"], dtype=bool).reshape(-1)
        selected_agent_ids = [int(value) for value in agent_ids_all[agent_mask]]
        origin_xy = np.asarray(data["ego_origin_xy"], dtype=np.float64).reshape(2)
        origin_heading = float(np.asarray(data["ego_heading"]).item())

    if npz_scenario_id != row["scenario_id"]:
        raise ValueError(
            f"NPZ scenario mismatch in {npz_path}: {npz_scenario_id} vs {row['scenario_id']}"
        )
    if npz_focus_id != focus_id:
        raise ValueError(
            f"NPZ focus mismatch in {npz_path}: {npz_focus_id} vs {focus_id}"
        )
    if not selected_agent_ids or selected_agent_ids[0] != focus_id:
        raise ValueError(
            f"NPZ slot 0 is not focus {focus_id} in {npz_path}: {selected_agent_ids[:1]}"
        )
    unknown_agent_ids = set(selected_agent_ids) - set(tracks_by_id)
    if unknown_agent_ids:
        raise ValueError(
            f"NPZ contains IDs absent from Scenario {row['scenario_id']}: "
            f"{sorted(unknown_agent_ids)}"
        )

    focus_track = tracks_by_id[focus_id]
    if len(focus_track.states) <= CURRENT_TIME_INDEX:
        raise ValueError(f"Focus track {focus_id} lacks current state in {row['scenario_id']}")
    focus_current = focus_track.states[CURRENT_TIME_INDEX]
    expected_origin = np.asarray(
        [focus_current.center_x, focus_current.center_y], dtype=np.float64
    )
    origin_error_m = float(np.linalg.norm(origin_xy - expected_origin))
    if origin_error_m > POSITION_MATCH_TOLERANCE_METERS:
        raise ValueError(
            f"Focus origin mismatch for {row['scenario_id']}/{focus_id}: "
            f"NPZ={origin_xy.tolist()}, Scenario={expected_origin.tolist()}, "
            f"error_m={origin_error_m}"
        )
    heading_error_rad = _wrapped_angle_difference(
        origin_heading, float(focus_current.heading)
    )
    if heading_error_rad > HEADING_MATCH_TOLERANCE_RADIANS:
        raise ValueError(
            f"Focus heading mismatch for {row['scenario_id']}/{focus_id}: "
            f"NPZ={origin_heading}, Scenario={focus_current.heading}, "
            f"error_rad={heading_error_rad}"
        )

    partner_ids = [value for value in row_ooi_ids if value != focus_id]
    partner_id = partner_ids[0] if len(partner_ids) == 1 else -1

    def current_valid(track_id: int) -> bool:
        if track_id not in tracks_by_id:
            return False
        states = tracks_by_id[track_id].states
        return len(states) > CURRENT_TIME_INDEX and bool(states[CURRENT_TIME_INDEX].valid)

    focus_current_valid = current_valid(focus_id)
    partner_current_valid = current_valid(partner_id)
    selected_set = set(selected_agent_ids)
    partner_selected = partner_id in selected_set
    selected_current_valid = sum(current_valid(track_id) for track_id in selected_agent_ids)
    all_current_valid = sum(
        len(track.states) > CURRENT_TIME_INDEX
        and bool(track.states[CURRENT_TIME_INDEX].valid)
        for track in scenario.tracks
    )
    eligible = bool(
        len(row_ooi_ids) == 2
        and focus_current_valid
        and partner_current_valid
        and partner_selected
    )

    return {
        "split": row["split"],
        "scenario_id": row["scenario_id"],
        "focus_track_id": focus_id,
        "partner_track_id": partner_id,
        "eligible_ooi_pair": eligible,
        "focus_current_valid": focus_current_valid,
        "partner_current_valid": partner_current_valid,
        "partner_selected": partner_selected,
        "ooi_track_ids": ";".join(str(value) for value in row_ooi_ids),
        "selected_agent_ids": ";".join(str(value) for value in selected_agent_ids),
        "num_selected_slots": len(selected_agent_ids),
        "num_selected_current_valid": selected_current_valid,
        "num_all_current_valid": all_current_valid,
        "focus_origin_error_m": origin_error_m,
        "focus_heading_error_rad": heading_error_rad,
        "npz_path": str(Path(row["npz_path"]).resolve()),
        "scenario_pb_path": str(scenario_path.resolve()),
        "source_tfexample_path": str(request.source_tfexample_path),
        "source_scenario_path": str(request.source_scenario_path),
        "source_tfexample_record_index": request.source_tfexample_record_index,
        "source_scenario_record_index": request.source_scenario_record_index,
    }


def _write_csv_atomic(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty evaluation manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def prepare(args: argparse.Namespace) -> None:
    input_manifest = Path(args.input_manifest).resolve()
    scenario_root = Path(args.scenario_root).resolve()
    scenario_index_path = Path(args.scenario_index_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    scenario_dir = output_dir / "scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    rows = _select_scenarios(
        _load_view_rows(input_manifest, args.split), args.max_scenarios
    )
    requests = _build_requests(rows, scenario_root, scenario_index_path)
    print(
        f"preparing split={args.split} views={len(rows)} "
        f"unique_scenarios={len(requests)}",
        flush=True,
    )
    cached_paths, extracted, reused = _cache_scenarios(
        requests, scenario_dir, args.num_workers
    )

    output_rows = []
    for index, row in enumerate(rows, start=1):
        scenario_id = row["scenario_id"]
        output_rows.append(
            _validate_view(row, cached_paths[scenario_id], requests[scenario_id])
        )
        if index == 1 or index % 500 == 0 or index == len(rows):
            print(f"view validation {index}/{len(rows)}", flush=True)

    output_manifest = output_dir / "eval_manifest.csv"
    _write_csv_atomic(output_rows, output_manifest)
    eligible_rows = [
        row for row in output_rows if bool(row["eligible_ooi_pair"])
    ]
    eligible_output_manifest = output_dir / "eligible_ooi_pair_manifest.csv"
    _write_csv_atomic(eligible_rows, eligible_output_manifest)
    views_per_scenario = Counter(row["scenario_id"] for row in output_rows)
    eligible_scenario_ids = {
        str(row["scenario_id"])
        for row in output_rows
        if bool(row["eligible_ooi_pair"])
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": _sha256(input_manifest),
        "split": args.split,
        "scenario_root": str(scenario_root),
        "scenario_index_json": str(scenario_index_path),
        "scenario_index_sha256": _sha256(scenario_index_path),
        "output_dir": str(output_dir),
        "output_manifest": str(output_manifest),
        "eligible_ooi_pair_manifest": str(eligible_output_manifest),
        "views": len(output_rows),
        "unique_scenarios": len(requests),
        "scenario_protos_cached_total": len(cached_paths),
        "scenario_protos_extracted": extracted,
        "scenario_protos_reused": reused,
        "extraction_workers": int(args.num_workers),
        "eligible_ooi_pair_views": sum(
            bool(row["eligible_ooi_pair"]) for row in output_rows
        ),
        "eligible_ooi_pair_scenarios": len(eligible_scenario_ids),
        "focus_origin_error_m_max": max(
            float(row["focus_origin_error_m"]) for row in output_rows
        ),
        "focus_heading_error_rad_max": max(
            float(row["focus_heading_error_rad"]) for row in output_rows
        ),
        "views_per_scenario_histogram": {
            str(count): frequency
            for count, frequency in sorted(Counter(views_per_scenario.values()).items())
        },
        "selected_slots_histogram": {
            str(count): frequency
            for count, frequency in sorted(
                Counter(int(row["num_selected_slots"]) for row in output_rows).items()
            )
        },
        "selected_current_valid_histogram": {
            str(count): frequency
            for count, frequency in sorted(
                Counter(
                    int(row["num_selected_current_valid"]) for row in output_rows
                ).items()
            )
        },
    }
    summary_path = output_dir / "summary.json"
    _write_atomic(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache WOMD Scenario protos corresponding to OOI-centered NPZ views."
    )
    parser.add_argument(
        "--input_manifest",
        default=(
            "/p/yufeng/tri30/dreamer4/data/"
            "waymo_vector_dataset_ooi_centered_50k/manifest.csv"
        ),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--scenario_root",
        default="/p/liverobotics/waymo_open_dataset_motion/scenario/training",
    )
    parser.add_argument(
        "--scenario_index_json",
        default="/p/yufeng/tri30/simulation/scenario_sids.json",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/p/yufeng/tri30/dreamer4/waymo/cache/"
            "wosac_internal_val_scenarios"
        ),
    )
    parser.add_argument(
        "--max_scenarios",
        type=int,
        default=0,
        help="Limit unique scenarios for a smoke test; 0 extracts the full split.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    return parser


def main() -> None:
    prepare(build_argparser().parse_args())


if __name__ == "__main__":
    main()
