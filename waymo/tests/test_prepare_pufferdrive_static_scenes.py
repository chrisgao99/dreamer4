import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from waymo.data_prep.prepare_pufferdrive_static_scenes import (
    INVALID_POSITION,
    NpzView,
    ScenarioResolver,
    _write_manifest,
    build_puffer_scene,
    convert_view,
    load_npz_view,
)


def _state(value: float, *, valid: bool = True):
    return SimpleNamespace(
        center_x=value,
        center_y=value + 1.0,
        center_z=value + 2.0,
        velocity_x=value + 3.0,
        velocity_y=value + 4.0,
        heading=0.1 * value,
        valid=valid,
        width=2.0,
        length=4.5,
        height=1.6,
    )


def _track(track_id: int, object_type: int, *, first_invalid: bool = False):
    states = [_state(float(index)) for index in range(91)]
    if first_invalid:
        states[0] = _state(0.0, valid=False)
    return SimpleNamespace(id=track_id, object_type=object_type, states=states)


class _MapFeature:
    def __init__(self, feature_id: int, name: str, payload):
        self.id = feature_id
        self._name = name
        setattr(self, name, payload)

    def WhichOneof(self, _field: str):
        return self._name


def _scenario():
    lane = SimpleNamespace(
        type=2,
        polyline=[
            SimpleNamespace(x=1.0, y=2.0, z=3.0),
            SimpleNamespace(x=4.0, y=5.0, z=6.0),
        ],
    )
    return SimpleNamespace(
        scenario_id="0123456789abcdef",
        tracks=[
            _track(10, 1, first_invalid=True),
            _track(20, 2),
            _track(30, 3),
        ],
        sdc_track_index=2,
        objects_of_interest=[20, 30],
        tracks_to_predict=[
            SimpleNamespace(track_index=0, difficulty=1),
            SimpleNamespace(track_index=1, difficulty=2),
            SimpleNamespace(track_index=2, difficulty=1),
        ],
        map_features=[_MapFeature(100, "lane", lane)],
        dynamic_map_states=[],
    )


class PreparePufferdriveStaticScenesTest(unittest.TestCase):
    def test_npz_mask_and_focus_define_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view.npz"
            np.savez_compressed(
                path,
                scenario_id=np.asarray("0123456789abcdef"),
                agent_ids=np.asarray([10, 20, -1], dtype=np.int64),
                agent_mask=np.asarray([True, True, False]),
                focus_track_id=np.asarray(20, dtype=np.int64),
                ego_origin_xy=np.asarray([100.0, 200.0], dtype=np.float32),
                ego_heading=np.asarray(0.25, dtype=np.float32),
            )
            view = load_npz_view(path)

        self.assertEqual(view.agent_ids, (20, 10))
        self.assertEqual(view.focus_track_id, 20)
        self.assertEqual(view.ego_origin_xy, (100.0, 200.0))
        self.assertAlmostEqual(view.ego_heading, 0.25)

    def test_build_scene_filters_objects_and_remaps_index_metadata(self):
        view = NpzView(
            path=Path("/tmp/view.npz"),
            scenario_id="0123456789abcdef",
            focus_track_id=20,
            agent_ids=(20, 10),
            ego_origin_xy=(100.0, 200.0),
            ego_heading=0.25,
        )
        built = build_puffer_scene(_scenario(), view)
        scene = built.data

        self.assertEqual([obj["id"] for obj in scene["objects"]], [20, 10])
        self.assertEqual(built.selected_source_track_indices, (1, 0))
        self.assertEqual(scene["metadata"]["sdc_track_index"], 0)
        self.assertEqual(scene["metadata"]["source_sdc_track_index"], 2)
        self.assertEqual(scene["metadata"]["source_sdc_track_id"], 30)
        self.assertEqual(scene["metadata"]["objects_of_interest"], [20])
        self.assertEqual(
            scene["metadata"]["tracks_to_predict"],
            [
                {"track_index": 1, "difficulty": 1},
                {"track_index": 0, "difficulty": 2},
            ],
        )
        self.assertEqual(scene["objects"][0]["type"], "pedestrian")
        self.assertFalse(scene["objects"][0]["mark_as_expert"])
        self.assertEqual(scene["objects"][1]["position"][0]["x"], INVALID_POSITION)
        self.assertEqual(scene["roads"][0]["map_element_id"], 2)
        self.assertEqual(scene["roads"][0]["geometry"][0]["z"], 3.0)

    def test_resolver_falls_back_from_missing_manifest_pb_to_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz_path = root / "view.npz"
            npz_path.touch()
            cache_dir = root / "cache"
            cache_dir.mkdir()
            cached_pb = cache_dir / "0123456789abcdef.pb"
            cached_pb.write_bytes(b"cached")
            manifest = root / "eval_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["npz_path", "scenario_id", "scenario_pb_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "npz_path": str(npz_path),
                        "scenario_id": "0123456789abcdef",
                        "scenario_pb_path": str(root / "missing.pb"),
                    }
                )
            view = NpzView(
                path=npz_path.resolve(),
                scenario_id="0123456789abcdef",
                focus_track_id=20,
                agent_ids=(20, 10),
                ego_origin_xy=None,
                ego_heading=None,
            )

            resolved = ScenarioResolver(cache_dir, manifest).resolve(view)

        self.assertEqual(resolved, cached_pb.resolve())

    def test_convert_view_emits_directly_loadable_map_and_bridge_columns(self):
        view = NpzView(
            path=Path("/tmp/view.npz"),
            scenario_id="0123456789abcdef",
            focus_track_id=20,
            agent_ids=(20, 10),
            ego_origin_xy=(100.0, 200.0),
            ego_heading=0.25,
        )

        def fake_converter(json_path: str, unique_map_id: int, binary_path: str):
            self.assertEqual(unique_map_id, 0)
            with open(json_path) as handle:
                data = json.load(handle)
            self.assertEqual([obj["id"] for obj in data["objects"]], [20, 10])
            Path(binary_path).write_bytes(b"puffer-bin")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pb_path = root / "scenario.pb"
            pb_path.write_bytes(b"unused by fake loader")
            row = convert_view(
                view=view,
                scenario_pb_path=pb_path,
                views_root=root / "views",
                converter=fake_converter,
                overwrite=False,
                scenario_loader=lambda _path: _scenario(),
            )
            manifest_path = root / "manifest.csv"
            _write_manifest([row], manifest_path)
            with manifest_path.open(newline="") as handle:
                loaded_row = next(csv.DictReader(handle))

            self.assertEqual(Path(row["puffer_map_dir"]).name, row["view_key"])
            self.assertEqual(Path(row["puffer_bin_path"]).name, "map_000.bin")
            self.assertEqual(Path(row["puffer_bin_path"]).read_bytes(), b"puffer-bin")
            self.assertEqual(row["ordered_agent_ids"], "20;10")
            self.assertEqual(row["ordered_source_track_indices"], "1;0")
            self.assertEqual(
                json.loads(row["ordered_agent_id_to_puffer_index"]),
                {"20": 0, "10": 1},
            )
            self.assertIn("puffer_bin_path", loaded_row)
            self.assertIn("npz_path", loaded_row)
            self.assertIn("scenario_pb_path", loaded_row)


if __name__ == "__main__":
    unittest.main()
