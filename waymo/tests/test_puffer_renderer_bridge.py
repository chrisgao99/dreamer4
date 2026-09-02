import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from waymo.puffer_renderer_bridge import (
    PufferFrameState,
    PufferRendererClient,
    PufferSceneReference,
    ScenarioManifest,
    local_to_world_pose,
)


class LocalToWorldPoseTest(unittest.TestCase):
    def test_rotates_and_translates_pose_and_velocity(self):
        xy = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        yaw = np.asarray([0.0, math.pi], dtype=np.float32)
        velocity = np.asarray([[3.0, 0.0], [0.0, -4.0]], dtype=np.float32)

        world_xy, world_yaw, world_velocity = local_to_world_pose(
            xy,
            yaw,
            velocity,
            origin_xy=np.asarray([10.0, 20.0]),
            origin_heading=math.pi / 2,
        )

        np.testing.assert_allclose(world_xy, [[10.0, 21.0], [8.0, 20.0]], atol=1e-6)
        np.testing.assert_allclose(world_velocity, [[0.0, 3.0], [4.0, 0.0]], atol=1e-6)
        np.testing.assert_allclose(world_yaw, [math.pi / 2, -math.pi / 2], atol=1e-6)

    def test_rejects_incompatible_shapes(self):
        with self.assertRaisesRegex(ValueError, "velocity_xy"):
            local_to_world_pose(
                np.zeros((2, 2)),
                np.zeros(2),
                np.zeros((1, 2)),
                origin_xy=np.zeros(2),
                origin_heading=0.0,
            )


class ScenarioManifestTest(unittest.TestCase):
    def test_exact_npz_view_resolves_preconverted_puffer_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            view_a = root / "views/scene__focus_11"
            view_b = root / "views/scene__focus_22"
            view_a.mkdir(parents=True)
            view_b.mkdir(parents=True)
            bin_a = view_a / "map_000.bin"
            bin_b = view_b / "map_000.bin"
            bin_a.write_bytes(b"a")
            bin_b.write_bytes(b"b")
            npz_a = root / "scene_focus_11.npz"
            npz_b = root / "scene_focus_22.npz"
            npz_a.write_bytes(b"")
            npz_b.write_bytes(b"")
            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "scenario_id",
                        "focus_track_id",
                        "npz_path",
                        "puffer_map_dir",
                        "puffer_bin_path",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario_id": "scene",
                        "focus_track_id": 11,
                        "npz_path": npz_a,
                        "puffer_map_dir": view_a,
                        "puffer_bin_path": bin_a,
                    }
                )
                writer.writerow(
                    {
                        "scenario_id": "scene",
                        "focus_track_id": 22,
                        "npz_path": npz_b,
                        "puffer_map_dir": view_b,
                        "puffer_bin_path": bin_b,
                    }
                )

            manifest = ScenarioManifest(manifest_path)
            resolved = manifest.resolve(
                scenario_id="scene", npz_path=npz_b, focus_track_id=22
            )
            self.assertEqual(resolved.puffer_map_dir, view_b)
            self.assertEqual(resolved.puffer_bin_path, bin_b)
            self.assertEqual(resolved.focus_track_id, 22)
            self.assertEqual(
                manifest.mapped_npz_paths,
                frozenset({str(npz_a.resolve()), str(npz_b.resolve())}),
            )
            with self.assertRaisesRegex(ValueError, "scenario mismatch"):
                manifest.resolve(
                    scenario_id="wrong-scene", npz_path=npz_b, focus_track_id=22
                )

    def test_raw_scenario_cache_is_diagnostic_not_a_render_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "abc.pb").write_bytes(b"raw scenario")
            manifest = ScenarioManifest(scenario_cache_dir=root)
            with self.assertRaisesRegex(FileNotFoundError, "preconverted Puffer map"):
                manifest.resolve(
                    scenario_id="abc",
                    npz_path=root / "abc_focus_1.npz",
                    focus_track_id=1,
                )


class PufferFrameStateTest(unittest.TestCase):
    def test_nonfinite_agent_is_invalidated_without_nonstandard_json_numbers(self):
        frame = PufferFrameState(
            step=3,
            agent_ids=np.asarray([10, 20]),
            agent_types=np.asarray([1, 2]),
            xy=np.asarray([[1.0, 2.0], [np.nan, 4.0]]),
            yaw=np.asarray([0.5, 0.25]),
            velocity_xy=np.asarray([[3.0, 4.0], [5.0, 6.0]]),
            valid=np.asarray([True, True]),
        )
        scene = PufferSceneReference(
            scenario_id="abc",
            npz_path=Path("abc.npz"),
            focus_track_id=10,
            puffer_map_dir=Path("view"),
        )
        request = frame.as_request(scene)
        self.assertEqual(request["valid"], [True, False])
        self.assertEqual(request["x"], [1.0, 0.0])
        self.assertTrue(request["preserve_scene_z"])

    def test_long_interactive_step_is_preserved_as_diagnostic_metadata(self):
        frame = PufferFrameState(
            step=160,
            agent_ids=np.asarray([10]),
            agent_types=np.asarray([1]),
            xy=np.asarray([[1.0, 2.0]]),
            yaw=np.asarray([0.5]),
            velocity_xy=np.asarray([[3.0, 4.0]]),
            valid=np.asarray([True]),
            source_time_index=160,
        )
        scene = PufferSceneReference(
            scenario_id="abc",
            npz_path=Path("abc.npz"),
            focus_track_id=10,
            puffer_map_dir=Path("view"),
        )
        request = frame.as_request(scene)
        self.assertEqual(request["step"], 160)
        self.assertEqual(request["source_time_index"], 160)

    def test_explicit_ground_truth_z_is_sent_instead_of_preserving_logged_z(self):
        frame = PufferFrameState(
            step=4,
            agent_ids=np.asarray([10, 20]),
            agent_types=np.asarray([1, 1]),
            xy=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            z=np.asarray([8.5, 9.25]),
            yaw=np.asarray([0.5, 0.25]),
            velocity_xy=np.asarray([[3.0, 4.0], [5.0, 6.0]]),
            valid=np.asarray([True, True]),
        )
        scene = PufferSceneReference(
            scenario_id="abc",
            npz_path=Path("abc.npz"),
            focus_track_id=10,
            puffer_map_dir=Path("view"),
        )
        request = frame.as_request(scene)
        self.assertFalse(request["preserve_scene_z"])
        self.assertEqual(request["z"], [8.5, 9.25])

    def test_explicit_z_must_match_agent_count(self):
        frame = PufferFrameState(
            step=0,
            agent_ids=np.asarray([10, 20]),
            agent_types=np.asarray([1, 1]),
            xy=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            z=np.asarray([8.5]),
            yaw=np.asarray([0.5, 0.25]),
            velocity_xy=np.asarray([[3.0, 4.0], [5.0, 6.0]]),
            valid=np.asarray([True, True]),
        )
        scene = PufferSceneReference(
            scenario_id="abc",
            npz_path=Path("abc.npz"),
            focus_track_id=10,
            puffer_map_dir=Path("view"),
        )
        with self.assertRaisesRegex(ValueError, "z must match agent_ids"):
            frame.as_request(scene)

    def test_nonfinite_explicit_z_invalidates_agent_and_serializes_zero(self):
        frame = PufferFrameState(
            step=0,
            agent_ids=np.asarray([10, 20]),
            agent_types=np.asarray([1, 1]),
            xy=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            z=np.asarray([8.5, np.nan]),
            yaw=np.asarray([0.5, 0.25]),
            velocity_xy=np.asarray([[3.0, 4.0], [5.0, 6.0]]),
            valid=np.asarray([True, True]),
        )
        scene = PufferSceneReference(
            scenario_id="abc",
            npz_path=Path("abc.npz"),
            focus_track_id=10,
            puffer_map_dir=Path("view"),
        )
        request = frame.as_request(scene)
        self.assertEqual(request["valid"], [True, False])
        self.assertEqual(request["z"], [8.5, 0.0])


class PufferRendererClientTest(unittest.TestCase):
    def test_length_prefixed_scene_ack_and_jpeg_response(self):
        worker_source = r'''
import json
import os
import struct
import sys

header = struct.Struct("<I")
while True:
    raw_size = sys.stdin.buffer.read(4)
    if not raw_size:
        break
    (size,) = header.unpack(raw_size)
    request = json.loads(sys.stdin.buffer.read(size))
    if request["type"] == "load_scene":
        response = (b'{"ok":true}' if os.environ.get("PUFFER_TEST_DISPLAY") == "yes"
                    else b'{"ok":false,"error":"missing environment"}')
    elif request["type"] == "render_frame":
        response = b"\xff\xd8\xff\xd9"
    elif request["type"] == "close":
        response = b'{"ok":true}'
    else:
        response = b'{"ok":false,"error":"unexpected request"}'
    sys.stdout.buffer.write(header.pack(len(response)) + response)
    sys.stdout.buffer.flush()
    if request["type"] == "close":
        break
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_dir = root / "view"
            map_dir.mkdir()
            scene = PufferSceneReference(
                scenario_id="abc",
                npz_path=root / "abc.npz",
                focus_track_id=10,
                puffer_map_dir=map_dir,
            )
            frame = PufferFrameState(
                step=0,
                agent_ids=np.asarray([10]),
                agent_types=np.asarray([1]),
                xy=np.asarray([[1.0, 2.0]]),
                yaw=np.asarray([0.5]),
                velocity_xy=np.asarray([[3.0, 4.0]]),
                valid=np.asarray([True]),
            )
            client = PufferRendererClient(
                [sys.executable, "-u", "-c", worker_source],
                width=320,
                height=180,
                timeout_s=2.0,
                environment={"PUFFER_TEST_DISPLAY": "yes"},
            )
            try:
                self.assertEqual(client.render(scene, frame), b"\xff\xd8\xff\xd9")
                self.assertIsNotNone(client.pid)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
