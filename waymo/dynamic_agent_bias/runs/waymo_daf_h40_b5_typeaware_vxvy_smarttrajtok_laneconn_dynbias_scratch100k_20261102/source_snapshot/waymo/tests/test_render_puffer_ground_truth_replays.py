import inspect
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from waymo.evaluation.puffer_video_export import load_subset_records
from waymo.evaluation import render_puffer_ground_truth_replays as replay
from waymo.puffer_renderer_bridge import ScenarioManifest


class RenderPufferGroundTruthReplaysTest(unittest.TestCase):
    def test_module_has_no_world_model_or_torch_imports(self):
        source = Path(inspect.getsourcefile(replay)).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("interactive_world_model_game", source)
        self.assertNotIn("train_waymo_world_model", source)
        self.assertNotIn("build_ego_action_features", source)

    def test_isolated_import_does_not_load_torch(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import waymo.evaluation.render_puffer_ground_truth_replays; "
                    "raise SystemExit(1 if 'torch' in sys.modules else 0)"
                ),
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_parser_exposes_only_renderer_replay_controls(self):
        parser = replay.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("device", destinations)
        self.assertNotIn("checkpoint_profile", destinations)
        self.assertNotIn("context_frames", destinations)
        self.assertNotIn("max_steps", destinations)
        args = parser.parse_args(["--output-dir", "/tmp/ground_truth"])
        self.assertEqual(args.sample_start, 0)
        self.assertEqual(args.num_scenes, 5)
        self.assertEqual(args.fps, 10.0)

    def test_same_fixed_first_five_are_selected(self):
        records = load_subset_records(
            replay.DEFAULT_SUBSET_MANIFEST,
            sample_start=0,
            num_scenes=5,
        )
        self.assertEqual(
            [int(row["dataset_index"]) for row in records],
            [3155, 3445, 331, 2121, 4188],
        )

    def test_real_first_scene_loads_all_91_frames_with_per_frame_z(self):
        record = load_subset_records(
            replay.DEFAULT_SUBSET_MANIFEST,
            sample_start=0,
            num_scenes=1,
        )[0]
        manifest = ScenarioManifest(replay.DEFAULT_PUFFER_MANIFEST)
        scene = replay.load_ground_truth_scene(record, manifest)
        self.assertEqual(scene.frames.count, 91)
        self.assertEqual(scene.frames.xy.shape, (91, 32, 2))
        self.assertEqual(scene.frames.z.shape, (91, 32))
        self.assertTrue(scene.frames.valid[:, 0].all())
        self.assertGreater(float(np.ptp(scene.frames.z[:, 0])), 0.0)

        request0 = replay.build_puffer_frame(scene, 0).as_request(scene.puffer_scene)
        request90 = replay.build_puffer_frame(scene, 90).as_request(scene.puffer_scene)
        self.assertEqual(request0["source_time_index"], 0)
        self.assertEqual(request90["source_time_index"], 90)
        self.assertFalse(request0["preserve_scene_z"])
        self.assertEqual(len(request0["z"]), int(scene.agent_mask.sum()))

    def test_output_name_marks_full_ground_truth_replay(self):
        record = {
            "sample_order": 0,
            "dataset_index": 3155,
            "scenario_id": "a9a7480e5c110232",
            "focus_track_id": 1,
        }
        self.assertEqual(
            replay.output_video_name(record, total_frames=91),
            "sample_000_scene_3155_a9a7480e5c110232_focus_1_ground_truth_91f.mp4",
        )


if __name__ == "__main__":
    unittest.main()
