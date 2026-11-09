import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from waymo.evaluation import render_puffer_gt_focus_rollouts as render_module
from waymo.evaluation.render_puffer_gt_focus_rollouts import (
    build_parser,
    generate_gt_focus_rollout,
    load_subset_records,
    output_video_name,
    overlay_label,
    resolve_dataset_index,
)
from waymo.training.world_model import train_waymo_world_model as wm


class RenderPufferGtFocusRolloutsTest(unittest.TestCase):
    def test_subset_selection_uses_sample_order_not_json_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = []
            for sample_order in (2, 0, 1):
                path = root / f"sample_{sample_order}.npz"
                samples.append(
                    {
                        "sample_order": sample_order,
                        "dataset_index": sample_order,
                        "scenario_id": f"scenario_{sample_order}",
                        "focus_track_id": sample_order + 10,
                        "path": str(path),
                    }
                )
            manifest = root / "subset.json"
            manifest.write_text(json.dumps({"samples": samples}))
            selected = load_subset_records(
                manifest, sample_start=0, num_scenes=2
            )
            self.assertEqual([row["sample_order"] for row in selected], [0, 1])

    def test_dataset_index_is_resolved_by_exact_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [str(Path(temp_dir) / "b.npz"), str(Path(temp_dir) / "a.npz")]
            record = {"path": paths[1], "dataset_index": 1}
            self.assertEqual(resolve_dataset_index(record, paths), 1)

    def test_output_name_records_protocol_and_identity(self):
        record = {
            "sample_order": 3,
            "dataset_index": 2121,
            "scenario_id": "77046993fcd9f0bc",
            "focus_track_id": 278,
        }
        self.assertEqual(
            output_video_name(record, context_frames=11, unroll_steps=80),
            "sample_003_scene_2121_77046993fcd9f0bc_focus_278_ctx11_u80.mp4",
        )

    def test_overlay_preserves_jpeg_dimensions(self):
        source = io.BytesIO()
        Image.new("RGB", (160, 90), (20, 30, 40)).save(source, format="JPEG")
        encoded = overlay_label(source.getvalue(), "scenario x | focus 1")
        self.assertTrue(encoded.startswith(b"\xff\xd8\xff"))
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual(image.size, (160, 90))

    def test_batch_parser_defaults_to_requested_protocol(self):
        args = build_parser().parse_args(["--output-dir", "/tmp/videos"])
        self.assertEqual(args.renderer, "puffer")
        self.assertEqual(args.context_frames, 11)
        self.assertEqual(args.max_steps, 80)
        self.assertEqual(args.num_scenes, 5)
        self.assertFalse(args.puffer_use_inherited_display)

    def test_rollout_uses_action_11_and_interactive_trailing_decode_window(self):
        total_steps, num_agents, context_frames = 91, 2, 11
        agents = torch.zeros((1, num_agents, total_steps, 8), dtype=torch.float32)
        steps = torch.arange(total_steps, dtype=torch.float32)
        agents[0, 0, :, 0] = steps
        agents[0, 0, :, 1] = 2.0 * steps
        agents[0, 0, :, 2] = 5.0
        agents[0, 0, :, 3] = 1.0
        agents[0, 0, :, 4] = 2.0
        agents[0, 0, :, 5] = 1.0
        agents[0, 0, :, 6] = 0.01 * steps
        agents[0, 0, :, 7] = 1.0
        agents[0, 1, :, 0] = 100.0 + steps
        agents[0, 1, :, 1] = 200.0 + steps
        agents[0, 1, :, 5] = 1.0
        agents[0, 1, :, 7] = 1.0
        base_batch = {
            "agents": agents,
            "agent_mask": torch.ones((1, num_agents), dtype=torch.bool),
            "lights": torch.zeros((1, total_steps, 1, 4), dtype=torch.float32),
            "light_mask": torch.ones((1, total_steps, 1), dtype=torch.bool),
        }
        model_args = SimpleNamespace(
            use_ego_actions=True,
            ego_action_source="focus",
            ego_action_normalization="raw",
            packing_factor=1,
            k_max=64,
            agent_xy_loss="smooth_l1",
            agent_xy_parameterization="absolute",
        )
        full_actions, full_masks, _ = wm.build_ego_action_features(
            base_batch, model_args
        )
        decode_calls = []

        class FakeDecoder:
            attend_map = False

            def __call__(self, z, *, agent_mask, light_mask):
                batch, window = z.shape[:2]
                continuous = torch.zeros(
                    (batch, window, num_agents, 7), dtype=torch.float32
                )
                continuous[..., 6] = 1.0
                return SimpleNamespace(
                    agent_continuous=continuous,
                    agent_valid_logits=torch.full(
                        (batch, window, num_agents), 10.0, dtype=torch.float32
                    ),
                    agent_xy_gmm=None,
                )

        def decode_batch(_state, start, count):
            decode_calls.append((int(start), int(count)))
            return {
                "agent_mask": base_batch["agent_mask"],
                "light_mask": torch.ones((1, count, 1), dtype=torch.bool),
            }

        state = SimpleNamespace(
            context_focus=[None] * context_frames,
            context_start_frame=0,
            base_batch=base_batch,
            action_history=[value for value in full_actions[0, :context_frames]],
            action_mask_history=[value for value in full_masks[0, :context_frames]],
            z_history=[torch.ones((1, 2), dtype=torch.float32)] * context_frames,
            map_tokens=None,
            map_mask=None,
            agent_mask=np.ones((num_agents,), dtype=bool),
            agent_ids=np.asarray([10, 20], dtype=np.int64),
        )
        server = SimpleNamespace(
            model_args=model_args,
            dynamics=torch.nn.Identity(),
            model_rollout_window=11,
            schedule={"unused": True},
            tokenizer=SimpleNamespace(decoder=FakeDecoder()),
            args=SimpleNamespace(valid_threshold=0.5, decode_window=32),
            _decode_batch=decode_batch,
        )
        captured = {}

        def fake_sample(*_args, **kwargs):
            captured.update(kwargs)
            return kwargs["z_gt_packed"]

        with (
            mock.patch.object(
                render_module.wm,
                "sample_autoregressive_packed_sequence",
                side_effect=fake_sample,
            ),
            mock.patch.object(
                render_module.wm,
                "unpack_spatial_to_bottleneck",
                side_effect=lambda value, **_kwargs: value,
            ),
        ):
            rollout = generate_gt_focus_rollout(
                server, state, unroll_steps=80, seed=0
            )

        self.assertEqual(rollout.count, 91)
        np.testing.assert_allclose(rollout.xy[:, 0, 0], np.arange(91))
        np.testing.assert_allclose(rollout.xy[:, 0, 1], 2 * np.arange(91))
        np.testing.assert_allclose(rollout.xy[:11, 1, 0], 100 + np.arange(11))
        np.testing.assert_allclose(rollout.xy[11:, 1], 0.0)
        torch.testing.assert_close(captured["actions"][0, 11], full_actions[0, 11])
        torch.testing.assert_close(captured["act_mask"][0, 11], full_masks[0, 11])
        self.assertTrue(bool((captured["z_gt_packed"][:, 11:] == 0).all()))
        self.assertEqual(decode_calls[0], (0, 12))
        self.assertIn((1, 32), decode_calls)
        self.assertEqual(decode_calls[-1], (59, 32))


if __name__ == "__main__":
    unittest.main()
