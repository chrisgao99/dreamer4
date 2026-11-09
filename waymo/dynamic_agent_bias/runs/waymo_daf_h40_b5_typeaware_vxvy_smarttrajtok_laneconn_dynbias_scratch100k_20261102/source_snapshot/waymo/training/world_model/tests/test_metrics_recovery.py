import json
import shutil
import tempfile
import unittest
from pathlib import Path

from waymo.training.world_model.train_waymo_direct_action_flow import append_metrics


class MetricsRecoveryTest(unittest.TestCase):
    def test_logging_continues_after_run_directory_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run" / "metrics.jsonl"
            path.parent.mkdir()
            append_metrics(path, {"step": 20})
            append_metrics(path, {"step": 40})
            self.assertEqual(
                [json.loads(line)["step"] for line in path.read_text().splitlines()],
                [20, 40],
            )
            shutil.rmtree(path.parent)
            append_metrics(path, {"step": 60})
            self.assertEqual(json.loads(path.read_text()), {"step": 60})
