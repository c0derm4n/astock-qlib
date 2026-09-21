"""管道异常、真实子进程日志和指标落盘的离线回归。"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from backend import pipeline
from backend.artifacts import write_json_atomic


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(pipeline.config, "OUTPUT_DIR", self.root))
        self.stack.enter_context(patch.object(pipeline, "_state", pipeline._State()))
        self.stack.enter_context(patch.object(pipeline, "LOG_DIR", self.root))
        self.stack.enter_context(patch.object(pipeline, "METRICS_FILE", self.root / "metrics.json"))
        self.stack.enter_context(patch.object(pipeline.data_service, "read_run_meta", return_value={}))

    def run_worker(self, steps):
        pipeline._state.running = True
        pipeline._state.step_status = {step: "pending" for step in steps}
        pipeline._worker(steps)
        state = pipeline.status()
        self.assertFalse(state["running"])
        self.assertIsNone(state["current_step"])
        self.assertIsNotNone(state["finished_at"])
        return state

    def test_launch_exception_marks_failure_and_skips_remaining_steps(self):
        with patch.object(pipeline, "_run_step", side_effect=FileNotFoundError("missing python")):
            state = self.run_worker(["fetch", "train"])
        self.assertEqual(state["steps"], {"fetch": "failed", "train": "skipped"})
        self.assertIn("missing python", state["error"])
        self.assertIn("FileNotFoundError", "\n".join(state["log_tail"]))

    def test_failed_step_stops_pipeline(self):
        with patch.object(pipeline, "_run_step", return_value=False) as run:
            state = self.run_worker(["fetch", "decide"])
        run.assert_called_once_with("fetch")
        self.assertEqual(state["steps"], {"fetch": "failed", "decide": "skipped"})

    def test_train_metrics_survive_later_decision_failure(self):
        (self.root / "train.log").write_text("IC均值: 0.04\nRankICIR: nan\n", encoding="utf-8")
        with patch.object(pipeline, "_run_step", side_effect=[True, False]):
            state = self.run_worker(["train", "decide"])
        self.assertEqual(state["steps"], {"train": "done", "decide": "failed"})
        self.assertEqual(pipeline.read_metrics()["ic"], {"IC均值": 0.04, "RankICIR": None})

    def test_metric_write_failure_is_visible(self):
        with patch.object(pipeline, "_run_step", return_value=True), patch.object(
            pipeline, "_save_metrics", side_effect=OSError("disk full")
        ):
            state = self.run_worker(["train", "decide"])
        self.assertEqual(state["steps"], {"train": "failed", "decide": "skipped"})
        self.assertIn("disk full", state["error"])

    def test_thread_start_failure_releases_running_flag(self):
        with patch.object(pipeline.threading.Thread, "start", side_effect=RuntimeError("no thread")):
            with self.assertRaisesRegex(RuntimeError, "无法启动"):
                pipeline.start(["fetch"])
        state = pipeline.status()
        self.assertFalse(state["running"])
        self.assertEqual(state["steps"], {"fetch": "skipped"})
        self.assertIsNotNone(state["finished_at"])

    def test_running_pipeline_rejects_duplicate_launch(self):
        pipeline._state.running = True
        with patch.object(pipeline.threading, "Thread") as thread:
            with self.assertRaises(RuntimeError):
                pipeline.start(["fetch"])
            thread.assert_not_called()

    def test_start_deduplicates_and_orders_steps(self):
        with patch.object(pipeline.threading, "Thread") as thread:
            self.assertEqual(pipeline.start(["train", "fetch", "train"]), ["fetch", "train"])
        thread.assert_called_once_with(target=pipeline._worker, args=(["fetch", "train"],), daemon=True)

    def test_real_subprocess_captures_utf8_unbuffered_output(self):
        script = self.root / "child.py"
        script.write_text(
            "import os\nprint('中文日志')\nprint(os.environ['PYTHONUNBUFFERED'])\n", encoding="utf-8"
        )
        with patch.dict(pipeline.STEPS, {"fetch": ("测试", [sys.executable, str(script)])}):
            self.assertTrue(pipeline._run_step("fetch"))
        self.assertEqual(pipeline.get_step_log("fetch"), "中文日志\n1")

    def test_real_subprocess_nonzero_exit(self):
        with patch.dict(pipeline.STEPS, {"fetch": ("测试", [sys.executable, "-c", "raise SystemExit(3)"])}):
            self.assertFalse(pipeline._run_step("fetch"))

    def test_large_log_tail_reads_only_needed_blocks(self):
        class CountingStream(io.BytesIO):
            bytes_read = 0

            def read(self, size=-1):
                chunk = super().read(size)
                self.bytes_read += len(chunk)
                return chunk

        lines = [f"训练中文日志 {i}" for i in range(50000)]
        stream = CountingStream(("\r\n".join(lines) + "\r\n").encode("utf-8"))
        with patch.object(Path, "open", return_value=stream):
            self.assertEqual(pipeline.get_step_log("train", 5), "\n".join(lines[-5:]))
        self.assertLessEqual(stream.bytes_read, 8192)

    def test_log_tail_boundaries(self):
        for content in ("", "中文末行", "首行\n末行", "首行\n末行\n", "首行\r\n末行\r\n"):
            with self.subTest(content=content):
                (self.root / "train.log").write_bytes(content.encode("utf-8"))
                self.assertEqual(pipeline.get_step_log("train", 1), "\n".join(content.splitlines()[-1:]))
        self.assertIsNone(pipeline.get_step_log("unknown"))
        for tail in (0, -1, 10001):
            with self.assertRaises(ValueError):
                pipeline.get_step_log("train", tail)

    def test_running_or_failed_train_does_not_publish_partial_metrics(self):
        (self.root / "train.log").write_text("IC均值: 0.5\n", encoding="utf-8")
        pipeline._state.running = True
        self.assertIsNone(pipeline.read_metrics())
        pipeline._state.running = False
        pipeline._state.step_status = {"train": "failed"}
        self.assertIsNone(pipeline.read_metrics())
        self.assertFalse(pipeline.METRICS_FILE.exists())

    def test_invalid_metrics_are_tolerated_and_nonfinite_values_are_null(self):
        pipeline.METRICS_FILE.write_text('{"ic":{"ICIR":NaN,"RankICIR":Infinity}}', encoding="utf-8")
        self.assertEqual(pipeline.read_metrics()["ic"], {"ICIR": None, "RankICIR": None})
        pipeline.METRICS_FILE.write_text("{", encoding="utf-8")
        self.assertIsNone(pipeline.read_metrics())

    def test_atomic_write_failure_preserves_previous_metrics(self):
        path = pipeline.METRICS_FILE
        path.write_text('{"version":1}', encoding="utf-8")
        with patch("backend.artifacts.os.replace", side_effect=OSError("busy")):
            with self.assertRaises(OSError):
                write_json_atomic(path, {"version": 2})
        self.assertEqual(json.loads(path.read_text()), {"version": 1})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_atomic_write_cleans_nonfinite_metrics(self):
        write_json_atomic(pipeline.METRICS_FILE, {"ic": {"value": float("inf")}})
        self.assertEqual(pipeline.METRICS_FILE.read_text(encoding="utf-8").count("null"), 1)


if __name__ == "__main__":
    unittest.main()
