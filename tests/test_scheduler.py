import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from hf_benchmark_agent.scheduler import (
    BenchmarkScanScheduler,
    ScanResult,
    _build_scan_command,
    run_scan_subprocess,
)


class TestBuildScanCommand(unittest.TestCase):
    def test_default_args(self):
        cmd = _build_scan_command()
        self.assertIn("--read-bot", cmd)
        self.assertIn("--hours", cmd)
        self.assertEqual(cmd[cmd.index("--hours") + 1], "24")
        self.assertIn("--limit", cmd)
        self.assertEqual(cmd[cmd.index("--limit") + 1], "100")
        self.assertNotIn("--arena-base-url", cmd)

    def test_custom_args(self):
        cmd = _build_scan_command(hours=6, limit=50, arena_base_url="https://custom.ai")
        self.assertEqual(cmd[cmd.index("--hours") + 1], "6")
        self.assertEqual(cmd[cmd.index("--limit") + 1], "50")
        self.assertIn("--arena-base-url", cmd)
        self.assertEqual(cmd[cmd.index("--arena-base-url") + 1], "https://custom.ai")


class TestRunScanSubprocess(unittest.TestCase):
    @patch("hf_benchmark_agent.scheduler.subprocess.run")
    def test_successful_scan(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"processed": 2, "items": [{"chat_id": 1}, {"chat_id": 2}]}),
            stderr="",
        )
        result = run_scan_subprocess(hours=12, limit=50)
        self.assertIsNone(result.error)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.processed, 2)
        self.assertEqual(len(result.items), 2)

    @patch("hf_benchmark_agent.scheduler.subprocess.run")
    def test_nonzero_exit_code(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr='{"error": "something broke"}',
        )
        result = run_scan_subprocess()
        self.assertEqual(result.returncode, 1)
        self.assertIn("something broke", result.error)
        self.assertEqual(result.processed, 0)

    @patch("hf_benchmark_agent.scheduler.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["x"], timeout=5)
        result = run_scan_subprocess(timeout_seconds=5)
        self.assertEqual(result.returncode, -1)
        self.assertIn("timed out", result.error)

    @patch("hf_benchmark_agent.scheduler.subprocess.run")
    def test_os_error(self, mock_run):
        mock_run.side_effect = OSError("No such file")
        result = run_scan_subprocess()
        self.assertEqual(result.returncode, -1)
        self.assertIn("No such file", result.error)

    @patch("hf_benchmark_agent.scheduler.subprocess.run")
    def test_invalid_json(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not valid json",
            stderr="",
        )
        result = run_scan_subprocess()
        self.assertIn("Invalid JSON", result.error)

    @patch("hf_benchmark_agent.scheduler.subprocess.run")
    def test_empty_stdout_is_not_an_error(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )
        result = run_scan_subprocess()
        self.assertIsNone(result.error)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.processed, 0)
        self.assertEqual(result.items, [])


class TestBenchmarkScanScheduler(unittest.TestCase):
    def test_interval_minimum(self):
        with self.assertRaises(ValueError):
            BenchmarkScanScheduler(interval_seconds=5)

    def test_stop_halts_loop(self):
        scheduler = BenchmarkScanScheduler(interval_seconds=10)
        scan_results = [
            ScanResult(timestamp=0, returncode=0, processed=1, items=[], error=None),
        ]

        with patch.object(scheduler, "_run_one_scan", side_effect=scan_results):
            scheduler.stop()
            scheduler.run_forever()

        self.assertEqual(scheduler.scan_count, 0)

    @patch("hf_benchmark_agent.scheduler.run_scan_subprocess")
    def test_runs_scans_and_stops(self, mock_scan):
        mock_scan.return_value = ScanResult(
            timestamp=0, returncode=0, processed=3,
            items=[{"chat_id": 1}], error=None,
        )

        scheduler = BenchmarkScanScheduler(interval_seconds=10)

        def stop_after_one():
            import time
            while scheduler.scan_count < 1:
                time.sleep(0.05)
            scheduler.stop()

        stopper = threading.Thread(target=stop_after_one)
        stopper.start()
        scheduler.run_forever()
        stopper.join(timeout=5)

        self.assertEqual(scheduler.scan_count, 1)
        self.assertEqual(len(scheduler.history), 1)
        self.assertEqual(scheduler.history[0].processed, 3)
        self.assertIsNone(scheduler.history[0].error)

    @patch("hf_benchmark_agent.scheduler.run_scan_subprocess")
    def test_records_failed_scans(self, mock_scan):
        mock_scan.return_value = ScanResult(
            timestamp=0, returncode=1, processed=0,
            items=[], error="connection refused",
        )

        scheduler = BenchmarkScanScheduler(interval_seconds=10)

        def stop_after_one():
            import time
            while scheduler.scan_count < 1:
                time.sleep(0.05)
            scheduler.stop()

        stopper = threading.Thread(target=stop_after_one)
        stopper.start()
        scheduler.run_forever()
        stopper.join(timeout=5)

        self.assertEqual(scheduler.scan_count, 1)
        self.assertEqual(scheduler.history[0].error, "connection refused")

    def test_scheduler_passes_config_to_subprocess(self):
        scheduler = BenchmarkScanScheduler(
            interval_seconds=60,
            hours=6,
            limit=50,
            arena_base_url="https://custom.ai",
            subprocess_timeout=30.0,
        )
        self.assertEqual(scheduler.hours, 6)
        self.assertEqual(scheduler.limit, 50)
        self.assertEqual(scheduler.arena_base_url, "https://custom.ai")
        self.assertEqual(scheduler.subprocess_timeout, 30.0)

    @patch("hf_benchmark_agent.scheduler.run_scan_subprocess")
    def test_scheduler_tracks_answered_ids_across_scans(self, mock_scan):
        call_count = [0]

        def fake_scan(**kwargs):
            call_count[0] += 1
            answered = kwargs.get("answered_update_ids", set())
            if call_count[0] == 1:
                assert not answered
                return ScanResult(
                    timestamp=0, returncode=0, processed=2,
                    items=[
                        {"chat_id": 1, "update_id": 100},
                        {"chat_id": 2, "update_id": 101},
                    ],
                    error=None,
                )
            else:
                assert 100 in answered
                assert 101 in answered
                return ScanResult(
                    timestamp=0, returncode=0, processed=1,
                    items=[{"chat_id": 3, "update_id": 102}],
                    error=None,
                )

        mock_scan.side_effect = fake_scan

        scheduler = BenchmarkScanScheduler(interval_seconds=10)

        def stop_after_two():
            import time
            while scheduler.scan_count < 2:
                time.sleep(0.05)
            scheduler.stop()

        stopper = threading.Thread(target=stop_after_two)
        stopper.start()
        scheduler.run_forever()
        stopper.join(timeout=5)

        self.assertEqual(scheduler.scan_count, 2)
        self.assertEqual(scheduler.answered_update_ids, {100, 101, 102})

    @patch("hf_benchmark_agent.scheduler.run_scan_subprocess")
    def test_failed_scan_does_not_add_answered_ids(self, mock_scan):
        mock_scan.return_value = ScanResult(
            timestamp=0, returncode=1, processed=0, items=[], error="network error",
        )

        scheduler = BenchmarkScanScheduler(interval_seconds=10)

        def stop_after_one():
            import time
            while scheduler.scan_count < 1:
                time.sleep(0.05)
            scheduler.stop()

        stopper = threading.Thread(target=stop_after_one)
        stopper.start()
        scheduler.run_forever()
        stopper.join(timeout=5)

        self.assertEqual(scheduler.answered_update_ids, set())


class TestBuildScanCommandAnsweredIds(unittest.TestCase):
    def test_answered_ids_included_in_command(self):
        cmd = _build_scan_command(answered_update_ids={100, 200, 300})
        self.assertIn("--answered-ids", cmd)
        ids_str = cmd[cmd.index("--answered-ids") + 1]
        self.assertEqual(ids_str, "100,200,300")

    def test_no_answered_ids_omits_flag(self):
        cmd = _build_scan_command(answered_update_ids=None)
        self.assertNotIn("--answered-ids", cmd)

    def test_empty_answered_ids_omits_flag(self):
        cmd = _build_scan_command(answered_update_ids=set())
        self.assertNotIn("--answered-ids", cmd)


if __name__ == "__main__":
    unittest.main()
