import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chiplet_tuner.simulators.compass import resolve_config_paths


class RequestSnapshotTests(unittest.TestCase):
    def test_snapshot_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "requests.json"
            trace.write_text(json.dumps({"batches": []}))
            config = {"request_trace_path": "requests.json",
                      "request_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
            resolved = resolve_config_paths(config, root / "config.json", root / "run")
            self.assertEqual(Path(resolved["request_trace_path"]).read_bytes(), trace.read_bytes())
            trace.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Request trace hash"):
                resolve_config_paths(config, root / "config.json", root / "run2")


if __name__ == "__main__":
    unittest.main()
