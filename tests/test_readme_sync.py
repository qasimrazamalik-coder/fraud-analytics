import json
import tempfile
import unittest
from pathlib import Path

from fraudlab import readme_sync

RESULT = json.loads((Path(__file__).resolve().parents[1] / "reports/metrics.json").read_text())


class Sync(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.results = self.tmp / "m.json"
        self.results.write_text(json.dumps(RESULT))
        self.readme = self.tmp / "README.md"

    def write(self, body):
        self.readme.write_text(f"intro\n{readme_sync.START}\n{body}\n{readme_sync.END}\noutro\n")

    def test_block_carries_the_numbers(self):
        block = readme_sync.render_block(RESULT)
        sel = RESULT["models"][RESULT["selected_model"]]
        self.assertIn(f"{sel['pr_auc']:.3f}", block)
        self.assertIn("SYNTHETIC", block)
        self.assertIn(f"{RESULT['ablation_with_ledger_artefact']['pr_auc']:.3f}", block)

    def test_stale_block_fails_check_then_sync_fixes_it(self):
        self.write("old numbers")
        self.assertEqual(readme_sync.sync(self.readme, self.results, check=True), 1)
        self.assertEqual(readme_sync.sync(self.readme, self.results), 0)
        self.assertEqual(readme_sync.sync(self.readme, self.results, check=True), 0)

    def test_text_outside_the_markers_is_untouched(self):
        self.write("old")
        readme_sync.sync(self.readme, self.results)
        text = self.readme.read_text()
        self.assertTrue(text.startswith("intro\n"))
        self.assertTrue(text.endswith("outro\n"))

    def test_missing_markers_is_an_error(self):
        self.readme.write_text("no markers here")
        self.assertEqual(readme_sync.sync(self.readme, self.results), 2)


if __name__ == "__main__":
    unittest.main()
