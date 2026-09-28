"""Regression checks for dataset identity, frame integrity, and human mapping."""

import tempfile
import unittest
from pathlib import Path

from mmau_mapping import merge_decision, merge_forms
from mmau_preflight import sequence_info, signature


class TestMMAU(unittest.TestCase):
    def test_global_metadata_id_is_not_original_video_number(self) -> None:
        m = dict(type="10", id="1963", video_name="10_1", total_frames="50", t_co="34", t_ai="18", t_ae="50")
        m["accident occurred"] = "1"
        a = dict(type=10, video=1, total_frames=50, anchor=34, abnormal_start=18, abnormal_end=50, accident=1)
        self.assertEqual(signature(m, True), signature(a))
        a["total_frames"] = 878
        self.assertNotEqual(signature(m, True), signature(a))

    def test_sequence_gaps_duplicates_and_zero_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ["0001.jpg", "0003.jpg", "3.png"]:
                (p / name).write_bytes(b"image")
            (p / "0004.jpg").touch()
            s = sequence_info(p)
            self.assertEqual((s["missing_indices"], s["duplicate_indices"], s["zero_byte_frames"]), (1, 1, 1))

    def test_empty_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(sequence_info(Path(tmp))["last_frame"])

    def test_merge_rules(self) -> None:
        self.assertEqual(merge_decision("single", "single"), ("unique", "single"))
        self.assertEqual(merge_decision("single", "out_of_scope"), ("ambiguous", None))
        self.assertEqual(merge_decision("out_of_scope", "out_of_scope"), ("out_of_scope", None))
        self.assertEqual(merge_decision("ambiguous", "ambiguous"), ("ambiguous", None))
        with self.assertRaises(ValueError):
            merge_decision("", "single")

    def test_merge_by_code_and_reject_modified_definitions(self) -> None:
        codes = sorted(set(range(1, 63)) - {25, 28, 31, 46})
        definitions = [dict(code=c, definition=f"fixture {c}") for c in codes]
        rows = [
            dict(code=c, name_from_definition_text=f"fixture {c}", decision="ambiguous", note="test fixture")
            for c in codes
        ]
        self.assertEqual(len(merge_forms(rows, list(reversed(rows)), definitions)), 58)
        bad = [dict(r) for r in rows]
        bad[0]["name_from_definition_text"] = "altered"
        with self.assertRaises(ValueError):
            merge_forms(rows, bad, definitions)
        with self.assertRaises(ValueError):
            merge_forms(rows, rows[:-1] + [rows[0]], definitions)


if __name__ == "__main__":
    unittest.main()
