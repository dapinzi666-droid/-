"""TEST_ONLY：只检查实验编排和导出；不作为任何真实模型结果发布。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import render_record, run_experiments
from rag_core import build_prompt


class TestOnlySystem:
    """TEST_ONLY：追踪调用，不下载或生成模型答案。"""

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def answer(self, question, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("TEST_ONLY: intentional failure")
        hits = kwargs["hits"]
        if hits is None:
            hits = [{"id": "TEST_ONLY", "source": "TEST_ONLY.txt", "page": None,
                     "text": "TEST_ONLY <script>not executable</script>", "score": 0.8,
                     "start": 0, "end": 9}]
        return {"question": question, "strategy": kwargs["strategy"],
                "temperature": kwargs["temperature"], "seed": kwargs["seed"],
                "top_k": kwargs["top_k"], "chunk_size": 240, "overlap": 80,
                "generation_kwargs": {"max_new_tokens": kwargs["max_new_tokens"]},
                "hits": hits, "messages": build_prompt(question, hits),
                "answer": "TEST_ONLY <script>not executable</script>", "seconds": 0,
                "backend": "TEST_ONLY", "model_versions": {"TEST_ONLY": {"revision": "TEST_ONLY"}}}


class ExperimentTests(unittest.TestCase):
    questions = [{"id": "q1", "question": "TEST_ONLY q1", "kind": "TEST_ONLY"},
                 {"id": "q2", "question": "TEST_ONLY q2", "kind": "TEST_ONLY"}]

    def test_quick_run_reuses_context_and_exports_only_the_selected_question(self):
        system = TestOnlySystem()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = run_experiments(system, self.questions, quick=True, output_root=Path(directory))
            records = json.loads((output / "records.json").read_text(encoding="utf-8"))
            report = (output / "report.html").read_text(encoding="utf-8")
            self.assertTrue((output / "summary.csv").exists())

        self.assertEqual(len(system.calls), 7)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(record["question_id"] == "q2" for record in records))
        self.assertEqual([call["strategy"] for call in system.calls[:3]],
                         ["paragraph", "fixed", "overlap"])
        self.assertTrue(all(call["hits"] is None for call in system.calls[:3]))
        shared_hits = system.calls[3]["hits"]
        self.assertTrue(all(call["hits"] is shared_hits for call in system.calls[3:]))
        self.assertTrue(all(call["top_k"] == 2 and call["max_new_tokens"] == 96
                            for call in system.calls))
        temperatures = [record for record in records if record["experiment"] == "temperature"]
        self.assertEqual([(record["temperature"], record["seed"]) for record in temperatures],
                         [(0, 42), (0, 43), (1, 42), (1, 43), (1, 44)])
        self.assertEqual(len({record["context_sha256"] for record in temperatures}), 1)
        self.assertTrue(temperatures[0]["generation_reused"])
        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)

    def test_render_escapes_source_question_and_answer(self):
        fragment = render_record({"question": "TEST_ONLY <img src=x onerror=alert(1)>",
                                  "answer": "TEST_ONLY <script>bad</script>",
                                  "hits": [{"source": "TEST_ONLY <b>source</b>",
                                            "text": "TEST_ONLY <iframe>", "score": 0.2}]})
        for literal in ("<img", "<script>", "<b>", "<iframe>"):
            self.assertNotIn(literal, fragment)
        self.assertIn("&lt;img", fragment)

    def test_failed_generation_does_not_publish_a_report(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / "TEST_ONLY-results"
            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                run_experiments(TestOnlySystem(fail_at=4), self.questions,
                                quick=True, output_root=root)
            self.assertFalse(root.exists())

    def test_length_limit_is_visible_when_real_generation_metadata_flags_it(self):
        flagged = render_record({"answer": "TEST_ONLY unfinished",
                                 "generation_info": {"output_tokens": 96,
                                                     "possibly_truncated": True}})
        self.assertIn("回答达到长度上限", flagged)
        self.assertIn("max_new_tokens", flagged)
        self.assertIn("160", flagged)
        self.assertNotIn("回答达到长度上限", render_record({"answer": "TEST_ONLY"}))


if __name__ == "__main__":
    unittest.main()
