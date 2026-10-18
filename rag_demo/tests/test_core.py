"""Core correctness checks; these tests never download or invoke an AI model.

Run from the project directory:
    python -m unittest discover -s tests -v
"""

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core import (  # noqa: E402
    Document,
    build_prompt,
    generation_kwargs,
    load_documents,
    split_documents,
)


class DocumentLoadingTests(unittest.TestCase):
    def test_txt_preserves_the_full_unicode_document(self):
        text = "大数据与人工智能🙂\n保留换行与空行。\n\n第二段不应消失。\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文资料.txt"
            path.write_text(text, encoding="utf-8")
            documents = load_documents(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, text)
        self.assertIn("中文资料.txt", str(documents[0].source))
        self.assertIsNone(documents[0].page)

    def test_directory_loads_each_txt_and_ignores_other_file_types(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "甲.txt").write_text("甲文件的知识。", encoding="utf-8")
            (folder / "乙.txt").write_text("乙文件的知识。", encoding="utf-8")
            (folder / "说明.md").write_text("不应加入知识库。", encoding="utf-8")
            documents = load_documents(folder)

        self.assertCountEqual(
            [document.text for document in documents],
            ["甲文件的知识。", "乙文件的知识。"],
        )
        self.assertEqual(len({str(document.source) for document in documents}), 2)


class ChunkingTests(unittest.TestCase):
    def assert_valid_offsets(self, chunks, original):
        for chunk in chunks:
            with self.subTest(chunk=chunk.id):
                self.assertGreaterEqual(chunk.start, 0)
                self.assertGreater(chunk.end, chunk.start)
                self.assertLessEqual(chunk.end, len(original))
                self.assertEqual(chunk.text, original[chunk.start : chunk.end])

    def test_paragraphs_preserve_multiline_chinese_content_and_offsets(self):
        paragraphs = [
            "中文第一段🙂。\n第二行仍属于同一段。",
            "第二段：全角标点不能丢！",
            "最后一段。",
        ]
        text = "\n\n".join(paragraphs)
        document = Document(source="课程.pdf", text=text, page=7)

        chunks = split_documents([document], "paragraph")

        self.assertEqual([chunk.text for chunk in chunks], paragraphs)
        self.assert_valid_offsets(chunks, text)
        self.assertTrue(all(chunk.source == "课程.pdf" for chunk in chunks))
        self.assertTrue(all(chunk.page == 7 for chunk in chunks))

    def test_fixed_windows_retain_the_short_final_chunk(self):
        text = "甲乙丙丁戊己庚辛壬癸"
        chunks = split_documents(
            [Document(source="十个字.txt", text=text)], "fixed", chunk_size=4
        )

        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in chunks],
            [(0, 4), (4, 8), (8, 10)],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assert_valid_offsets(chunks, text)

    def test_overlap_uses_the_requested_step_and_covers_every_character(self):
        text = "甲乙丙丁戊己庚辛壬癸子丑"
        chunks = split_documents(
            [Document(source="重叠.txt", text=text)],
            "overlap",
            chunk_size=5,
            overlap=2,
        )

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in chunks[:3]],
            [(0, 5), (3, 8), (6, 11)],
        )
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, len(text))
        self.assert_valid_offsets(chunks, text)
        covered = set()
        for index, chunk in enumerate(chunks):
            self.assertLessEqual(len(chunk.text), 5)
            covered.update(range(chunk.start, chunk.end))
            if index:
                self.assertEqual(chunk.start - chunks[index - 1].start, 3)
        self.assertEqual(covered, set(range(len(text))))

    def test_zero_overlap_agrees_with_fixed_windows(self):
        documents = [Document(source="边界.txt", text="abcdefghijk")]
        fixed = split_documents(documents, "fixed", chunk_size=4)
        overlap = split_documents(
            documents, "overlap", chunk_size=4, overlap=0
        )
        self.assertEqual(
            [(chunk.text, chunk.start, chunk.end) for chunk in fixed],
            [(chunk.text, chunk.start, chunk.end) for chunk in overlap],
        )

    def test_documents_never_share_chunks_or_offsets(self):
        documents = [
            Document(source="甲.txt", text="甲乙丙丁戊"),
            Document(source="乙.txt", text="12345"),
        ]
        for strategy in ("paragraph", "fixed", "overlap"):
            with self.subTest(strategy=strategy):
                chunks = split_documents(
                    documents, strategy, chunk_size=4, overlap=1
                )
                self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))
                for document in documents:
                    own_chunks = [
                        chunk for chunk in chunks if chunk.source == document.source
                    ]
                    self.assertTrue(own_chunks)
                    self.assertEqual(own_chunks[0].start, 0)
                    self.assert_valid_offsets(own_chunks, document.text)

    def test_empty_input_has_no_chunks(self):
        for strategy in ("paragraph", "fixed", "overlap"):
            with self.subTest(strategy=strategy):
                self.assertEqual(split_documents([], strategy), [])

    def test_invalid_window_parameters_are_rejected(self):
        documents = [Document(source="参数.txt", text="abcdef")]
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    split_documents(documents, "fixed", chunk_size=size)
        for overlap in (-1, 4, 5):
            with self.subTest(overlap=overlap):
                with self.assertRaises(ValueError):
                    split_documents(
                        documents, "overlap", chunk_size=4, overlap=overlap
                    )


class PromptTests(unittest.TestCase):
    def test_question_and_retrieved_sources_are_visible_with_citation_labels(self):
        question = "Llama 2 提供哪些参数规模？"
        hits = [
            {
                "id": "chunk-a",
                "text": "Llama 2 提供 7B、13B 和 70B 三种参数规模。",
                "source": "论文摘要.txt",
                "score": 0.82,
                "page": None,
            },
            {
                "id": "chunk-b",
                "text": "聊天版本经过了面向对话的微调。",
                "source": "模型说明.pdf",
                "score": 0.75,
                "page": 3,
            },
        ]

        messages = build_prompt(question, hits)

        self.assertIsInstance(messages, list)
        self.assertTrue(messages)
        for message in messages:
            self.assertIn(message["role"], ("system", "user", "assistant"))
            self.assertIsInstance(message["content"], str)
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn(question, prompt)
        self.assertIn("[S1]", prompt)
        self.assertIn("[S2]", prompt)
        for hit in hits:
            self.assertIn(hit["text"], prompt)
            self.assertIn(hit["source"], prompt)
        # Accept natural wording while requiring an explicit unsupported-answer rule.
        self.assertTrue(
            any(word in prompt for word in ("不知道", "无法回答", "资料不足", "信息不足"))
        )
        self.assertTrue(any(word in prompt for word in ("仅", "只能", "只用", "只根据")))


class GenerationParametersTests(unittest.TestCase):
    def test_temperature_zero_selects_greedy_decoding_without_sampling_flags(self):
        kwargs = generation_kwargs(0, seed=42, max_new_tokens=64)

        self.assertIs(kwargs["do_sample"], False)
        self.assertEqual(kwargs["max_new_tokens"], 64)
        for key in ("temperature", "top_p", "top_k", "seed"):
            self.assertNotIn(key, kwargs)

    def test_temperature_one_enables_unrestricted_temperature_sampling(self):
        kwargs = generation_kwargs(1, seed=42, max_new_tokens=64)

        self.assertIs(kwargs["do_sample"], True)
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertEqual(kwargs["top_k"], 0)
        self.assertEqual(kwargs["max_new_tokens"], 64)
        self.assertNotIn("seed", kwargs)

    def test_seed_is_applied_outside_model_generation_arguments(self):
        for temperature in (0, 1):
            with self.subTest(temperature=temperature):
                self.assertEqual(
                    generation_kwargs(temperature, seed=42),
                    generation_kwargs(temperature, seed=123),
                )

    def test_invalid_generation_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            generation_kwargs(-0.1)
        for token_limit in (0, -1):
            with self.subTest(max_new_tokens=token_limit):
                with self.assertRaises(ValueError):
                    generation_kwargs(1, max_new_tokens=token_limit)


if __name__ == "__main__":
    unittest.main()
