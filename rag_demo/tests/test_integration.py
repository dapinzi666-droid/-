"""真实 Chroma 持久化集成测试；已知向量只用来验证数据库，不代表模型效果。"""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_core import Document, RagSystem

AVAILABLE = importlib.util.find_spec("chromadb") is not None and importlib.util.find_spec("numpy") is not None


class TestOnlyKnownVectors:
    """TEST_ONLY：测试边界显式提供向量，绝不进入正式演示和导出。"""
    manifest = {"embedding": {"repo_id": "TEST_ONLY", "revision": "known-vectors-v1"}}

    def __init__(self):
        self.calls = 0

    def embed(self, texts, is_query=False):
        import numpy as np
        self.calls += 1
        vectors = {"甲资料": [1.0, 0.0, 0.0], "乙资料": [0.0, 1.0, 0.0],
                   "乙资料更新": [0.0, 1.0, 0.0], "寻找乙": [0.0, 1.0, 0.0]}
        return np.asarray([vectors[t] for t in texts], dtype="float32")


@unittest.skipUnless(AVAILABLE, "需要先安装依赖；这是Chroma集成检查，不是模型效果测试")
class ChromaIntegrationTests(unittest.TestCase):
    def test_cosine_search_returns_expected_source_and_survives_reopen(self):
        documents = [Document("甲.txt", "甲资料"), Document("乙.txt", "乙资料")]
        with tempfile.TemporaryDirectory() as folder:
            models = TestOnlyKnownVectors()
            system = RagSystem(documents, models, Path(folder) / "db")
            self.assertEqual(system.build_index("paragraph"), 2)
            hits = system.retrieve("寻找乙", "paragraph", top_k=9)
            self.assertEqual(len(hits), 2)
            self.assertEqual(hits[0]["source"], "乙.txt")
            self.assertEqual(hits[0]["text"], "乙资料")
            self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)
            fresh_models = TestOnlyKnownVectors()
            reopened = RagSystem(documents, fresh_models, Path(folder) / "db")
            reopened.build_index("paragraph")
            self.assertEqual(fresh_models.calls, 0, "有效的持久化索引应该复用")
            self.assertEqual(reopened.retrieve("寻找乙", "paragraph")[0]["source"], "乙.txt")

    def test_changed_documents_get_a_new_index_without_resetting_existing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            original = RagSystem([Document("乙.txt", "乙资料")], TestOnlyKnownVectors(), folder)
            original.build_index("paragraph")
            updated = RagSystem([Document("乙.txt", "乙资料更新")], TestOnlyKnownVectors(), folder)
            updated.build_index("paragraph")
            self.assertEqual(updated.retrieve("寻找乙", "paragraph")[0]["text"], "乙资料更新")
            self.assertEqual(original.retrieve("寻找乙", "paragraph")[0]["text"], "乙资料")


if __name__ == "__main__":
    unittest.main()
