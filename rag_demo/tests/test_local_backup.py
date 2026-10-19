"""TEST_ONLY：临时假文件只测试打包与校验，不表示真实模型运行成功。"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_backup import EXCLUDED, create_backup
from rag_core import MODEL_SPECS


class LocalBackupTests(unittest.TestCase):
    def make_project(self, folder):
        project = Path(folder) / "TEST_ONLY-project"
        project.mkdir()
        (project / "README.md").write_text("TEST_ONLY 备份测试，不是真实运行结果。", encoding="utf-8")
        manifest = {}
        for name, repo in MODEL_SPECS.items():
            target = project / "models" / name
            target.mkdir(parents=True)
            payload = f"TEST_ONLY fake weights for {name}".encode()
            (target / "TEST_ONLY.safetensors").write_bytes(payload)
            manifest[name] = {"repo_id": repo, "revision": "TEST_ONLY",
                              "sha256": {"TEST_ONLY.safetensors": hashlib.sha256(payload).hexdigest()}}
        (project / "models" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return project, manifest

    def backup(self, project, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return create_backup(project, **kwargs)

    def test_zip_excludes_environments_symlinks_and_itself_and_hashes_archived_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            project, _ = self.make_project(directory)
            for name in EXCLUDED:
                for parent in (project, project / "docs"):
                    target = parent / name
                    target.mkdir(parents=True)
                    (target / "TEST_ONLY.txt").write_text("TEST_ONLY excluded")
            for name, filename in (("software", "TEST_ONLY.exe"), ("wheelhouse", "TEST_ONLY.whl"),
                                   ("results", "TEST_ONLY.json")):
                (project / name).mkdir()
                (project / name / filename).write_text("TEST_ONLY not an actual installer/model result")
            outside = Path(directory) / "TEST_ONLY-outside"
            outside.mkdir()
            (outside / "private.txt").write_text("TEST_ONLY never archive")
            try:
                (project / "linked-directory").symlink_to(outside, target_is_directory=True)
                (project / "linked-file.txt").symlink_to(outside / "private.txt")
            except (OSError, NotImplementedError):
                pass  # Windows 未授予符号链接权限时，其余校验仍有意义。
            output = self.backup(project, output=project / "TEST_ONLY-backup.zip")
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                manifest = json.loads(archive.read("backup_manifest.json"))
                self.assertNotIn(output.name, names)
                self.assertFalse(any(".rag-backup-" in name for name in names))
                self.assertFalse(any(part in EXCLUDED for name in names for part in Path(name).parts))
                self.assertFalse(any("linked-" in name for name in names))
                self.assertTrue(manifest["models_complete"])
                self.assertTrue(manifest["software_present"])
                self.assertTrue(manifest["wheelhouse_present"])
                self.assertEqual(set(names), {entry["path"] for entry in manifest["files"]}
                                 | {"backup_manifest.json"})
                for entry in manifest["files"]:
                    data = archive.read(entry["path"])
                    self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
                    self.assertEqual(entry["size"], len(data))

    def test_missing_models_fail_unless_explicitly_selecting_incomplete_code_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            project, _ = self.make_project(directory)
            (project / "models" / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.backup(project)
            output = self.backup(project, code_only=True)
            self.assertEqual(output.parent, project.parent)
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("backup_manifest.json"))
                self.assertFalse(manifest["models_complete"])
                self.assertEqual(manifest["mode"], "code-only")
                self.assertIn("README.md", archive.namelist())
                self.assertFalse(any(name.startswith("models/") for name in archive.namelist()))

    def test_manifest_rejects_parent_absolute_and_windows_paths(self):
        for unsafe in ("../outside.safetensors", "/tmp/outside.safetensors", "C:/outside.safetensors",
                       "..\\outside.safetensors"):
            with self.subTest(path=unsafe), tempfile.TemporaryDirectory() as directory:
                project, manifest = self.make_project(directory)
                manifest["embedding"]["sha256"][unsafe] = "0" * 64
                (project / "models" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.backup(project)
                self.assertFalse(list(Path(directory).rglob("*.zip")))

    def test_changed_model_bytes_are_rejected_before_creating_a_complete_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            project, _ = self.make_project(directory)
            (project / "models" / "embedding" / "TEST_ONLY.safetensors").write_bytes(b"TEST_ONLY corrupted")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.backup(project)
            self.assertFalse(list(Path(directory).rglob("*.zip")))


if __name__ == "__main__":
    unittest.main()
