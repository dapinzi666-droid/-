"""本机安装器的缓存、校验和失败状态检查，不声称运行过Windows。"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_setup


def test_wheel(folder, name, version):
    path = folder / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return path


class LocalSetupTests(unittest.TestCase):
    def test_exact_versions_and_cpu_build_are_locked_to_real_file_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            wheel = test_wheel(directory, "torch", "2.6.0+cpu")
            test_wheel(directory, "Example_Package", "1.0")
            lock, hashes = local_setup.make_hash_lock("torch==2.6.0+cpu\nexample-package==1.0\n", directory)
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            self.assertIn(f"torch==2.6.0+cpu --hash=sha256:{digest}", lock)
            self.assertIn("example-package==1.0", lock)
            self.assertEqual(hashes[wheel.name], digest)
            with self.assertRaises(ValueError):
                local_setup.make_hash_lock("torch==2.7.0\n", directory)
            with self.assertRaises(ValueError):
                local_setup.make_hash_lock("example @ file:///TEST_ONLY", directory)

    def test_tampered_wheel_or_lock_is_not_an_accepted_offline_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            wheel = test_wheel(directory, "example", "1.0")
            lock, hashes = local_setup.make_hash_lock("example==1.0", directory)
            (directory / "requirements-locked.txt").write_text(lock)
            hashes["requirements-locked.txt"] = hashlib.sha256(lock.encode()).hexdigest()
            (directory / "cache_manifest.json").write_text(json.dumps({"sha256": hashes}))
            local_setup.verify_cache(directory)
            wheel.write_bytes(b"TEST_ONLY tampered wheel")
            with self.assertRaises(ValueError):
                local_setup.verify_cache(directory)

    def test_cache_identity_separates_python_minor_and_requirements(self):
        base = local_setup.cache_name(b"numpy==1.26.4", (3, 12))
        self.assertNotEqual(base, local_setup.cache_name(b"numpy==1.26.4", (3, 11)))
        self.assertNotEqual(base, local_setup.cache_name(b"numpy==2.0", (3, 12)))

    def test_cache_manifest_cannot_read_outside_its_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "cache_manifest.json").write_text(json.dumps({"sha256": {"../TEST_ONLY": "0" * 64}}))
            with self.assertRaises(ValueError):
                local_setup.verify_cache(directory)

    def test_failed_install_clears_previous_ready_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = root / ".venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            state = root / "local_installation.json"
            state.write_text('{"ready":true}')
            info = json.dumps([list(sys.version_info[:2]), True])
            with (patch.object(local_setup, "ROOT", root), patch.object(local_setup, "STATE", state),
                  patch.object(local_setup.platform, "system", return_value="Windows"),
                  patch.object(local_setup.platform, "machine", return_value="AMD64"),
                  patch.object(local_setup, "run", return_value=info),
                  patch.object(local_setup, "retain_python_installer", side_effect=OSError("TEST_ONLY unavailable")),
                  patch.object(sys, "argv", ["local_setup.py"])):
                self.assertEqual(local_setup.main(), 1)
            self.assertFalse(json.loads(state.read_text())["ready"])

    def test_backup_preflight_rejects_deleted_assets_without_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = root / "local_installation.json"
            state.write_text('{"ready":true,"offline_dependencies_verified":true}')
            (root / "requirements.txt").write_text("TEST_ONLY==1.0")
            with (patch.object(local_setup, "ROOT", root), patch.object(local_setup, "STATE", state),
                  patch.object(local_setup.platform, "system", return_value="Windows"),
                  patch.object(local_setup.platform, "machine", return_value="AMD64"),
                  patch.object(local_setup, "retain_python_installer") as installer,
                  patch.object(local_setup, "run") as run,
                  patch.object(sys, "argv", ["local_setup.py", "--verify-assets"])):
                self.assertEqual(local_setup.main(), 1)
                installer.assert_not_called()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
