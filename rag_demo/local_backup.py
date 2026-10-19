"""校验并备份本地项目；不收集项目外文件，不备份不可迁移的虚拟环境。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from uuid import uuid4
import zipfile

from rag_core import MODEL_SPECS, ROOT, sha256_file

EXCLUDED = {".venv", "runtime", ".cache", "__pycache__", ".git"}
LARGE_ASSETS = {"models", "software", "wheelhouse", "results"}


def _safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or any(c in value for c in ("\\", ":", "\0")):
        raise ValueError("模型清单包含不安全的相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in value.split("/")):
        raise ValueError(f"模型清单路径越界或不规范：{value}")
    return path


def _validate_models(project: Path) -> dict[str, str]:
    """先检查全部路径，再读取权重；返回备份时应再次匹配的模型摘要。"""
    directory = project / "models"
    manifest_path = directory / "manifest.json"
    if directory.is_symlink() or manifest_path.is_symlink():
        raise ValueError("模型目录或清单不能是符号链接")
    if not manifest_path.is_file():
        raise ValueError("缺少 models/manifest.json；完整备份需要先下载模型，或明确选择 --code-only")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("模型清单必须是对象")
    for name, repo_id in MODEL_SPECS.items():
        entry = manifest.get(name, {})
        if (not isinstance(entry, dict) or entry.get("repo_id") != repo_id
                or not entry.get("revision") or not entry.get("sha256")):
            raise ValueError(f"模型清单不完整：{name}")
    expected = {}
    for name, entry in manifest.items():
        model_name = _safe_relative(name)
        if len(model_name.parts) != 1 or not isinstance(entry, dict):
            raise ValueError("模型名称或条目不合法")
        hashes = entry.get("sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"模型文件摘要缺失：{name}")
        if not any(str(file).endswith(".safetensors") for file in hashes):
            raise ValueError(f"模型清单缺少 safetensors 权重：{name}")
        for file, digest in hashes.items():
            relative = _safe_relative(file)
            local = directory / name / Path(*relative.parts)
            if any(part in EXCLUDED for part in relative.parts):
                raise ValueError("模型清单引用了不会备份的缓存或环境目录")
            for parent in [local, *local.parents]:
                if parent == project:
                    break
                if parent.is_symlink():
                    raise ValueError(f"模型文件路径不能经过符号链接：{file}")
            if not local.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"模型文件路径越界：{file}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"模型文件摘要格式不合法：{file}")
            expected[local.relative_to(project).as_posix()] = digest.lower()
    for relative, digest in expected.items():
        local = project / relative
        if not local.is_file() or sha256_file(local) != digest:
            raise ValueError(f"模型文件缺失或 SHA256 不匹配：{relative}")
    expected["models/manifest.json"] = sha256_file(manifest_path)
    return expected


def create_backup(project_dir: Path | str = ROOT, output: Path | str | None = None,
                  code_only: bool = False) -> Path:
    """完整模式校验清单模型；代码模式明确不包含模型、软件、轮子及实验结果。"""
    project = Path(project_dir).resolve()
    if not project.is_dir():
        raise ValueError(f"项目目录不存在：{project}")
    expected = {} if code_only else _validate_models(project)
    now = datetime.now(timezone.utc)
    name = f"rag_demo_local_backup-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}.zip"
    destination = Path(output).absolute() if output else project.parent / name
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(f"备份目标已存在，请更换文件名：{destination}")
    if destination.suffix.lower() != ".zip":
        raise ValueError("备份输出必须使用 .zip 后缀")
    # 输出不可覆盖源模型文件；先快照文件列表，再创建临时 ZIP，避免递归。
    destination.parent.mkdir(parents=True, exist_ok=True)
    omitted = EXCLUDED | (LARGE_ASSETS if code_only else set())
    files = []
    for current, directories, names in os.walk(project, followlinks=False):
        folder = Path(current)
        directories[:] = sorted(d for d in directories if d not in omitted and not (folder / d).is_symlink())
        for name in sorted(names):
            file = folder / name
            relative = file.relative_to(project).as_posix()
            if (name in EXCLUDED or file.is_symlink() or not file.is_file()
                    or file.resolve() == destination.resolve()
                    or relative == "backup_manifest.json"):
                continue
            files.append((file, relative))
    entries = []
    handle, temporary_name = tempfile.mkstemp(prefix=".rag-backup-", suffix=".partial",
                                             dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True, compresslevel=6) as archive:
            for file, relative in files:
                if file.is_symlink():
                    raise ValueError(f"备份期间文件变为符号链接：{relative}")
                info = zipfile.ZipInfo.from_file(file, arcname=relative)
                # 权重、安装器和轮子通常无需再次高强度压缩。
                info.compress_type = (zipfile.ZIP_STORED if file.suffix.lower() in
                                      {".safetensors", ".whl", ".exe", ".msi", ".zip", ".gz"}
                                      else zipfile.ZIP_DEFLATED)
                digest, size = hashlib.sha256(), 0
                with file.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        target.write(chunk)
                hexdigest = digest.hexdigest()
                if relative in expected and hexdigest != expected[relative]:
                    raise ValueError(f"备份期间模型文件发生变化：{relative}")
                entries.append({"path": relative, "size": size, "sha256": hexdigest})
            included = {entry["path"] for entry in entries}
            if not code_only and not set(expected).issubset(included):
                raise ValueError("模型清单中的部分文件没有进入 ZIP，停止生成完整备份")
            manifest = {
                "created_at": now.isoformat(), "mode": "code-only" if code_only else "full",
                "models_complete": not code_only,
                "wheelhouse_present": any(p.startswith("wheelhouse/") and p.endswith(".whl") for p in included),
                "software_present": any(p.startswith("software/") for p in included),
                "files": entries,
                "notes": ["不包含 .venv/runtime/缓存/Git；虚拟环境应由离线 wheel 重建。",
                          "安装器和 wheel 存在不代表依赖完整；需匹配目标操作系统、CPU 和 Python 版本，不保证跨平台。",
                          "models_complete 仅表示模型清单和所列文件校验通过，不能代替真实运行验证。",
                          "代码备份不包含模型、软件、wheel 或实验结果，不能据此声称可离线运行。" if code_only
                          else "请同时保留 backup_manifest.json，可逐文件校验备份内容。"],
            }
            archive.writestr("backup_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        if destination.exists():
            raise FileExistsError(f"备份目标已出现，请更换文件名：{destination}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"{'不完整代码备份' if code_only else '已校验模型的项目备份'}：{destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="输出 ZIP；默认在项目父目录")
    parser.add_argument("--code-only", action="store_true", help="不完整的代码备份，不含模型等大文件")
    args = parser.parse_args()
    try:
        create_backup(output=args.output, code_only=args.code_only)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"备份失败：{exc}\n")


if __name__ == "__main__":
    main()
