"""在用户自己的 Windows x64 电脑准备项目，并保留可校验的离线安装材料。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile

from rag_core import ROOT, check_models, sha256_file

TORCH = "torch==2.6.0+cpu"
PIP = "pip==25.0.1"
STATE = ROOT / "local_installation.json"
PYTHON_INSTALLERS = {
    (3, 12): ("3.12.10", "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"),
    (3, 11): ("3.11.9", "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"),
}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def cache_name(requirements: bytes, minor: tuple[int, int], machine: str = "AMD64") -> str:
    signature = hashlib.sha256(requirements + TORCH.encode() + PIP.encode()).hexdigest()[:12]
    return f"win_{machine.lower()}-cp{minor[0]}{minor[1]}-{signature}"


def make_hash_lock(frozen: str, wheel_dir: Path) -> tuple[str, dict]:
    """从本机实际安装版本和对应 wheel 的 METADATA 生成全依赖哈希锁。"""
    wheels = {}
    for wheel in sorted(wheel_dir.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            members = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(members) != 1:
                raise ValueError(f"wheel 元数据不明确：{wheel.name}")
            meta = BytesParser().parsebytes(archive.read(members[0]))
        if not meta["Name"] or not meta["Version"]:
            raise ValueError(f"wheel 缺少名称或版本：{wheel.name}")
        identity = (canonical_name(meta["Name"]), meta["Version"])
        wheels.setdefault(identity, []).append((wheel.name, sha256_file(wheel)))
    lines, selected = [], {}
    for raw in frozen.splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", raw.strip())
        if not match:
            raise ValueError(f"离线备份只接受精确版本，不能包含可编辑安装或外部路径：{raw}")
        name, version = match.groups()
        matches = wheels.get((canonical_name(name), version), [])
        if not matches:
            raise ValueError(f"离线 wheel 不完整：缺少 {name}=={version}")
        hashes = " ".join("--hash=sha256:" + digest for _, digest in matches)
        lines.append(f"{canonical_name(name)}=={version} {hashes}")
        selected.update({file: digest for file, digest in matches})
    if not lines:
        raise ValueError("不能生成空的依赖锁")
    return "\n".join(sorted(lines)) + "\n", selected


def verify_cache(directory: Path) -> dict:
    manifest_path = directory / "cache_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"缺少本机离线缓存清单：{manifest_path}。首次应联网安装。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("离线缓存清单格式错误")
    files = manifest.get("sha256", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("离线缓存清单没有文件")
    for name, digest in files.items():
        if not name or Path(name).name != name or any(c in name for c in "\\/:"):
            raise ValueError("离线缓存清单包含越界路径")
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"离线文件缺失或校验失败：{name}。停止安装，请重新获取可靠副本。")
    if "requirements-locked.txt" not in files:
        raise ValueError("缓存没有包含精确依赖锁")
    return manifest


def run(args: list[str], capture: bool = False, extra_env: dict | None = None) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONUTF8"] = "1"
    if extra_env:
        environment.update(extra_env)
    result = subprocess.run(args, cwd=ROOT, env=environment, check=True,
                            text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout if capture else ""


def pip(python: Path, *args: str, capture: bool = False):
    # --isolated 避免意外继承第三方软件源或 trusted-host 配置；不关闭 TLS 验证。
    return run([str(python), "-m", "pip", "--isolated", "--disable-pip-version-check", *args], capture)


def verify_python_installer(path: Path) -> None:
    """交给 Windows 验证数字签名和发布者；无签名或信任失败时不执行。"""
    script = (
        "$ErrorActionPreference='Stop'; "
        "$s=Get-AuthenticodeSignature -LiteralPath $env:RAG_INSTALLER_PATH; "
        "if($s.Status -ne 'Valid'){throw 'Python installer signature is not valid'}; "
        "if($s.SignerCertificate.Subject -notmatch '(?i)(^|,\\s*)(O|CN)\\s*=\\s*Python Software Foundation(,|$)')"
        "{throw 'Unexpected installer publisher'}"
    )
    run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        extra_env={"RAG_INSTALLER_PATH": str(path)})


def retain_python_installer(offline: bool) -> Path:
    version, url = PYTHON_INSTALLERS[sys.version_info[:2]]
    directory = ROOT / "software"
    directory.mkdir(exist_ok=True)
    target = directory / f"python-{version}-amd64.exe"
    if not target.is_file():
        if offline:
            raise ValueError(f"备份中缺少对应 Python 安装器：{target.name}")
        print(f"保留官方 Python {version} 安装器到 software/（用于以后重建）。", flush=True)
        temporary = target.with_name(target.stem + ".downloading.exe")
        try:
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            verify_python_installer(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    verify_python_installer(target)
    (directory / "python_installer.json").write_text(json.dumps({
        "version": version, "source": url, "filename": target.name,
        "sha256": sha256_file(target), "windows_signature_verified": True,
        "note": "用于对应Python小版本的重建，不宣称是最新Python版本。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def prepare_dependencies(python: Path, cache: Path, offline: bool) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / "requirements-locked.txt"
    if offline:
        manifest = verify_cache(cache)
        if manifest.get("platform") != "win_amd64" or manifest.get("python_minor") != list(sys.version_info[:2]):
            raise ValueError("缓存平台或Python小版本不匹配，不能用于这台电脑。")
    else:
        print("下载匹配本机的CPU PyTorch与依赖，并保留wheel安装文件。", flush=True)
        pip(python, "download", "--only-binary=:all:", "--dest", str(cache), TORCH,
            "--index-url", "https://download.pytorch.org/whl/cpu")
        pip(python, "download", "--only-binary=:all:", "--dest", str(cache),
            "--find-links", str(cache), "--index-url", "https://pypi.org/simple",
            "-r", str(ROOT / "requirements.txt"), PIP)
        pip(python, "install", "--no-index", "--find-links", str(cache), TORCH, PIP,
            "-r", str(ROOT / "requirements.txt"))
        frozen = pip(python, "freeze", "--all", capture=True)
        (cache / "installed-versions.txt").write_text(frozen, encoding="utf-8")
        # 补齐已有虚拟环境中版本较旧但仍满足约束的依赖，不能只保存最新版轮子。
        pip(python, "download", "--only-binary=:all:", "--dest", str(cache),
            "--find-links", str(cache), "--index-url", "https://pypi.org/simple",
            "-r", str(cache / "installed-versions.txt"))
        contents, hashes = make_hash_lock(frozen, cache)
        lock.write_text(contents, encoding="utf-8")
        hashes[lock.name] = sha256_file(lock)
        (cache / "cache_manifest.json").write_text(json.dumps({
            "platform": "win_amd64", "python_minor": list(sys.version_info[:2]),
            "sha256": hashes,
        }, indent=2), encoding="utf-8")
    pip(python, "install", "--no-index", "--find-links", str(cache),
        "--require-hashes", "-r", str(lock))
    pip(python, "check")


def verify_offline_rebuild(cache: Path) -> None:
    print("在临时新环境检查离线重建，不复用当前已安装的依赖。", flush=True)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="offline-check-", dir=ROOT / ".cache") as folder:
        venv.EnvBuilder(with_pip=True).create(folder)
        python = Path(folder) / "Scripts" / "python.exe"
        pip(python, "install", "--no-index", "--find-links", str(cache),
            "--require-hashes", "-r", str(cache / "requirements-locked.txt"))
        pip(python, "check")
        run([str(python), "-c", "import rag_core; import torch, transformers, chromadb, onnxruntime; print('Offline dependency imports passed')"],
            extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="只使用已有安装器、wheel与模型恢复")
    mode.add_argument("--verify-assets", action="store_true", help="备份前复查安装器、依赖轮子和模型，不下载")
    args = parser.parse_args()
    if platform.system() != "Windows" or platform.machine().upper() not in {"AMD64", "X86_64"}:
        parser.error("这个入口面向 Windows 10/11 x64；其他系统请使用原项目说明。")
    if sys.version_info[:2] not in PYTHON_INSTALLERS or sys.maxsize <= 2**32 or sys.implementation.name != "cpython":
        parser.error("需要64位Python 3.11或3.12，请运行install_local_windows.bat。")
    if args.verify_assets:
        try:
            previous = json.loads(STATE.read_text(encoding="utf-8"))
            if not previous.get("ready") or not previous.get("offline_dependencies_verified"):
                raise ValueError("还没有完成本机真实自检及离线依赖重建。")
            cache = ROOT / "wheelhouse" / cache_name((ROOT / "requirements.txt").read_bytes(), sys.version_info[:2])
            verify_cache(cache)
            retain_python_installer(offline=True)
            check_models(verify=True)
            print("当前安装器、离线依赖与模型已重新校验，可以开始备份。")
            return 0
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"备份材料校验失败：{exc}", file=sys.stderr)
            return 1
    state = {"ready": False, "started_at": datetime.now(timezone.utc).isoformat(),
             "python": platform.python_version(), "phase": "starting"}

    def status(phase):
        state["phase"] = phase
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n=== " + phase + " ===", flush=True)

    try:
        status("准备本机Python和项目虚拟环境")
        (ROOT / "local_python_minor.txt").write_text(".".join(map(str, sys.version_info[:2])) + "\n", encoding="ascii")
        python = ROOT / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
        info = json.loads(run([str(python), "-c", "import sys,json; print(json.dumps([list(sys.version_info[:2]),sys.maxsize>2**32]))"], True))
        if info != [list(sys.version_info[:2]), True]:
            raise ValueError("现有.venv版本不同；请用原来的Python运行，或保留旧.venv后再建新环境。")
        status("保存经过签名校验的官方Python安装器")
        retain_python_installer(args.offline)
        cache = ROOT / "wheelhouse" / cache_name((ROOT / "requirements.txt").read_bytes(), sys.version_info[:2])
        status("安装依赖并保留本机精确版本与哈希锁")
        prepare_dependencies(python, cache, args.offline)
        status("验证新环境可以离线重建依赖")
        verify_offline_rebuild(cache)
        state["offline_dependencies_verified"] = True
        state["wheelhouse"] = cache.relative_to(ROOT).as_posix()
        run([str(python), "-m", "ipykernel", "install", "--prefix", str(ROOT / ".venv"),
             "--name", "rag-demo", "--display-name", "RAG Demo (project)"])
        status("准备本地模型")
        if args.offline:
            check_models(verify=True)
        else:
            run([str(python), str(ROOT / "run_demo.py"), "--prepare"])
        status("真实模型自检（检索和生成，不是模拟回答）")
        run([str(python), str(ROOT / "run_demo.py"), "--smoke"],
            extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        state.update(ready=True, finished_at=datetime.now(timezone.utc).isoformat(),
                     model_manifest_sha256=sha256_file(ROOT / "models" / "manifest.json"))
        status("本机真实自检通过；仍需核对答案并完成正式对照实验")
        print("运行入口：run_local_windows.bat；本地备份：backup_local_windows.bat", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        state.update(ready=False, error=str(exc))
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n本机准备未完成：{exc}\n已下载的文件已保留。请记录最后错误。", file=sys.stderr)
        print("若错误包含DLL加载失败，请查看docs/本机安装与保留.md中的微软运行库说明。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
