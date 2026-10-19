"""Windows安装入口用的小检查，不修改系统。"""
import os
import platform
import sys

minor = ".".join(map(str, sys.version_info[:2]))
target = os.environ.get("RAG_TARGET_MINOR", "")
valid = (sys.implementation.name == "cpython" and minor in ("3.11", "3.12") and sys.maxsize > 2**32
         and platform.machine().upper() in ("AMD64", "X86_64")
         and (not target or target == minor))
sys.exit(0 if valid else 1)
