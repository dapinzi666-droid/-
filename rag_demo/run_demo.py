"""初次准备、离线自检和实验入口。运行 --help 查看命令。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import uuid
from pathlib import Path

from rag_core import (ROOT, DEFAULT_QUESTION, ModelUnavailableError, check_models,
                      make_system, prepare_models)


def doctor() -> int:
    print("Python:", sys.version.split()[0], "|", platform.system(), platform.machine())
    print("解释器：", sys.executable)
    valid = sys.version_info[:2] in {(3, 11), (3, 12)} and sys.maxsize > 2**32
    if not valid:
        print("请使用 64 位 Python 3.11 或 3.12。")
    for package in ("torch", "transformers", "huggingface-hub", "numpy", "chromadb", "jupyterlab", "pypdf"):
        try:
            print(f"  {package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            valid = False
            print(f"  {package}: 未安装，请先运行 setup 脚本")
    try:
        check_models()
        print("模型文件已准备；请运行 --smoke 验证实际推理。")
    except ModelUnavailableError:
        print("模型尚未完整准备：安装依赖后运行 --prepare。此检查不等于模型问答通过。")
    return 0 if valid else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="任务2 · 选题A · 本地基础RAG（真实模型，无需API）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--doctor", action="store_true", help="检查Python、依赖与模型文件是否就绪")
    mode.add_argument("--prepare", action="store_true", help="首次联网下载两个官方模型")
    mode.add_argument("--smoke", action="store_true", help="本地校验模型文件、建索引并实际问答一次")
    mode.add_argument("--experiments", action="store_true", help="运行切分/温度对照，导出JSON CSV HTML")
    mode.add_argument("--ask", action="store_true", help="运行一次真实问答")
    parser.add_argument("--quick", action="store_true", help="对照实验只使用一个综合问题")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "knowledge.txt")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--strategy", choices=["paragraph", "fixed", "overlap"], default="overlap")
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    if args.doctor:
        return doctor()
    if args.prepare:
        manifest = prepare_models()
        print("两个模型均准备完成。版本：")
        print(json.dumps({k: v["revision"] for k, v in manifest.items()}, indent=2))
        print("下一步：python run_demo.py --smoke")
        return 0
    if not (args.smoke or args.experiments or args.ask):
        parser.print_help()
        return 0
    if args.quick and not args.experiments:
        parser.error("--quick 需要与 --experiments 一起使用")
    check_models(verify=args.smoke)
    print("加载本地模型；初次加载需要一些时间。", flush=True)
    system = make_system(args.data)
    if args.experiments:
        from experiments import run_experiments
        questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
        if args.data.resolve() != (ROOT / "data" / "knowledge.txt").resolve():
            print("自定义资料：默认问题针对虚构图书馆，请同步修改 data/questions.json。", flush=True)
        path = run_experiments(system, questions, quick=args.quick)
        print(f"真实实验完成，结果目录：{path}")
        return 0
    if args.smoke:
        import numpy as np
        vectors = system.models.embed(["本科生借阅额度", "研讨室预约"])
        if vectors.shape[0] != 2 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise RuntimeError("真实 Embedding 输出无效或未归一化")
        for strategy in ("paragraph", "fixed", "overlap"):
            print(f"{strategy}: {system.build_index(strategy)} 个文本块", flush=True)
        question = "本科生最多同时借几册纸质图书？每册借期多久？"
        record = system.answer(question, max_new_tokens=96)
        if not record["hits"] or not record["answer"]:
            raise RuntimeError("实际问答没有返回检索资料或模型回答")
        ROOT.joinpath("results").mkdir(exist_ok=True)
        output = ROOT / "results" / f"smoke-{uuid.uuid4().hex[:8]}.json"
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("真实模型、向量库及生成流程已执行。答案事实是否正确仍需对照原文检查。")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print("自检记录：", output)
    else:
        record = system.answer(args.question, args.strategy, args.temperature)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModelUnavailableError as exc:
        print(f"\n尚未就绪：{exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\n已取消本次操作。", file=sys.stderr)
        raise SystemExit(130)
