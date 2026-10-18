"""运行真实模型实验，并生成可离线查看的原始数据与报告。"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from uuid import uuid4

from rag_core import ROOT


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _context_hash(messages: list[dict]) -> str:
    # 对整个提示求摘要，确保温度实验连问题和系统指令也相同。
    data = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def render_record(record: dict) -> str:
    """所有模型输出与资料均转义，不执行其中可能出现的 HTML。"""
    e = _escape
    title = {"chunk": "切分对照", "temperature": "温度对照"}.get(record.get("experiment"), "现场问答")
    reused = "；复用切分实验的同一次生成" if record.get("generation_reused") else ""
    info = record.get("generation_info") or {}
    warning = (
        "<p class='warning'><strong>回答达到长度上限，可能未说完，需核对或调高 "
        "max_new_tokens 重跑。</strong>多点综合问题可在 Notebook 中传入 160。</p>"
        if info.get("possibly_truncated", False) else ""
    )
    chunk_size = record.get("chunk_size")
    chunk_size = "自然段（不限定长度）" if chunk_size is None else chunk_size
    hits = []
    for index, hit in enumerate(record.get("hits", []), 1):
        page = f"第 {hit['page']} 页" if hit.get("page") else "无页码"
        hits.append(
            f"<section><h4>[S{index}] {e(hit.get('source', ''))} · {e(page)}</h4>"
            f"<p>相似度：{e(hit.get('score', ''))}；"
            f"字符区间：[{e(hit.get('start', ''))}, {e(hit.get('end', ''))})</p>"
            f"<pre>{e(hit.get('text', ''))}</pre></section>"
        )
    prompt = json.dumps(record.get("messages", []), ensure_ascii=False, indent=2)
    versions = json.dumps(record.get("model_versions", {}), ensure_ascii=False, indent=2)
    return (
        f"<article><h3>{e(title)} · {e(record.get('question_id', ''))} · "
        f"{e(record.get('kind', ''))}</h3><p><strong>{e(record.get('question', ''))}</strong></p>"
        f"<p>切分：{e(record.get('strategy', ''))}；温度：{e(record.get('temperature', ''))}；"
        f"种子：{e(record.get('seed', ''))}；Top-K：{e(record.get('top_k', ''))}；"
        f"生成上限：{e(record.get('max_new_tokens', 96))} tokens；"
        f"耗时：{e(record.get('seconds', ''))} 秒{e(reused)}</p>"
        f"<p>块大小：{e(chunk_size)}；重叠：{e(record.get('overlap', ''))}；"
        f"上下文字符数：{e(record.get('context_characters', ''))}；"
        f"输出 tokens：{e(info.get('output_tokens', ''))}。{e(record.get('chunk_note', ''))}</p>"
        f"{warning}<pre class='answer'>{e(record.get('answer', ''))}</pre>"
        f"<details><summary>完整检索资料（{len(hits)} 条）</summary>{''.join(hits)}</details>"
        f"<details><summary>实际提示、摘要与模型版本</summary><p>context_sha256："
        f"{e(record.get('context_sha256', ''))}</p><pre>{e(prompt)}</pre>"
        f"<pre>{e(versions)}</pre></details></article>"
    )


def _html_report(records: list[dict], created: str) -> str:
    style = """
body{font-family:system-ui,sans-serif;max-width:1080px;margin:32px auto;padding:0 18px;
line-height:1.7;color:#213047;background:#f5f7fb}article{background:white;padding:22px;
margin:20px 0;border:1px solid #dbe2ee;border-radius:12px}pre{white-space:pre-wrap;
overflow-wrap:anywhere;font:inherit;background:#f0f3f9;padding:14px;border-radius:8px}
.answer{border-left:4px solid #4763bd}summary{cursor:pointer}h1,h2,h3{line-height:1.4}
.warning{background:#fff2d6;border:2px solid #bf7600;padding:12px;color:#744700}
section{border-top:1px solid #dbe2ee;margin-top:12px}small{color:#53657d}
"""
    sections = []
    for group, label in (("chunk", "一、切分对照"), ("temperature", "二、温度对照")):
        sections.append(f"<h2>{label}</h2>")
        sections.extend(render_record(record) for record in records if record["experiment"] == group)
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>基础 RAG 真实实验报告</title><style>{style}</style><body>"
        f"<h1>基础 RAG 真实实验报告</h1><small>生成时间（UTC）：{_escape(created)}</small>"
        "<p>本项目默认知识库为虚构教学资料，不代表真实学校规定。模型使用真实的本地 BGE、"
        "Chroma 和 Qwen；小模型可能答错、遗漏或编造，引用编号也需要人工核对。</p>"
        "<p>相似度是检索排序分数，不是正确率或可信度。切分或温度调整不保证结果存在差异，"
        "也不保证某组更好。温度为 0 使用贪心解码，温度为 1 采样；温度实验固定完整提示，"
        "重复种子用于观察随机性。切分实验均固定 Top-K=2、温度 0、种子 42、生成上限 96 tokens。</p>"
        "<p>温度 0、种子 42 的基线复用对应切分实验的同一次回答，其耗时也属于该次调用；"
        "其余温度记录重新生成。原始 messages、检索片段和模型版本保存在 records.json 中。</p>"
        + "".join(sections) + "</body></html>"
    )


def run_experiments(system, questions: list[dict], quick: bool = False,
                    output_root: Path = ROOT / "results") -> Path:
    """调用真实系统；失败直接抛出，不保存成功报告，也不伪造答案。"""
    selected = [item for item in questions if item["id"] == "q2"] if quick else questions
    if not selected:
        raise ValueError("没有可运行的问题；快速实验需要 id 为 q2 的综合问题。")
    records = []
    completed, total = 0, len(selected) * 7

    def generate(item, strategy, temperature, seed, experiment, hits=None):
        nonlocal completed
        print(f"[{completed + 1}/{total}] {item['id']} {experiment}："
              f"{strategy}，temperature={temperature}，seed={seed}", flush=True)
        result = system.answer(item["question"], strategy=strategy,
                               temperature=temperature, seed=seed, top_k=2,
                               max_new_tokens=96, hits=hits)
        result = dict(result)
        result.update(experiment=experiment, question_id=item["id"], kind=item.get("kind", ""),
                      max_new_tokens=96, generation_reused=False,
                      context_sha256=_context_hash(result["messages"]))
        completed += 1
        records.append(result)
        return result

    for item in selected:
        baseline = None
        for strategy in ("paragraph", "fixed", "overlap"):
            baseline = generate(item, strategy, 0, 42, "chunk")
        # 基线不重复生成；后续全部传入同一份检索结果，禁止重新检索。
        repeated = dict(baseline, experiment="temperature", generation_reused=True)
        records.append(repeated)
        for temperature, seed in ((0, 43), (1, 42), (1, 43), (1, 44)):
            result = generate(item, "overlap", temperature, seed, "temperature", baseline["hits"])
            if result["context_sha256"] != baseline["context_sha256"]:
                raise RuntimeError("温度实验提示发生变化，停止导出，请检查上下文复用。")

    created = datetime.now(timezone.utc)
    folder = Path(output_root) / (created.strftime("%Y%m%dT%H%M%S%fZ_") + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["experiment", "question_id", "kind", "question", "strategy", "chunk_size", "overlap",
              "temperature", "seed", "top_k", "max_new_tokens", "context_sha256", "seconds",
              "answer", "sources", "backend", "model_versions", "generation_reused",
              "output_tokens", "possibly_truncated", "context_characters", "chunk_note"]
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            info = record.get("generation_info") or {}
            row.update(output_tokens=info.get("output_tokens", ""),
                       possibly_truncated=info.get("possibly_truncated", False))
            row["sources"] = json.dumps(
                [{key: hit.get(key) for key in ("source", "page", "score", "id")} for hit in record["hits"]],
                ensure_ascii=False)
            row["model_versions"] = json.dumps(record["model_versions"], ensure_ascii=False)
            writer.writerow(row)
    (folder / "report.html").write_text(_html_report(records, created.isoformat()), encoding="utf-8")
    print(f"实验完成：{completed} 次真实生成，{len(records)} 条记录。报告：{folder / 'report.html'}", flush=True)
    return folder
