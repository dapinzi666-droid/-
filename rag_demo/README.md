# 任务2：基础RAG课堂演示项目

依据作业 Excel 的「分组汇报要求」，为「程序演示工程师」选择 **A：基础 RAG 系统**。

**先读 [新手操作与分工说明](docs/新手操作与分工说明.md)**，再按下面步骤运行。不需要 API 密钥，不需要训练模型。

## 这个项目完成什么

读取自带的虚构图书馆资料，用真实 BGE 模型生成向量，存入嵌入式 Chroma，检索后交给真实 Qwen 小模型生成中文回答。

- 段落切分 vs 带 overlap 的切分；另加相同块大小的无重叠对照。
- 同一问题、完全相同的检索上下文，比较 temperature=0 和 1.0，多次采样。
- Notebook 展示检索来源和真实回答，导出 JSON、CSV 和可离线打开的 HTML 对比报告。

原课程代码依赖论文PDF、API密钥和部分教师电脑路径，本项目替换这些输入依赖，保留课堂的检索增强生成逻辑。未加入选题B/C的额外功能。

## 先安装，再运行

使用64位 Python 3.11或3.12；首次需联网下载依赖和约1.1GB模型文件，建议16GB内存和至少8GB可用空间。

**Windows**：先安装Python，双击 `setup_windows.bat`，在项目目录打开PowerShell：

```powershell
.\.venv\Scripts\python.exe run_demo.py --prepare
.\.venv\Scripts\python.exe run_demo.py --smoke
.\.venv\Scripts\python.exe run_demo.py --experiments
```

然后双击 `start_notebook_windows.bat`，打开 `task2_rag_demo.ipynb`。

**macOS/Linux**：

```bash
bash setup_unix.sh
./.venv/bin/python run_demo.py --prepare
./.venv/bin/python run_demo.py --smoke
./.venv/bin/python run_demo.py --experiments
bash start_notebook_unix.sh
```

快速预演可使用 `--experiments --quick`；正式报告使用完整实验。先课前运行并检查答案，再在现场运行一次问答，切勿上台才下载模型。

## 文件导航

| 文件 | 用途 |
|---|---|
| `task2_rag_demo.ipynb` | 面向新手的逐格运行和现场展示 |
| `rag_core.py` | 文档加载、三种切分、模型、Chroma检索、生成 |
| `experiments.py` | 公平对照及真实报告导出 |
| `run_demo.py` | 环境检查、模型下载、真实自检、命令行实验 |
| `data/knowledge.txt` | 虚构教学资料，不是真实校规 |
| `data/questions.json` | 4类实验问题与人工核对参考点，不把参考答案送给生成器 |
| `docs/` | 安装、分工、讲稿、答辩和实验记录模板 |
| `models/` | 首次下载后存放模型，压缩包不包含权重 |
| `results/` | 实际运行后生成报告；交付包没有假模型答案 |
| `tests/` | 数据切分、公平参数、报告导出和真实向量库的检查 |

## 验证边界

本项目已做的检查和未通过的检查详见 [验证记录](docs/验证记录.md)。当前开发云端拒绝模型下载站访问（HTTP 403），因此**尚未完成两个预训练模型的实际问答和效果验证**。不能把依赖安装、自动测试通过当成课堂演示已就绪。必须在可以下载模型的电脑上完成 `--prepare`、`--smoke` 和完整实验。

模型下载失败会明确停止，没有固定答案后备。相似度不代表正确率，overlap不保证更好，temperature=0也不能消除幻觉。回答达到长度上限会提醒；需要时把双方的 `max_new_tokens` 一起提高后重做公平比较。

## 开发检查

```bash
python -m unittest discover -s tests -v
```

未安装依赖时，纯逻辑测试可运行，Chroma集成检查会明确跳过；安装依赖后应执行全部检查。模型端到端检查单独使用 `--smoke`。
