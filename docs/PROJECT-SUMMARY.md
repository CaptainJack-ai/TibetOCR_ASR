# 藏语 OCR / ASR 开源模型评测项目 · 总结

> 版本：2026-11-14 ｜ 仓库：`CaptainJack-ai/TibetOCR_ASR` ｜ 全部数字可在本仓库复现
> 相关文档：[调研](docs/research.md) · [测试方案](docs/test-plan.md) · [实测结果](docs/results.md) · [指标名词表](docs/metrics-glossary.md) · [数据集](docs/datasets.md) · [模型使用](docs/usage.md) · [场景手册](docs/scenarios.md) · [单模型](docs/models/README.md) · [GUI](docs/gui.md)

---

## 1. 项目一句话

在**消费级硬件（Apple M4 / 16GB，无 CUDA）**上，把藏语 OCR 与 ASR 的**开源模型选型、数据准备、批量推理、指标评测、可视化检查、生产落地**全链路跑通，并给出可直接使用的结论：**OCR 选 Yigdzin（0.8B）；ASR 快选 wav2vec2-base、准选 whisper-small（必须 9 秒分块）**。

---

## 2. 为什么做（问题定义）

| 现实问题 | 本项目给出的答案 |
|---|---|
| 藏文 OCR 到底能不能用开源方案做？ | 能，但**只有专门微调的模型可用**；通用 VLM（Gemini/Claude/GPT/Qwen-VL）CER 39%–100%，不可用 |
| 藏语 ASR 有哪些可选？精度多少？ | 6 个模型全量实测：CER 7.25%–11.57%（域内），跨域退化到 54%–56% |
| 报告里的 CER 能信吗？ | 所有指标用 BDRC 官方协议（botok 规范化 + 封顶 1.0），与公开榜单同口径，且独立重算验证过 |
| 拿到模型后怎么用在自己的书上/录音上？ | CLI 单文件 → 断点续跑批处理 → GUI 逐条检查，三层入口 |
| 会踩什么坑？ | 记录 9 个真实故障（`use_cache`、`mm_token_type_ids`、448 token、iCloud venv、git 代理…） |

---

## 3. 核心结论（可直接引用）

### 3.1 OCR

| 模型 | 参数 | 许可 | mean CER | median CER | 灾难页 | 速度 | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| **BDRC/tibetan-ocr（Yigdzin 1）** | 0.8B | Apache-2.0 | **2.50%** | **1.01%** | **0%** | 33s/页 | **生产首选** |
| Tesseract 5 + `bod` | — | Apache-2.0 | 53.52% | 56.06% | 53.3% | 3.3s/页 | 仅作兜底 |

- 公开榜单（472 页全量）同模型 mean CER 3.21%，排名 **1/46**；第二名 MITRA 0.8B 为 7.32%。
- 60 页中 **59 页优于 Tesseract**，整体差约 21 倍。
- 分材料：数字字体 0.61% / 金属活字 0.33% / 木刻 1.23% / 现代 1.34% / 手稿 3.61% / **打字机 12.77%**。

### 3.2 ASR（NICT-Tib1，统一 300 条 = 35.4 分钟）

| 模型 | 参数 | CER | SWER | ≤150 字 CER | RTF | 结论 |
|---|---:|---:|---:|---:|---:|---|
| **wav2vec2-base** | 94M | **7.25%** | 28.56% | 7.35% | **0.023** | 快、稳、CER 最低 |
| openpecha-general | ~315M | 7.78% | 29.09% | 7.80% | 0.049 | 混合域最好，许可待确认 |
| whisper-tiny | 39M | 11.57% | 27.84% | 8.07% | 0.251 | 跑通/极限轻量 |
| whisper-base | 74M | 10.22% | 23.78% | 6.68% | 0.392 | 折中 |
| **whisper-small** | 244M | 8.31% | **18.43%** | 4.82% | 0.653 | **词级质量首选** |
| tenzin-small | 244M | 8.32% | **17.85%** | **4.54%** | 0.472 | 精度最好，但 **CC-BY-NC 非商用** |

### 3.3 跨域（OpenPecha 8 领域 × 20 条）

| 模型 | CER | SWER |
|---|---:|---:|
| **openpecha-general** | **15.46%** | **38.27%** |
| tenzin-small | 24.86% | 46.85% |
| wav2vec2-base（NICT 微调） | 54.02% | 93.12% |
| whisper-small（NICT 微调） | 56.21% | 89.60% |

**结论：域匹配比模型结构更重要**——NICT 上最强的模型换域后 CER 涨 7 倍。

### 3.4 云端大模型对比（协作补充，8 组）

| 任务 | 模型 | CER | SWER | 观察 |
|---|---|---:|---:|---|
| OCR 60 页 | gpt-6.1-sol（Codex 订阅） | 35.76% | — | 灾难页 36.7%，76s/页（含 CLI 启动） |
| OCR 60 页 | deepseek-flash | 63.20% | — | 5 页输出截断 |
| ASR NICT | qwen3.8-omni-flash | **15.74%** | 31.35% | 大模型里最好，仍逊于本地 7.25% |
| ASR NICT | doubao-seed-2-1-lite | 59.22% | 93.24% | — |
| ASR NICT | qwen-audio-turbo | 96.51% | 99.71% | **137/300 条输出中文或乱码** |
| ASR OpenPecha | qwen-audio-3.1-asr-flash-streaming | 95.94% | 99.05% | **51/160 条输出中文译文** |

> 注：95% 级别的数字不是"模型差"，而是**该接口没在做藏文转写**（输出语言不符）。详见 `reports/large_models/runs/*/predictions_scored.csv` 的 `transcription` 列。

---

## 4. 三个关键发现（项目最有价值的部分）

### 4.1 Yigdzin 仓库自带 `use_cache=false` → 单页 >300 秒

- 现象：按模型卡示例跑，M4 上单页 5 分钟无输出。
- 原因：`generation_config.json` 里 `use_cache: false`，解码复杂度 O(n²)。
- 修复：显式 `use_cache=True`，**输出逐字节一致**，120 token 从 83.4s → 7.5s（11×），单页降到 ~30s。

### 4.2 Whisper 藏文有 448 token 硬上限 → 单次最多 ~145 字符

- 藏文 UTF-8 3 字节 + byte-level BPE ≈ **3.03 token/字符**；Whisper 解码器 448 位置 → **约 145 字符**。
- 拉萨朗读约 **16 字符/秒** → **约 9 秒**音频。
- 实测（whisper-small，最长 20 条，参考 243–296 字符）：

| 分块 | CER | SWER |
|---|---:|---:|
| 不分块 | **45.65%** | 51.92% |
| 7.5 s | **7.74%** | 19.40% |
| **9 s** | 9.65% | **19.92%** |
| 10.5 s | 19.49% | 29.50% |
| 12 s | 30.38% | 39.82% |
| 16 s | 45.21% | 51.68% |

→ **7.5–9 秒是安全区，≥10.5 秒迅速劣化**。GUI 与 `pipeline.py` 已默认 9 秒静音分块。

### 4.3 CER 会系统性低估藏语错误 → 必须同时报 SWER

- NICT 参考文本平均每词 **4.57 字符** → `SWER ≈ CER × 4.57`（实测 7.25% × 4.57 ≈ 33%，实测 28.56%）。
- 最小例子：参考 55 字符，只丢 1 个字符 → **CER 3.64%**，但 **SWER 22.22%**（9 词错 2 个）。
- 这与论文（IEEE Access 2026）的核心论点一致；本项目所有 ASR 表格都同时给两个指标。

---

## 5. 交付物清单

### 5.1 可复现脚本（`scripts/`，19 个）

| 类别 | 脚本 | 作用 |
|---|---|---|
| 数据 | `prepare_ocr_data.py` / `prepare_asr_data.py` / `verify_data.py` | 下载 + 解包 + 抽样；29 项完整性校验 |
| 评测 | `run_ocr.py` / `run_asr.py` / `score_ocr.py` / `score_asr.py` / `summarize_asr.py` | 批量推理、打分、统一子集对比 |
| 生产 | **`pipeline.py`** | PDF/目录/音视频 → 文本；断点续跑 + 质量闸门 + 报告 |
| 单文件 | `predict_ocr.py` / `predict_asr.py` | 单图/单音频，Whisper 自动 9s 分块 |
| 实验 | `exp_whisper_chunking.py` | 分块长度敏感性实验（上文表格的来源） |
| 工具 | `make_samples.py` / `fix_git_proxy.sh` / `gui.sh` | 样例导出、代理修复、GUI 启动 |

### 5.2 本地评测 GUI（`gui/`，零新增依赖）

- 标准库 `http.server` + 原生 HTML/CSS/JS；`./scripts/gui.sh` → `http://127.0.0.1:8765`
- 四个 Tab：**正式评测结果**（含云端大模型对比）/ OCR 评测 / ASR 评测 / 批量抽检
- 核心能力：选样例（自带答案）或上传自己的图/音视频 → **四色逐字 diff**（正确/替换/漏识别/多识别）+ CER/SWER/RTF
- 指标全部走 `scripts/common.py`，与 `reports/*.csv` **逐位一致**（已验证差值 ≤0.0001）

### 5.3 文档（10 篇，约 3,300 行）

| 文档 | 内容 |
|---|---|
| `docs/research.md` | 模型/数据集调研，榜单 46 个系统对比，许可与方言限制 |
| `docs/test-plan.md` | 环境、数据、采样、指标口径、验收标准、风险 |
| `docs/results.md` | 全部实测数字、分技术/分领域、失败案例 |
| `docs/metrics-glossary.md` | CER/SWER/RTF/灾难页等 41 个术语的定义与误读陷阱 |
| `docs/datasets.md` | 三套数据格式（parquet 内嵌 vs 外链）、字段字典、磁盘布局 |
| `docs/usage.md` | 每个推荐模型的必设参数、分块实验、故障排查 9 条 |
| `docs/scenarios.md` | 扫描书 / 视频转写 / 万页归档 / 混合材料 / 接 API 的完整流程与耗时估算 |
| `docs/models/*.md` | 9 个单模型文档（架构/参数/代码/坑） |
| `docs/gui.md` | GUI 用法、颜色语义、API 列表 |
| `README.md` | 结论速览 + 三层快速开始 |

---

## 6. 数据（三套，共 960MB）

| 用途 | 数据集 | 规模 | 许可 |
|---|---|---|---|
| OCR 评测 | BDRC Tibetan OCR Benchmark | 472 页（Uchen 238 / Ume 193；手稿 209 / 数字字体 133 / 木刻 91…） | CC0 |
| ASR 主评测 | NICT-Tib1 test | 1,547 条 / 185.5 分钟 / 3 说话人（与训练不重叠） | CC-BY-4.0 |
| ASR 泛化 | OpenPecha 语音基准 | 全量 8,367 条；本项目抽 160 条 / 8.94 分钟 / 8 领域 | MIT |

抽样固定种子 42，`sample_ids.json` 可复现；60 页 = 手稿 18 + 数字字体 14 + 木刻 14 + 现代 5 + 金属活字 5 + 打字机 4。

---

## 7. 工程侧的关键取舍

| 决定 | 理由 |
|---|---|
| 推理全局串行（1 把锁） | MPS 共享，并发反而更慢（实测单页 30s → 36s） |
| 指标统一走 `common.py` | 避免 GUI / 报告 / 论文三方口径漂移 |
| GUI 用标准库而非 Gradio | 零依赖、离线可用、排版与 diff 着色完全可控 |
| 断点续跑（每页写一行 JSONL） | 万页级任务可中断、可追溯；失败项自动重试 |
| venv 移出 iCloud 目录 | iCloud 同步曾产生 **7,136 个 `* 2.py`** 副本，导致 `import transformers` 阻塞 >120s；迁移后 **0.6s** |
| 模型不进仓库 | 权重 6.5GB 在 `~/.cache/huggingface`；`data/` 由脚本重下 |

---

## 8. 已知边界（诚实声明）

1. **只有卫藏（拉萨）口音**：安多/康巴、古典唱诵无开源模型可测，结论不可外推。
2. **OCR 无版面分析**：多栏/图文混排需先切块；插图、破损页会失败。
3. **打字机页面 CER 12.8%**：必须全量人工校对。
4. **OpenPecha 上的 15.46% 可能是乐观值**：与 openpecha 训练数据同源，存在重叠风险。
5. **许可**：`tenzin/whisper-small-tibetan` 为 CC-BY-NC-SA（禁商用）；`openpecha/general_stt_base_model` **未声明许可**，商用前必须确认。
6. **云端大模型结果可复核但不可重跑**：`reports/large_models/` 有逐条输出与参考，可独立重算指标（已验证）；但仓库没有调用代码、prompt、API key，无法重跑推理。

---

## 9. 一页速查

```bash
# 环境
python3 -m venv ~/.venvs/tibetocr && ln -s ~/.venvs/tibetocr .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/prepare_ocr_data.py && ./.venv/bin/python scripts/prepare_asr_data.py
./.venv/bin/python scripts/verify_data.py

# GUI（推荐入口）
./scripts/gui.sh

# 自己的数据
./.venv/bin/python scripts/pipeline.py ocr book.pdf shots/ --out outputs/myocr --offline
./.venv/bin/python scripts/pipeline.py asr talk.mp4 rec/ --out outputs/myasr --model whisper-small --offline

# 复现基准评测
./scripts/run_all_ocr.sh && ./scripts/run_all_asr.sh
```

| 我要… | 用这个 |
|---|---|
| 藏文图片转文字 | Yigdzin 1（0.8B，Apache-2.0，33s/页） |
| 藏语语音批量转写（快） | wav2vec2-base（RTF 0.023，43× 实时） |
| 藏语语音转写（可读） | whisper-small（9s 分块，SWER 18.4%） |
| 影视/儿童/口语音频 | openpecha-general（先确认许可） |
| 逐页/逐条看错在哪 | `./scripts/gui.sh` |
