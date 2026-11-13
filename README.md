# 藏语 OCR / ASR 开源模型调研与实测

在 **Apple M4 / 16GB（无 CUDA）** 上，对公开可用的藏语 OCR、ASR 模型做端到端评测：选型调研 → 数据集下载 → 推理 → 指标复现 → 文档记录。

## 一句话结论

- **OCR**：唯一达到实用精度的可本地部署开源模型是 **[BDRC/tibetan-ocr](https://huggingface.co/BDRC/tibetan-ocr)（Yigdzin 1，0.8B，Apache-2.0）**——60 页分层样本上 **mean CER 2.50%、median 1.01%、0 灾难页**，约 33 秒/页、2.9GB 内存。Tesseract（8.6MB）CER 53.5%，只能做兜底。
- **ASR**：`wav2vec2-base`（94M，NICT-Tib1 微调）**CER 7.25%、RTF 0.023**；`whisper-small`（244M）**SWER 18.43%、短句 CER 4.82%**；跨域场景则必须用多领域模型（openpecha-general，混合域 CER 15.46%）。
- **关键坑**：Whisper 藏文每字符约 3 个 byte-level token，**单次最多输出 ~145 字符**，长音频必须切分；Yigdzin 仓库默认 `use_cache=false`，开启后 10 倍加速且输出一致。

## 实测速览

### OCR（BDRC 472 页开放基准，60 页分层子集）

| 模型 | 参数 | mean CER | median CER | 灾难页 | 速度 |
|---|---:|---:|---:|---:|---:|
| **BDRC/tibetan-ocr（Yigdzin 1）** | 0.8B | **2.50%** | **1.01%** | 0% | ~33s/页 |
| Tesseract 5 + `bod` | - | 53.52% | 56.06% | 53.3% | 3.3s/页 |

### ASR（NICT-Tib1，统一 300 条 = 35.4 分钟）

| 模型 | 参数 | CER | SWER | ≤150 字 CER | RTF |
|---|---:|---:|---:|---:|---:|
| **wav2vec2-base** | 94M | **7.25%** | 28.56% | 7.35% | **0.023** |
| openpecha-general | ~315M | 7.78% | 29.09% | 7.80% | 0.049 |
| whisper-tiny | 39M | 11.57% | 27.84% | 8.07% | 0.251 |
| whisper-base | 74M | 10.22% | 23.78% | 6.68% | 0.392 |
| **whisper-small** | 244M | 8.31% | 18.43% | 4.82% | 0.653 |
| **tenzin-small**（非商用许可） | 244M | 8.32% | **17.85%** | **4.54%** | 0.472 |

> 词级指标（BoTok-SWER）比 CER 更能反映藏语真实质量；whisper-small / tenzin-small 在词级明显更好。

### 跨域（OpenPecha 8 领域 × 20 条）

| 模型 | CER | SWER |
|---|---:|---:|
| **openpecha-general** | **15.46%** | **38.27%** |
| tenzin-small（混合域） | 24.86% | 46.85% |
| wav2vec2-base（NICT 微调） | 54.02% | 93.12% |
| whisper-small（NICT 微调） | 56.21% | 89.60% |

## 文档

- [调研报告](docs/research.md)：模型/数据集清单、许可、榜单与论文数据、选型理由、方言局限
- [测试方案](docs/test-plan.md)：环境、数据、采样、指标口径、验收标准、风险
- [实测结果](docs/results.md)：完整指标、分技术/分领域结果、失败案例分析、复现命令
- [指标名词表](docs/metrics-glossary.md)：CER / SWER / RTF / 灾难页 / pp / token 等术语的含义、算法与误读陷阱
- **[本地评测 GUI](docs/gui.md)**：`./scripts/gui.sh` → 浏览器里选样例/上传图片或音频、逐字着色 diff、CER/SWER、批量抽检
- **[真实场景手册](docs/scenarios.md)：扫描书/视频转写/万页归档/混合材料/接 API 的完整流程、耗时估算、质量闸门、踩坑清单**
- [模型说明（单模型）](docs/models/README.md)：Yigdzin / wav2vec2-base / whisper-small 等每个模型一篇（架构、参数、精度、坑）
- [模型使用说明](docs/usage.md)：推荐模型怎么调（Yigdzin 必设参数、Whisper 9 秒分块实测）、接入自有数据、故障排查
- [数据集说明](docs/datasets.md)：三套数据的来源/许可、字段字典、原始格式（parquet 内嵌 vs 外链）、落盘位置与目录大小、流水线与校验
- 原始记录：`outputs/{ocr,asr}/<run_id>/`（预测 CSV、逐页/逐条指标、run.json、日志）
- 汇总表：`reports/ocr_results.csv`、`reports/asr_results.csv`、`reports/asr_final_*.csv`、`reports/ocr_per_page.csv`
- 样例：`reports/samples/`

## 快速开始（GUI）

Windows 本轮大模型正式评测可运行项目根目录的 `start_gui.ps1`，在浏览器打开 `http://127.0.0.1:8765`。默认“正式评测结果”页按 OCR 60 页、NICT 300 条、OpenPecha 160 条展示进度、完整指标和原报告基线。选择模型行可查看逐条结果、原图或音频、参考文本及识别差异，并下载 CSV。

正式评测快照保存在 `reports/large_models/`。运行期间由相邻 `LargeModel_Test_Related/publish_results.py --watch` 每20秒同步，完成后可直接浏览落盘结果。页面中的接口错误、待推理、输出截断分别标记。API 服务端模型峰值内存由服务商遥测提供，当前留空，客户端峰值 RSS 单列。Windows 启动脚本使用相邻评测项目的 `.venv`，结果浏览依赖 BoTok 与 rapidfuzz。原有本地模型推理功能仍需安装原项目的模型依赖。

2026-11-13 本轮8组正式评测已全部完成，模型和覆盖率见 [catalog.json](reports/large_models/catalog.json)。Kimi 已按用户要求退出本轮展示。OpenPecha 的 Qwen-Audio 采用用户指定的 `qwen-audio-3.1-asr-flash-streaming`，通过专属工作空间的 WebSocket 接口执行实时 ASR。指标、样本哈希和覆盖率核对记录见 [verification.json](reports/large_models/verification.json)。

```bash
./scripts/gui.sh            # → http://127.0.0.1:8765
```

左侧选样例（自带标准答案）或上传自己的藏文图片/语音，右侧看逐字着色 diff 与 CER/SWER。

## 快速开始（命令行）

```bash
# 1) 环境（Python 3.12；不要把 venv 放在 iCloud 同步目录，本机用 ~/.venvs 软链接）
python3 -m venv ~/.venvs/tibetocr && ln -s ~/.venvs/tibetocr .venv
./.venv/bin/pip install -r requirements.txt

# 2) 数据（OCR 472 页基准；NICT-Tib1 test；OpenPecha 分领域抽样）
./.venv/bin/python scripts/prepare_ocr_data.py
./.venv/bin/python scripts/prepare_asr_data.py

# 3) 用自己的数据跑（PDF / 图片目录 / 音频 / 视频）
./.venv/bin/python scripts/pipeline.py ocr book.pdf shots/ --out outputs/myocr --offline
./.venv/bin/python scripts/pipeline.py asr lecture.mp4 rec/ --out outputs/myasr --model whisper-small --offline
# --offline：模型已缓存时加，跳过 Hub 联网校验（否则可能卡 5 分钟）

# 4) 在基准上评测（OCR 60 页；ASR 6 模型 × NICT + 4 模型 × OpenPecha）
./scripts/run_all_ocr.sh
./scripts/run_all_asr.sh
./.venv/bin/python scripts/summarize_asr.py --dataset both
```

## 目录

```
scripts/            可复现脚本（数据准备 / 校验 / 推理 / 打分 / 汇总 / 样例）
docs/               调研报告、测试方案、实测结果
data/               数据集与音频（gitignore，约 0.9GB）
outputs/            原始预测与运行记录（gitignore，测试记录）
reports/            汇总表与样例
tessdata/           Tesseract bod 模型（gitignore）
```

## 硬件与耗时

- M4 MPS：Yigdzin 整页 OCR 约 30–35s/页；whisper-small 约 1.5 倍实时；CTC 模型 20–40 倍实时。
- 本次全流程（含下载）约 2.5 小时；下载量约 5GB（模型 4GB + 数据集 1GB）。

## 许可提示

- 可商用：Yigdzin（Apache-2.0）、billingsmoore 系列（Apache-2.0）、Tesseract（Apache-2.0）、NICT-Tib1（CC-BY-4.0）、BDRC 基准（CC0）、OpenPecha 基准（MIT）。
- 仅研究：`tenzin/whisper-small-tibetan`（CC-BY-NC-SA-4.0）。
- 许可待确认：`openpecha/general_stt_base_model`（未声明）。
