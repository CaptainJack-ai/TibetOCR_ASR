# 测试方案

## 1. 目标

在本地可复现地评测藏语 OCR / ASR 开源模型，回答三个问题：

1. 哪个模型在当前硬件上"能用"（精度 + 速度 + 内存）？
2. 实际精度与论文/榜单差多少？
3. 失败模式是什么，生产部署需要注意什么？

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| 机器 | Apple M4（10 核 GPU），16GB 统一内存 |
| 系统 | macOS 27.2 (arm64) |
| Python | 3.12（独立 `.venv`） |
| 深度学习框架 | torch 2.15.0（MPS 后端）、transformers 5.22.2 |
| 其他 | datasets 5.2.0、botok 1.1.7、rapidfuzz 3.14.6、librosa 1.0.0、jiwer 4.0.0 |
| 外部工具 | Tesseract 5.5.2 + `tessdata_best/bod.traineddata`、ffmpeg |

所有推理均在单机完成，无 CUDA、无外部 API。

## 3. 目录结构

```
scripts/
  common.py              # 路径、日志、藏文规范化、CER/SWER
  prepare_ocr_data.py    # BDRC 基准 -> 图片 + GT + 官方 scorer
  prepare_asr_data.py    # NICT test + OpenPecha 领域抽样
  run_ocr.py             # Yigdzin / Tesseract 推理
  run_asr.py             # 6 个 ASR 模型推理
  score_ocr.py           # 按 BDRC 协议打分
  score_asr.py           # 单次运行 CER / SWER / RTF
  summarize_asr.py       # 统一子集上的最终对比表
  make_samples.py        # 导出图片/音频 + 参考 + 预测样例
  verify_data.py         # 数据完整性校验（id/采样率/时长/与官方 GT 一致性）
  pipeline.py            # 端到端批处理（PDF/目录/音视频，断点续跑 + 质量闸门 + 报告）
  predict_ocr.py / predict_asr.py   # 单文件推理
  vendor/dry_logits_processor.py  # Yigdzin 官方 DRY 抗重复
data/        # 数据集与音频（不入库）
outputs/     # 原始预测 CSV、运行元数据、日志（测试记录）
reports/     # 汇总表与样例
```

复现：

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/prepare_ocr_data.py
./.venv/bin/python scripts/prepare_asr_data.py
./.venv/bin/python scripts/verify_data.py
./.venv/bin/python scripts/run_ocr.py --engine yigdzin --split sample --use-dry
./.venv/bin/python scripts/run_ocr.py --engine tesseract --split sample
./.venv/bin/python scripts/score_ocr.py
./.venv/bin/python scripts/run_asr.py --model wav2vec2-base --dataset nict
./.venv/bin/python scripts/score_asr.py
./.venv/bin/python scripts/summarize_asr.py --dataset both
./.venv/bin/python scripts/make_samples.py --ocr-run <run_id> --asr-run <run_id>
```

## 4. 数据准备

> 字段字典、原始格式与磁盘布局详见 [数据集说明](./datasets.md)。

### 4.1 OCR

- 数据源：`BDRC/tibetan-ocr-benchmark` test 分片（472 页，121MB parquet）。
- 导出 `data/ocr/images/{id}.png`、`ground_truth.csv`、`metadata.csv`；同时下载官方 `compute_cer.py` 作为口径参考。
- 抽样：
  - **pilot**：6 页，每种技术 1 页（短、清晰优先），用于跑通链路与测速；
  - **sample**：60 页，按技术分层配额（手稿 18、数字字体 14、木刻 14、现代 5、金属活字 5、打字机 4），层内兼顾 Uchen/Ume；
  - **all**：472 页全量（可选）。
- 抽样固定随机种子 42，`data/ocr/sample_ids.json` 记录。

### 4.2 ASR

- **NICT-Tib1 test**：1,547 条、185 分钟、3 位说话人。落盘 16k 单声道 WAV + `metadata.csv`。
  - 默认按说话人分层抽 **300 条**（每位 100 条）做模型对比；需要对论文复现时可跑全量。
- **OpenPecha 混合域**：8 个领域各抽 20 条（1000→20，儿童语音 1367→20），共 160 条、8.9 分钟；优先选 grade≥2 的转写。需要全量时加 `--openpecha-per-dept 2000`。

## 5. 评测协议

### 5.1 OCR

| 项目 | 设置 |
|---|---|
| Yigdzin | prompt `Extract all Tibetan text. Preserve line breaks.`；bf16 + MPS + SDPA；`mm_token_type_ids` 清零；greedy；`max_new_tokens=2048`；`use_cache=True`；DRY（mult=0.8, base=1.75, allowed_length=12） |
| Tesseract | `-l bod --psm 6`，tessdata_best，OEM 默认 |
| 输入 | 整页图（不切行），与榜单口径一致 |
| 打分 | BDRC 官方协议：botok 规范化 → 去空白 → 合并连续 tsheg → 去占位符 KOBIS → 字符级 Levenshtein / 参考长度，单页封顶 1.0 |
| 指标 | mean CER（主）、median CER、CER>50% 的"灾难页"占比、每页耗时、失败页数 |

> 说明：官方榜单的 mean CER 按全部 472 页计算（缺失页记 1.0）。本测试的 60 页子集只在该子集上统计，因此**绝对值不可与榜单直接比较**，只用于模型间对比与流程验证。

### 5.2 ASR

| 项目 | 设置 |
|---|---|
| 模型 | whisper-tiny / whisper-base / wav2vec2-base / whisper-small / tenzin-small / openpecha-general |
| 解码 | CTC：argmax；Whisper：greedy（`num_beams=1`），可选 beam=5；`max_new_tokens=440`（Whisper 上限 444） |
| 精度 | 16k 单声道；MPS fp16（Whisper）/ fp16（CTC） |
| 指标 | CER（字符级，botok 规范化）、BoTok-SWER（botok 分词后 WER）、RTF = 推理时间/音频时长、峰值内存 |
| 分层 | 全部样本 + 参考长度 ≤150 字符子集（规避 Whisper 448 token 上限）；OpenPecha 额外按领域拆分 |

### 5.3 指标定义

> 术语的完整解释、示例与常见误读见 [指标名词表](./metrics-glossary.md)。

- **CER** = Levenshtein(预测字符, 参考字符) / len(参考字符)，单条封顶 1.0；批次取算术平均。
- **BoTok-SWER** = Levenshtein(预测词序列, 参考词序列) / len(参考词序列)，词由 botok 分词得到（去掉标点）。
- **RTF** < 1 表示快于实时。
- **灾难页**：单页 CER > 50%。

## 6. 验收标准

| 项 | 标准 |
|---|---|
| OCR 链路 | Yigdzin 60 页子集 mean CER ≤ 6%；Tesseract 明显更差（>40%） |
| OCR 速度 | Yigdzin 单页 ≤ 30s（M4 MPS） |
| ASR 链路 | wav2vec2-base 在 300 条 NICT 子集上 CER 与论文（7.45%）差距 ≤ 3pp |
| ASR 速度 | CTC 模型 RTF < 0.1；Whisper-small RTF < 1.5 |
| 可复现 | 所有脚本可重跑，输出 CSV/日志/汇总表齐全 |

## 7. 风险与回退

| 风险 | 处理 |
|---|---|
| transformers 5.x 与 Whisper 长格式生成的 cache bug（`EncoderDecoderCache.layers`） | Whisper 一律显式传 `max_new_tokens`，走短格式路径 |
| Yigdzin `use_cache=false` 导致极慢 | 强制 `use_cache=True`（已验证输出一致，~11x 加速） |
| MPS bf16 数值问题 | 回退 fp16 / CPU fp32；实测 bf16 正常 |
| MPS 内存不足 | batch=1；必要时把 `longest_edge` 从 1280 降到 960 |
| Whisper 448 token 上限截断长句 | 同时报告全量与 ≤150 字符子集；生产建议按静音/标点切分 |
| 下载受限 | 全部仓库已确认公开可访问；HF 直连 200 |

## 8. 测试记录

- 每次运行生成 `outputs/{ocr,asr}/<run_id>/`：
  - `predictions.csv`：模型原始输出
  - `details.csv` / `details_scored.csv`（OCR）：逐页耗时、CER、技术/文字类型
  - `predictions_scored.csv`（ASR）：逐条 CER、SWER、RTF
  - `run.json`：模型、设备、dtype、参数、总耗时、峰值内存、错误数
- 汇总：`reports/ocr_results.csv`、`reports/asr_results.csv`
- 样例：`reports/samples/`（图片/音频 + 参考 + 预测）
