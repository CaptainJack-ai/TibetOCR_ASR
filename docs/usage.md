# 模型使用说明（How to use）

> 本文件面向"我要用它处理自己的数据"：每个推荐模型怎么调、参数怎么设、能期待什么精度、会遇到什么坑。
> 选型依据见 [调研报告](./research.md)，数字出处见 [实测结果](./results.md)，术语见 [指标名词表](./metrics-glossary.md)。

---

## 0. 先选模型

| 你的任务 | 用这个 | 一句话理由 | 本文档 |
|---|---|---|---|
| 藏文图片/扫描件 → 文字 | **BDRC/tibetan-ocr（Yigdzin 1，0.8B）** | 唯一达到实用精度的开源藏文 OCR，Apache-2.0 | **§2 详细** |
| 藏语语音 → 文字，要快、要批量 | **wav2vec2-base（NICT-Tib1）** | CER 最低 7.25%，43 倍实时，94M | **§3 详细** |
| 藏语语音 → 文字，要好读、词级要准 | **whisper-small（NICT-Tib1）** | SWER 最低 18.4%，**必须按 9 秒分块** | **§4 详细** |
| 语音是影视/儿童/口语等混合场景 | openpecha-general | 混合域 CER 15.5%（NICT 模型 54%） | §5 |
| 只要个能跑的对照基线 | Tesseract + `bod` | 8.6MB、3s/页，但 CER 53% | §5 |
| 研究复现（非商用） | tenzin/whisper-small-tibetan | SWER 17.9%，但 **CC-BY-NC-SA** | §5 |

> 单模型细节（架构、参数量、代码、坑）另见 [docs/models/](./models/README.md)，每个模型一篇。
> 手上已经有真实材料（PDF/录音/视频），想要端到端流程和耗时估算，直接看 [真实场景手册](./scenarios.md)。

**不要用**：通用多模态大模型（Gemini / Claude / GPT / Qwen / InternVL）做藏文 OCR——榜单 CER 39%–100%；通用 Whisper / MMS 直接做藏语 ASR——输出乱码。必须用上表里针对藏文微调过的权重。

---

## 1. 通用准备

### 1.1 环境

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt     # torch 2.15 + transformers 5.22 + librosa + botok …
```

- Python 3.12；macOS/Linux 均可，Apple 芯片走 MPS，有 CUDA 就把 `device` 换成 `"cuda"`。
- 需要 `transformers>=5.15`（Yigdzin 的 `paddleocr_vl` 架构在 5.15 才原生支持）。

### 1.2 模型下载位置

模型走 HuggingFace 默认缓存（**不在项目目录**）：

```bash
~/.cache/huggingface/hub/blobs/…      # 真实权重（按 sha256 存储）
export HF_HOME=/path/to/big-disk      # 想换位置就设这个环境变量
```

| 模型 | 权重体积 |
|---|---:|
| Yigdzin（OCR） | 1.63 GB |
| openpecha-general | 1.26 GB |
| whisper-small / tenzin-small | 各 0.97 GB |
| wav2vec2-base | 378 MB |
| whisper-base / whisper-tiny | 290 MB / 151 MB |

### 1.3 四个现成工具

| 工具 | 用途 |
|---|---|
| `scripts/predict_ocr.py` | **单张/多张图片** → 文本（含 DRY、可选温度重采样） |
| `scripts/predict_asr.py` | **单条/多条音频** → 文本（Whisper 自动按 9s 静音分块） |
| `scripts/run_ocr.py` / `run_asr.py` | 在数据集上批量跑（评测用，带 `run.json` 记录） |
| `scripts/score_*.py` | 有参考文本时打分 |

最小示例：

```bash
./.venv/bin/python scripts/predict_ocr.py mypage.jpg
./.venv/bin/python scripts/predict_asr.py mytalk.m4a --model wav2vec2-base
./.venv/bin/python scripts/predict_asr.py long_talk.wav --model whisper-small --json
```

---

## 2. 【推荐 ①】BDRC/tibetan-ocr（Yigdzin 1）— 藏文 OCR

### 2.1 能力边界

| 能做 | 不能做 |
|---|---|
| 整页或行图 → 藏文文本（Uchen / Ume，现代印刷 / 木刻 / 手抄 / 金属活字） | 版面分析（分栏、表格、图注定位） |
| 保留换行；识别 pecha 版式与正字法缩写 | 插图、照片、非藏文（会输出垃圾或空） |
| 单页一次前向，无需切行 | 破损/极模糊页面的可靠识别 |

**它只负责"给图出字"**。多栏排版、插图混排的页面，建议先用版面检测切块再逐块 OCR（BDRC 另有一套 layout 模型，本项目未涉及）。

### 2.2 两种用法

**A. 命令行（推荐日常用）**

```bash
./.venv/bin/python scripts/predict_ocr.py data/ocr/images/0187.png            # 贪婪 + DRY
./.venv/bin/python scripts/predict_ocr.py scan.jpg --temperature 0.4 --num-samples 3   # 循环兜底
./.venv/bin/python scripts/predict_ocr.py a.jpg b.jpg --json > out.json        # 批处理
```

**B. 嵌入自己的代码（最小实现，已实测）**

```python
import sys, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessorList

sys.path.insert(0, "scripts/vendor")            # 仓库自带的 DRY 实现
from dry_logits_processor import make_hf_dry_processor

MODEL_ID = "BDRC/tibetan-ocr"
PROMPT   = "Extract all Tibetan text. Preserve line breaks."
DEVICE   = "mps"                                 # 或 "cuda" / "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()

def ocr(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: (v.to(DEVICE, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(DEVICE))
              for k, v in inputs.items()}
    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])   # ① 必须

    with torch.inference_mode():
        out = model.generate(
            **inputs, do_sample=False, max_new_tokens=2048,
            use_cache=True,                                              # ② 必须
            logits_processor=LogitsProcessorList([make_hf_dry_processor(  # ③ 强烈建议
                prompt_len=inputs["input_ids"].shape[1],
                multiplier=0.8, base=1.75, allowed_length=12)]),
        )
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]
```

### 2.3 三个"不设就出错"的参数

| 参数 | 必须做什么 | 不做的后果 | 怎么发现的 |
|---|---|---|---|
| `use_cache` | **显式 `True`** | 仓库 `generation_config.json` 里是 `false` → 解码 O(n²)，**单页 >300s**（实测 83.4s → 7.5s，11×） | 首次跑 pilot 卡住 5 分钟无输出 |
| `mm_token_type_ids` | **清零**（`torch.zeros_like`） | 按 grid 位置编码走，结构化页面**漏行/串行**；官方称 CER 可恶化到 ~62% | 模型卡"Position regime"一节 |
| `logits_processor`（DRY） | 建议开，`0.8 / 1.75 / 12` | 少数页面进入重复循环，整页报废 | 实测 60 页 0 灾难页 |

> 精度：输出必须 `torch.bfloat16`（fp16 在本项目未验证）；`attn_implementation="sdpa"` 更快但非必需。

### 2.4 解码策略：正常用贪婪，异常页用温度重采样

| 场景 | 设置 | 说明 |
|---|---|---|
| 默认 | `do_sample=False` + DRY | 确定、可复现，本项目 60 页全部用它 |
| 输出出现循环（同一 12-gram 反复出现） | `temperature=0.4, num_samples=3`，取重复度最低者 | 官方推荐：DRY 触发次数 ≥100 的页面才需要（约占 2%） |
| `max_new_tokens` | 2048（本项目） | 该模型的分词器约 **1.26 字符/token**：本项目 60 页最长 1,562 token，未截断；但**全 472 页中有 4 页 >2,048 token**（最长 2,451，对应 3,084 字符）。跑全量时建议设 3,072–4,096 |

`predict_ocr.py --temperature 0.4` 已内置"取重复度最低样本"的选择逻辑（12-gram 重复计数）。

### 2.5 输入图像的处理建议

先用 Pillow 检查；模型自己会 resize（patch 14 / merge 2，即每 28px 一个视觉 token）。

| 情况 | 建议 |
|---|---|
| 手机拍照 | 转正（0/90/180/270）+ 去透视，模型不做旋转矫正 |
| 长条 pecha 页（如 3730×623） | 直接整条输入即可（本项目最快 14s/页） |
| 超长卷（>10000px 宽） | 建议切成 2–3 段分别识别再拼接，否则视觉 token 过多、耗时暴涨 |
| 二值/灰度扫描 | 可 `convert("RGB")`，无需转彩色 |
| 极低对比度 | 先做自适应二值化或 CLAHE，能显著降低 CER |
| 多栏/带插图 | 先版面切块（每块单独 OCR），否则阅读顺序会乱 |

### 2.6 输出后处理

```python
text = text.replace("\r\n", "\n").strip()
# 模型按行输出；若要段落，可把单换行合并：
text = text.replace("\n", " ")          # 谨慎：会丢失原分行信息
```

- **保留 `།` / `༎` / `་`**：这些是句读与音节分隔，后处理不要删。
- 空行通常表示分栏/段落，建议保留。
- 若下游要分词，用 `botok`（本项目 `scripts/common.py` 里的 `word_tokenize`）。

### 2.7 质量闸门（没有 GT 时怎么判断这页要不要人工）

按本项目 60 页的经验：

| 信号 | 阈值 | 处理 |
|---|---|---|
| 输出为空或 < 10 字符 | - | 视为失败页，人工/重拍 |
| `loop_score`（12-gram 最大重复次数）≥ 4 | 正常页通常 ≤ 2（经文重复也就 2–3） | 温度重采样 3 次取最优 |
| 字符密度异常 | 本数据集每页中位 861 字（386–3,084） | 远低于同批页面 → 可能是空白页/裁切错误 |
| 单页耗时 > 3× 中位数 | - | 通常是超长页或陷入循环 |

如果有一小批**人工转写过的页**，直接算 CER 建立基线更可靠（`scripts/score_ocr.py` 可直接复用）。

### 2.8 性能与容量规划（M4，bf16，batch=1）

| 指标 | 实测 |
|---|---|
| 单页耗时 | 中位 **33.3s**，均值 36.4s，P75 42.7s，最大 114.7s |
| 行条图（1506×344） | ~14s |
| 吞吐 | ≈ **109 页/小时**（单进程）；1,000 页 ≈ 9.2 小时 |
| 显存/内存峰值 | **≈2.9 GB** |
| 与耗时最相关的因素 | **输出字符数**（相关系数 0.64），不是图像面积（0.14）→ 每 100 字约 3.4s |
| 分词密度 | 约 **1.26 字符/token**（区别于 Whisper 的 3 token/字符）→ 容量规划按字符数算即可 |
| 模型加载 | 首次 ~6s（之后常驻） |

> **不要并发跑多个 Yigdzin 进程**：本项目在 OCR 与 ASR 并发时，单页从 ~30s 涨到 36.4s（共享 GPU）。要吞吐就串行批处理，或换更强硬件。

### 2.9 质量预期（决定人工复核比例）

| 材料 | mean CER | 是否需要人工 |
|---|---:|---|
| 数字字体 | 0.61% | 抽检即可 |
| 金属活字 | 0.33% | 抽检即可 |
| 木刻 | 1.23% | 抽检 |
| 现代印刷 | 1.34% | 抽检 |
| 手稿 | 3.61% | 建议过一遍 |
| 打字机 | 12.77% | **必须校对**（有整段漏行） |

综合：60 页 mean CER **2.50%**、median **1.01%**、灾难页 **0%**。按经验，CER <2% 的文本用于检索/统计基本够用；用于出版级校对仍需人工过一遍。

### 2.10 常见失败与对策

| 现象 | 原因 | 对策 |
|---|---|---|
| 整段漏行、行序错乱 | 长页面 + 多栏 | 切块（按栏/页面上中下三段）后逐块识别 |
| 结尾突然中断 | `max_new_tokens` 用尽 | 提高预算（4096）；或分块 |
| 同一句反复出现 | 重复循环 | DRY（已默认）+ `--temperature 0.4 --num-samples 3` |
| 输出里混入拉丁字母/数字乱码 | 页面含非藏文 | 后处理剔除，或先做文字体系分类（BDRC 另有分类器） |
| 插图区域输出乱码 | 模型不做版面分析 | 先切掉插图区域 |

---

## 3. 【推荐 ②】wav2vec2-base（NICT-Tib1）— 低延迟、批量 ASR

### 3.1 什么时候选它

- ✅ 音频是**卫藏（拉萨）口音的朗读/新闻/清晰语音**；要**快**（43 倍实时）、要**省内存**（1.3GB）、要**确定性输出**（CTC 无采样）。
- ❌ 影视对白、儿童语音、方言（安多/康巴）、嘈杂现场：实测跨域 CER **54%**，不可用。
- ❌ 需要标点：CTC 输出无标点，需另做标点恢复。

### 3.2 最小实现（已实测）

```python
import torch, librosa
from transformers import AutoProcessor, AutoModelForCTC

MODEL_ID = "billingsmoore/tibetan-asr-nict-tib1-wav2vec2-base"
DEVICE = "mps"                          # 或 "cuda" / "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(
    MODEL_ID, dtype=torch.float16).to(DEVICE).eval()

def transcribe(wav_path: str) -> str:
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)     # 任意格式 → 16k 单声道
    inputs = processor(audio, sampling_rate=16000,
                       return_tensors="pt").input_values.to(DEVICE, dtype=torch.float16)
    with torch.inference_mode():
        logits = model(inputs).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]
```

或直接用：

```bash
./.venv/bin/python scripts/predict_asr.py audio/*.wav --model wav2vec2-base --json
```

### 3.3 输入要求

| 项 | 要求 / 建议 |
|---|---|
| 采样率 | 16 kHz（librosa 自动重采样；其它采样率不重采样会明显掉点） |
| 声道 | 单声道（`mono=True`） |
| 单条时长 | 本项目实测 ≤19s 正常。**很长音频请按静音切到 ≤30s** 再逐段解码（省内存、避免位置编码超限） |
| 前端 | 无需 VAD/降噪，但明显噪声会显著掉点 |

### 3.4 性能与质量

| 指标 | 实测（M4） |
|---|---|
| RTF | **0.023** → **43 倍实时**；1 小时音频 ≈ 83 秒 |
| 内存 | ≈1.3 GB |
| NICT 300 条 | CER **7.25%** / SWER 28.56%（论文 7.45% / 27.47%） |
| ≤150 字子集 | CER 7.35% / SWER 28.74%（长句无截断优势，说明它对长音频友好） |
| 跨域（OpenPecha） | CER 54.02% → **不要跨域使用** |

> CER 7% 但 SWER 29%：字符看着不错，词级仍有约 3/10 出错（原因见 [指标名词表 §2.3](./metrics-glossary.md)）。用于检索/关键词可以，用于字幕需要校对。

---

## 4. 【推荐 ③】whisper-small（NICT-Tib1）— 词级质量优先

### 4.1 什么时候选它

- ✅ 要**可读性/词级准确率**（SWER 18.4%，比 wav2vec2 好 10pp）；能接受约 1.5 倍实时。
- ✅ 需要在同一套框架里做多语言/多任务（Whisper 生态）。
- ❌ 不能接受 40 分钟处理 1 小时音频；❌ 超长音频未分块（**必须分块**，见 4.3）。

### 4.2 最小实现（注意 processor 要重建）

```python
import torch, librosa
from transformers import (WhisperFeatureExtractor, WhisperTokenizer,
                          WhisperProcessor, WhisperForConditionalGeneration)

MODEL_ID = "billingsmoore/tibetan-asr-nict-tib1-whisper-small"
DEVICE = "mps"

# ⚠️ 这些仓库只存了 config.json + model.safetensors，没有 tokenizer/preprocessor
processor = WhisperProcessor(
    feature_extractor=WhisperFeatureExtractor(feature_size=80),
    tokenizer=WhisperTokenizer.from_pretrained("openai/whisper-small"),
)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16).to(DEVICE).eval()

def transcribe_9s(wav_path: str) -> str:
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    feats = processor(audio, sampling_rate=16000,
                      return_tensors="pt").input_features.to(DEVICE, dtype=torch.float16)
    with torch.inference_mode():
        ids = model.generate(feats, max_new_tokens=440)     # 448 上限 - 4 个起始 token
    return processor.batch_decode(ids, skip_special_tokens=True)[0]
```

> **不要**只传 `max_length=448`：transformers 5.22 的长格式生成路径会抛
> `AttributeError: 'EncoderDecoderCache' object has no attribute 'layers'`。
> 显式给 `max_new_tokens` 才走短格式路径（本项目踩过）。

更省事的方式（**已内置 9 秒静音分块**）：

```bash
./.venv/bin/python scripts/predict_asr.py long.wav --model whisper-small
./.venv/bin/python scripts/predict_asr.py long.wav --model whisper-small --chunk-seconds 0   # 关闭分块
```

### 4.3 ★ 关键：必须按约 9 秒分块（否则精度腰斩）

**为什么**：藏文每字符 ≈3 个 byte-level token，Whisper 解码器上限 448 位置 → 单次最多约 **145 字符**；而拉萨朗读约 **16 字符/秒** → **145 ÷ 16 ≈ 9 秒**。

实测（whisper-small，NICT-Tib1 最长的 20 条，参考 243–296 字符）：

| 分块长度 | CER | SWER | median CER |
|---|---:|---:|---:|
| 不分块 | **45.65%** | 51.92% | 45.63% |
| 7.5 s | **7.74%** | 19.40% | 4.40% |
| **9 s** | 9.65% | **19.92%** | 8.55% |
| 10.5 s | 19.49% | 29.50% | 19.27% |
| 12 s | 30.38% | 39.82% | 31.72% |
| 16 s | 45.21% | 51.68% | 45.58% |

在更长的 55 条（参考 214–296 字符）上验证 9 s：CER **14.96%** / SWER **25.33%** / median 10.27% —— 比不分块的 ~46% 好 3 倍。

**结论**：`7.5–9 s` 是最优区间，**≥10.5 s 迅速劣化**。默认用 **9 s**（略多上下文、更稳），切分点放在静音处。

```python
from scripts.predict_asr import split_on_silence   # 静音感知切分（能量 VAD）
spans = split_on_silence(audio, max_seconds=9)     # → [(0.0, 4.1), (4.5, 9.2), ...]
```

注意事项：

- 切分点落在**停顿**上，不要硬切词中；本项目用 `librosa.effects.split(top_db=32)`。
- 分块会**丢失块间上下文**（跨块的句子可能断开），对字幕友好、对篇章级连贯性有轻微影响。
- 分块会**增加耗时**（同一 17.7s 片段：不分块 2.93s、9 s 分块 4.64s，约 1.6×），因为每块都要单独过一遍编码器——这是换取「不截断」的必要代价。

### 4.4 解码参数

| 参数 | 建议 | 实测影响 |
|---|---|---|
| `max_new_tokens` | 440（≤444） | 越小越容易被截断；必须配合分块 |
| `num_beams` | 1（默认） | pilot 上 5 beam 只把 CER 43.1%→42.5%，慢 3–5 倍，不划算 |
| `language` / `task` | 可不传（`generation_config` 已是 `bo`/`transcribe`） | 传了结果一致（实测完全相同） |
| `return_timestamps` | False | 开时间戳会占用 token 预算，不建议 |

### 4.5 性能与质量

| 指标 | 实测（M4） |
|---|---|
| RTF | **0.653** → 1 小时音频 ≈ 39 分钟；300 条（35.4 分钟）≈ 23 分钟 |
| 内存 | ≈1.35 GB |
| NICT 300 条 | CER 8.31% / SWER **18.43%** |
| ≤150 字子集 | CER **4.82%** / SWER 15.19% |
| 长句 + 9s 分块 | CER 9.65%（20 条）/ 14.96%（55 条） |
| 跨域（OpenPecha） | CER 56.21% → 不要跨域使用 |

**许可**：`billingsmoore/*` 为 Apache-2.0，可商用。若想试 `tenzin/whisper-small-tibetan`（SWER 17.85%、CER 8.32%，略好），注意它是 **CC-BY-NC-SA-4.0 非商用**，且需要 `forced_decoder_ids`（其 `generation_config` 已带）。

---

## 5. 其他模型（简要）

| 模型 | 何时用 | 关键数字 | 注意 |
|---|---|---|---|
| **openpecha/general_stt_base_model** | 音频是影视/儿童/口语等**混合域** | NICT CER 7.78% / SWER 29.09%；OpenPecha **15.46% / 38.27%**；RTF 0.049；1.26GB | 与 OpenPecha 数据可能重叠 → 15.46% 偏乐观；**许可未声明**，商用前必须确认 |
| **tenzin/whisper-small-tibetan** | 研究对比，词级最优 | CER 8.32% / SWER **17.85%**；短句 4.54% | **非商用**（CC-BY-NC-SA-4.0）；同样受 448 token 限制，需 9s 分块 |
| **whisper-base** | 速度与质量折中 | CER 10.22% / SWER 23.78%；RTF 0.392 | 同样需要分块；比 small 快 1.7×、CER 差 2pp |
| **whisper-tiny** | 极限轻量（39M） | CER 11.57% / SWER 27.84%；RTF 0.251 | 明显更适合"先跑通"而不是"要质量" |
| **hubert-base** | 备选 CTC | 论文 13.52% / 44.77% | 词级误差最大，本项目未实测 |
| **Tesseract 5 + `bod`** | 极轻量 OCR 兜底/预筛 | CER 53.52%；3.3s/页；8.6MB | 只适合判断"这页有没有字"，不能生产 |
| **MITRA-BDRC 0.8B** | Yigdzin 的备选 OCR | 榜单 CER 7.32% | 同样 0.8B，精度差 2.3 倍，作为对照 |
| **PaddleOCR-VL 基座 / DONUT** | 不建议 | 37% / 28% | 未针对藏文微调 |

---

## 6. 接入自己的数据

### 6.1 图片 / PDF

```bash
# PDF → 每页 PNG（200–300 dpi 足够；太高只会变慢）
pdftoppm -r 250 -png book.pdf data/mine/page

# 批量 OCR + 保存
for f in data/mine/page/*.png; do
  ./.venv/bin/python scripts/predict_ocr.py "$f" --json >> out.jsonl
done
```

- 扫描件先确认方向正确；错向会让 CER 直接接近 100%。
- 每页尽量保持"一页一图"，不要让一页跨两张图。

### 6.2 音频 / 视频

```bash
# 视频取音轨 → 16k 单声道 wav
ffmpeg -i talk.mp4 -vn -ac 1 -ar 16000 talk.wav

# 直接喂 m4a/mp3/wav 都可以（librosa 会解码）
./.venv/bin/python scripts/predict_asr.py talk.wav --model wav2vec2-base --json
```

- 长录音（>30s）用 `whisper-small` 时**必须**保留默认 9s 分块；用 CTC 模型建议先按静音切到 ≤30s。
- 麦克风质量差时，先过一遍降噪（如 `ffmpeg -af afftdn`）再识别。

### 6.3 一条完整的批处理模板

```python
# 用仓库里的类，避免重复实现
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
from predict_asr import transcribe_file, split_on_silence
from run_asr import Transcriber

pipe = Transcriber("billingsmoore/tibetan-asr-nict-tib1-whisper-small",
                   device="mps", dtype="fp16", beams=1, language=None, max_new_tokens=440)

out = []
for wav in sorted(Path("data/mine").glob("*.wav")):
    text, meta = transcribe_file(pipe, wav, chunk_seconds=9)
    out.append({"file": wav.name, "text": text, **meta})
Path("transcripts.jsonl").write_text(
    "\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")
```

---

## 7. 上线前的质量监控建议

| 环节 | 做什么 |
|---|---|
| 采样基线 | 人工转写 50–100 条同分布样本，用 `scripts/score_*.py` 算出你自己的 CER/SWER 基线（**别人的数字不可直接套用**） |
| OCR 自动闸门 | `loop_score ≥ 4` 或 输出 <10 字 → 重采样/人工；字符密度明显异常 → 复查原图 |
| ASR 自动闸门 | 输出时长/字符比异常（正常 ≈16 字/秒）、输出为空、非藏文字符占比高 → 标记人工 |
| 人工复核比例 | 参考本项目：数字字体/木刻 CER<1.5% → 抽检 5%；手稿 ~3.6% → 20%；打字机 12.8% → 全量校对 |
| 回归 | 每次换模型/参数，保留 `run.json` + 同一批样本的分数，避免"感觉变好了" |

---

## 8. 故障排查

| 症状 | 原因 | 修复 |
|---|---|---|
| OCR 单页跑 5 分钟以上 | `use_cache` 为 false | `generate(..., use_cache=True)` |
| OCR 漏行/行序乱 | `mm_token_type_ids` 未清零 | `inputs["mm_token_type_ids"] = torch.zeros_like(...)` |
| OCR 输出同一句循环 | 重复循环 | 开 DRY；再不行 `temperature=0.4, n=3` 取最优 |
| OCR 输出突然中断 | `max_new_tokens` 不足 | 调大到 4096，或分块 |
| `OSError: Can't load feature extractor for 'billingsmoore/...'` | 仓库只有权重，无 tokenizer | 用 `openai/whisper-{tiny,base,small}` 重建 processor（§4.2） |
| `AttributeError: 'EncoderDecoderCache' object has no attribute 'layers'` | 只给了 `max_length`，走了长格式生成 | 传 `max_new_tokens`（≤444） |
| ASR 输出在 145 字符左右截断 | 448 token 上限 | 9 s 分块（§4.3） |
| MPS 报内存/OOM | 长音频或超长图 | 降低 chunk、缩小图像、batch=1 |
| 同一批音频换机器 CER 变差 | 采样率没重采样/精度不同 | 固定 `sr=16000, mono=True`，固定 dtype |

---

## 9. 资源速查

| 模型 | 权重 | 内存峰值 | 速度 | 许可 | 推荐场景 |
|---|---:|---:|---|---|---|
| BDRC/tibetan-ocr | 1.63GB | 2.9GB | 33s/页（~109 页/h） | Apache-2.0 | 藏文 OCR 生产 |
| wav2vec2-base | 378MB | 1.3GB | RTF 0.023（43×） | Apache-2.0 | ASR 低延迟/批量 |
| whisper-small | 967MB | 1.35GB | RTF 0.653（1.5×） | Apache-2.0 | ASR 词级质量 |
| tenzin-small | 967MB | 1.34GB | RTF 0.472 | **CC-BY-NC-SA** | 研究 |
| whisper-base | 290MB | 1.18GB | RTF 0.392 | Apache-2.0 | 折中 |
| whisper-tiny | 151MB | 1.15GB | RTF 0.251 | Apache-2.0 | 跑通/极限轻量 |
| openpecha-general | 1.26GB | 1.75GB | RTF 0.049（20×） | 未声明 | 混合域 |
| Tesseract bod | 8.6MB | <0.2GB | 3.3s/页 | Apache-2.0 | 兜底基线 |

---

## 10. 一句话推荐

- **单纯要藏文文字**：`predict_ocr.py` + Yigdzin，串行批处理，按材料设人工复核比例。
- **单纯要藏语转写且音量很大**：`predict_asr.py --model wav2vec2-base`，43 倍实时，但只限拉萨朗读类音频。
- **要可读的转写稿**：`predict_asr.py --model whisper-small`（默认 9 s 分块，别关），词级错误少一半。
- **数据是影视/儿童/口语**：先测 `openpecha-general`，并解决许可问题再上线。
