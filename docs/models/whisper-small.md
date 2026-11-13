# billingsmoore/tibetan-asr-nict-tib1-whisper-small — 藏语 ASR（词级质量优先）

> **状态：✅ 本项目实测｜推荐度：★★★★★（能接受约 1.5 倍实时、要求可读转写时首选）**
> 一句话：244M 的 Whisper 微调，**SWER 最低（18.4%）**，但**必须按约 9 秒分块**，否则长句被 448 token 上限截断。

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| HuggingFace | `billingsmoore/tibetan-asr-nict-tib1-whisper-small` |
| 基座 | `openai/whisper-small`（244M） |
| 架构 | Whisper encoder-decoder（`WhisperForConditionalGeneration`），全量微调 |
| 权重 | 967 MB |
| 许可 | **Apache-2.0**（可商用） |
| 训练数据 | 同 wav2vec2 版本（NICT-Tib1，拉萨朗读，20 说话人） |
| 特殊点 | 仓库**只含** `config.json + model.safetensors`，**没有 tokenizer / preprocessor** |

## 2. 实测表现

| 数据 | n | CER | SWER | 备注 |
|---|---:|---:|---:|---|
| NICT 300 条统一子集 | 300 | 8.31% | **18.43%** | 含被截断的长句 |
| 同上 ≤150 字子集 | 245 | **4.82%** | 15.19% | 无截断干扰 |
| 最长 20 条，**不分块** | 20 | 45.65% | 51.92% | 截断灾难 |
| 最长 20 条，**9 s 分块** | 20 | **9.65%** | 19.92% | 推荐配置 |
| 最长 55 条，9 s 分块 | 55 | 14.96% | 25.33% | 参考 214–296 字符 |
| OpenPecha 8 领域 | 160 | 56.21% | 89.60% | 跨域不可用 |

RTF **0.653**（1 小时音频 ≈ 39 分钟），内存 1.35 GB。论文值：CER 11.85% / SWER 20.42%。

## 3. 怎么用

### 3.1 CLI（已内置 9 秒静音分块）

```bash
./.venv/bin/python scripts/predict_asr.py talk.wav --model whisper-small          # 默认 9s 分块
./.venv/bin/python scripts/predict_asr.py talk.wav --model whisper-small --chunk-seconds 0   # 关闭（仅短音频）
```

### 3.2 嵌入代码（注意 processor 要重建）

```python
import torch, librosa
from transformers import (WhisperFeatureExtractor, WhisperTokenizer,
                          WhisperProcessor, WhisperForConditionalGeneration)

MODEL_ID, DEVICE = "billingsmoore/tibetan-asr-nict-tib1-whisper-small", "mps"

processor = WhisperProcessor(                                  # 仓库无 tokenizer，必须重建
    feature_extractor=WhisperFeatureExtractor(feature_size=80),
    tokenizer=WhisperTokenizer.from_pretrained("openai/whisper-small"),
)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16).to(DEVICE).eval()

def transcribe_clip(audio) -> str:                              # audio: 16k mono np.ndarray, ≤9s
    feats = processor(audio, sampling_rate=16000,
                      return_tensors="pt").input_features.to(DEVICE, dtype=torch.float16)
    with torch.inference_mode():
        ids = model.generate(feats, max_new_tokens=440)          # 448 - 4 起始 token
    return processor.batch_decode(ids, skip_special_tokens=True)[0]
```

## 4. ★ 为什么必须 9 秒分块

- 藏文在 Whisper 的 byte-level BPE 下 ≈ **3 token/字符** → 448 位置最多 **≈145 字符**；
- 拉萨朗读 ≈ **16 字符/秒** → 145 ÷ 16 ≈ **9 秒**。

实测（whisper-small，NICT 最长 20 条，参考 243–296 字符）：

| 分块 | CER | SWER |
|---|---:|---:|
| 不分块 | 45.65% | 51.92% |
| 7.5 s | **7.74%** | 19.40% |
| **9 s** | 9.65% | **19.92%** |
| 10.5 s | 19.49% | 29.50% |
| 12 s | 30.38% | 39.82% |
| 16 s | 45.21% | 51.68% |

结论：**7.5–9 s 最优，≥10.5 s 迅速劣化**。切分点放在静音处（`scripts/predict_asr.split_on_silence`，能量 VAD）。分块会让耗时增加约 1.6×（同一 17.7s 片段 2.93s → 4.64s）。

## 5. 关键参数

| 参数 | 建议 | 说明 |
|---|---|---|
| `max_new_tokens` | 440（≤444） | 只传 `max_length` 会触发 `EncoderDecoderCache has no attribute layers` 错误 |
| `num_beams` | 1 | 5 beam 在 pilot 上只把 CER 43.1%→42.5%，慢 3–5 倍 |
| `language` / `task` | 可不传 | `generation_config` 已固定 `bo` / `transcribe`，显式传结果相同 |
| `return_timestamps` | False | 会占用 token 预算 |

## 6. 能力边界与问题

- ✅ 拉萨朗读/新闻；词级质量优于 CTC（SWER 18.4% vs 28.6%）。
- ❌ 跨域（影视/儿童/口语）CER 56%，不可用；❌ 无分块的长音频。
- 错误示例：长句不分块时输出在 145 字符附近戛然而止。

## 7. 结论

**要可读的转写稿 → 选它，并保留默认 9 s 分块。** 相比 wav2vec2：CER 高 1pp、词级好 10pp、慢 28 倍。若只需检索/初稿，用 wav2vec2 更划算。
