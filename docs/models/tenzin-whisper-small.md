# tenzin/whisper-small-tibetan — 混合域藏语 ASR（非商用）

> **状态：✅ 本项目实测｜推荐度：★★☆☆☆（精度不错，但许可是 CC-BY-NC-SA-4.0，禁止商用）**

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| HuggingFace | `tenzin/whisper-small-tibetan` |
| 基座 | `openai/whisper-small`（244M），全量微调 |
| 权重 | 967 MB（仓库另含 last-checkpoint / 训练日志，实际下载按需） |
| 许可 | **CC-BY-NC-SA-4.0（禁止商用、相同方式共享）** |
| 训练数据 | NICT-Tib1 + 西北民大拉萨语料 + OpenPecha + TIBMD（多来源混合） |
| 自报指标 | CER 11.67% / SWER 16.84%（自有 300 条混合域测试切片） |

## 2. 实测表现

| 数据 | n | CER | SWER |
|---|---:|---:|---:|
| NICT 300 条统一子集 | 300 | 8.32% | **17.85%** |
| 同上 ≤150 字子集 | 245 | **4.54%** | **13.60%** |
| OpenPecha 8 领域 | 160 | **24.86%** | 46.85% |

RTF 0.472，内存 1.34 GB。

分领域（OpenPecha）：有声书 14.0%、自然口语 13.9%、播客 14.8%、新闻 20.6%、开示 20.3%、历史 27.0%、电影 34.8%、儿童 **53.6%**。

## 3. 怎么用

```bash
./.venv/bin/python scripts/predict_asr.py talk.wav --model tenzin-small
```

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
mid = "tenzin/whisper-small-tibetan"
processor = WhisperProcessor.from_pretrained(mid)          # 这个仓库自带 tokenizer
model = WhisperForConditionalGeneration.from_pretrained(mid, dtype=torch.float16).to("mps").eval()
```

它的 `generation_config.json` 已带 `forced_decoder_ids`（`<|bo|><|transcribe|><|notimestamps|>`），**不要**再手动传 `language`，否则会冲突。同样受 448 token 限制，需 9 s 分块。

## 4. 结论

- 本项目里 **SWER 最低（17.85%）、短句 CER 最低（4.54%）**，混合域泛化也明显好于 NICT 单域模型（24.9% vs 54–56%）。
- 但 **CC-BY-NC-SA 禁止商用**，且 SA 要求衍生作品同许可——企业项目基本不能用。
- 研究/内部评估可以优先试它；产品化请用 Apache-2.0 的 `whisper-small` 或 `openpecha-general`（后者需先确认许可）。
