# openpecha/general_stt_base_model — 多领域藏语 ASR

> **状态：✅ 本项目实测｜推荐度：★★★☆☆（混合域最好，但许可未声明，商用前必须确认）**

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| HuggingFace | `openpecha/general_stt_base_model` |
| 架构 | Wav2Vec2-large 规模 + CTC（`Wav2Vec2ForCTC`），约 315M |
| 权重 | 1.26 GB |
| 许可 | **未声明（模型卡是自动生成的空模板）** |
| 来源 | OpenPecha / Monlam AI 生态，面向多领域（新闻、开示、口语、有声书等） |
| 模型卡 | 无实质信息，无训练数据/指标说明 |

## 2. 实测表现

| 数据 | n | CER | SWER | RTF | 内存 |
|---|---:|---:|---:|---:|---:|
| NICT 全量 test | 1,547 | 8.28% | 30.19% | 0.049 | 1.75 GB |
| NICT 300 条统一子集 | 300 | 7.78% | 29.09% | 0.049 | 1.75 GB |
| OpenPecha 8 领域 | 160 | **15.46%** | **38.27%** | 0.060 | 1.81 GB |

分领域 CER：有声书 **6.8%**、自然口语 8.3%、新闻 11.2%、播客 11.4%、开示 13.1%、电影 18.9%、历史 20.3%、儿童 **33.7%**。

> ⚠️ 该模型与 OpenPecha 基准**同源**，可能存在训练集重叠 → 15.46% 是**乐观上界**，真实跨域泛化需要独立测试集。

## 3. 怎么用

```bash
./.venv/bin/python scripts/predict_asr.py talk.wav --model openpecha-general
```

```python
from transformers import AutoProcessor, AutoModelForCTC
import torch
mid = "openpecha/general_stt_base_model"
processor = AutoProcessor.from_pretrained(mid)
model = AutoModelForCTC.from_pretrained(mid, dtype=torch.float16).to("mps").eval()
# 之后与 wav2vec2-base 完全相同：processor(audio)... → argmax → batch_decode
```

CTC 模型默认不分块；长音频建议按静音切到 ≤30s。

## 4. 结论

- 它是本项目里**唯一在 8 个领域都保持可用（6.8%–33.7%）**的模型；NICT 模型换域直接 54%–56%。
- 速度也很快（20 倍实时）。
- **最大风险是许可**：训练数据来源不明、许可未声明，商业化前必须联系 OpenPecha/Monlam 确认；否则只能内部评估。
- 若业务音频是影视/儿童/口语，先用它做基线，再用自己的数据微调（微调数据可用 `openpecha/tibetan_voice`、`billingsmoore/nict-tib1`）。
