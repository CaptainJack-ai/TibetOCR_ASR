# billingsmoore/tibetan-asr-nict-tib1-wav2vec2-base — 藏语 ASR（CTC）

> **状态：✅ 本项目实测｜推荐度：★★★★★（拉萨朗读/新闻，追求速度与稳定时首选）**
> 一句话：94M 的 CTC 模型，**CER 最低（7.25%）**、**43 倍实时**、输出确定，但不加标点、跨域不可用。

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| HuggingFace | `billingsmoore/tibetan-asr-nict-tib1-wav2vec2-base` |
| 基座 | `facebook/wav2vec2-base` |
| 架构 | Wav2Vec2 + CTC 头（`Wav2Vec2ForCTC`），全量微调 |
| 参数量 / 权重 | 94M / 378 MB |
| 许可 | **Apache-2.0**（可商用） |
| 训练数据 | NICT-Tib1：拉萨藏语朗读新闻，20 位说话人，15,099 条 |
| 评测集 | 同仓库 test split：1,547 条 / 3 位说话人（与训练说话人不重叠） |
| 论文 | J. Moore et al., *Evaluating Tibetan ASR With Segmented Word Error Rate*, IEEE Access 2026 |
| 词表 | 86（藏文字符级） |

## 2. 实测表现

| 数据 | n | CER | SWER | RTF | 内存 |
|---|---:|---:|---:|---:|---:|
| NICT 全量 test | 1,547 | 7.90% | 29.81% | 0.023 | 1.30 GB |
| NICT 300 条统一子集 | 300 | **7.25%** | 28.56% | 0.023 | 1.30 GB |
| 同上 ≤150 字子集 | 245 | 7.35% | 28.74% | - | - |
| OpenPecha 8 领域 | 160 | **54.02%** | 93.12% | 0.027 | 1.30 GB |

对照论文：CER 7.45% / SWER 27.47%（本机复现基本一致，略好）。

## 3. 怎么用

```python
import torch, librosa
from transformers import AutoProcessor, AutoModelForCTC

MODEL_ID = "billingsmoore/tibetan-asr-nict-tib1-wav2vec2-base"
DEVICE = "mps"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()

def transcribe(wav_path: str) -> str:
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000,
                       return_tensors="pt").input_values.to(DEVICE, dtype=torch.float16)
    with torch.inference_mode():
        logits = model(inputs).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]
```

CLI（批处理）：

```bash
./.venv/bin/python scripts/predict_asr.py audio/*.wav --model wav2vec2-base --json
```

## 4. 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| 采样率 | 16 kHz，单声道 | 不重采样会显著掉点 |
| 解码 | `argmax` + CTC | 无采样、无 beam，输出确定 |
| dtype | fp16（MPS/CUDA） | 本机实测可用 |
| 分块 | 不需要（默认关闭） | 实测 ≤19s 单条正常；很长音频建议按静音切到 ≤30s |
| 标点 | 无 | 后处理需另做标点恢复 |

## 5. 能力边界

- ✅ 拉萨（卫藏）口音、朗读/新闻/清晰语音。
- ❌ 影视对白、儿童语音、口语闲聊、安多/康巴方言、强噪声：跨域 CER 54%，不可用。
- ❌ 需要标点/大小写/数字规范化：CTC 只输出藏文字符串。

## 6. 已知问题

| 现象 | 说明 |
|---|---|
| 数字/专名错 | 如 `ཉིས་སྟོང་`(2000) → `ཉེས་སྟོང་`，属字符级同音/形近混淆 |
| CER 好看但 SWER 高 | 7.25% vs 28.56%：错一个字符常毁掉整词（见指标名词表 §2.3） |
| 长音频内存增长 | 位置/激活随长度增长，建议切分 |

## 7. 结论

**要快、要稳、要省内存 → 选它。** 1 小时音频约 83 秒处理完，适合大批量转写、字幕初稿、检索建库；成品字幕仍需校对（SWER 29%）。跨域场景请改 `openpecha-general`。
