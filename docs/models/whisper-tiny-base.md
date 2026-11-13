# billingsmoore whisper-tiny / whisper-base — 轻量藏语 ASR

> **状态：✅ 本项目实测｜推荐度：★★★☆☆（跑通、极限轻量、或需要更快的 Whisper 时用）**

## 1. 基本信息

| 项目 | whisper-tiny | whisper-base |
|---|---|---|
| HuggingFace | `billingsmoore/tibetan-asr-nict-tib1-whisper-tiny` | `billingsmoore/tibetan-asr-nict-tib1-whisper-base` |
| 基座 | `openai/whisper-tiny`（39M） | `openai/whisper-base`（74M） |
| 权重 | 151 MB | 290 MB |
| 许可 | Apache-2.0 | Apache-2.0 |
| 训练数据 | NICT-Tib1（拉萨朗读，20 说话人） | 同左 |
| 共同限制 | 仓库无 tokenizer，需要 `openai/whisper-*` 重建 processor；**同样受 448 token 限制，需 9s 分块** | |

## 2. 实测表现（NICT 300 条统一子集）

| 模型 | CER | SWER | ≤150 字 CER | ≤150 字 SWER | RTF | 内存 |
|---|---:|---:|---:|---:|---:|---:|
| whisper-tiny | 11.57% | 27.84% | 8.07% | 24.62% | **0.251** | 1.15 GB |
| whisper-base | 10.22% | 23.78% | 6.68% | 20.64% | 0.392 | 1.18 GB |
| （对照）whisper-small | 8.31% | 18.43% | 4.82% | 15.19% | 0.653 | 1.35 GB |

论文值：tiny 15.60% / 29.75%，base 14.17% / 26.00%。

## 3. 怎么用

```bash
./.venv/bin/python scripts/predict_asr.py talk.wav --model whisper-tiny     # 默认 9s 分块
./.venv/bin/python scripts/predict_asr.py talk.wav --model whisper-base
```

代码与 [whisper-small](./whisper-small.md#32-嵌入代码注意-processor-要重建) 完全相同，只换 `MODEL_ID` 和 `WhisperTokenizer.from_pretrained("openai/whisper-tiny|base")`。

## 4. 结论

- **whisper-tiny**：4 倍实时，适合"先看能不能用"、端侧/低资源，CER 11.6% 只能做草稿。
- **whisper-base**：tiny 与 small 之间的折中，比 small 快 1.7×、CER 差约 2pp。
- 两者都**必须分块**；不分块时和 small 一样会被截断（长句 CER 45%+）。
