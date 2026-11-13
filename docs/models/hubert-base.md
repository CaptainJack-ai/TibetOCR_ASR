# billingsmoore/tibetan-asr-nict-tib1-hubert-base — 藏语 ASR（CTC）

> **状态：📄 仅调研（本项目未跑）｜推荐度：★★☆☆☆（词级误差最大，通常被 wav2vec2 取代）**

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| HuggingFace | `billingsmoore/tibetan-asr-nict-tib1-hubert-base` |
| 基座 | `facebook/hubert-base-ls960`（94M） |
| 架构 | HuBERT + CTC 头 |
| 权重 / 许可 | 378 MB / Apache-2.0 |
| 训练数据 | NICT-Tib1（与 wav2vec2-base 同数据、同论文） |

## 2. 论文指标（未在本机复现）

| 指标 | HuBERT-base | 对照 wav2vec2-base |
|---|---:|---:|
| CER | 13.52% | **7.45%** |
| BoTok-SWER | 44.77% | **27.47%** |

## 3. 怎么用

与 wav2vec2 完全一致（同为 `Wav2Vec2ForCTC` 接口）：

```bash
./.venv/bin/python scripts/predict_asr.py talk.wav \
  --model billingsmoore/tibetan-asr-nict-tib1-hubert-base
```

## 4. 结论

在同数据同架构量级下，hubert 的 CER 是 wav2vec2 的 1.8 倍、SWER 的 1.6 倍，**没有选它的理由**，除非做架构对比实验。
