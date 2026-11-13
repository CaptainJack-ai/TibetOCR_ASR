# 模型说明文档（每个模型一篇）

> 单模型说明：是什么、权重多大、怎么调、精度如何、坑在哪。
> 跨模型的"怎么选/怎么接入"见 [模型使用说明](../usage.md)；选型依据见 [调研报告](../research.md)；数字出处见 [实测结果](../results.md)。

## OCR

| 模型 | 推荐度 | 状态 | 一句话 | 文档 |
|---|---|---|---|---|
| BDRC/tibetan-ocr（Yigdzin 1） | ★★★★★ | ✅ 实测 | 唯一生产级开源藏文 OCR，0.8B，CER 2.5% | [yigdzin.md](./yigdzin.md) |
| Tesseract 5 + bod | ★☆☆☆☆ | ✅ 实测 | 8.6MB 兜底，CER 53% | [tesseract-bod.md](./tesseract-bod.md) |
| MITRA / DONUT / PaddleOCR-VL / Easter2 / Transkribus | - | 📄 调研 | 榜单对比与未采用原因 | [other-ocr.md](./other-ocr.md) |

## ASR

| 模型 | 推荐度 | 状态 | 一句话 | 文档 |
|---|---|---|---|---|
| wav2vec2-base（NICT-Tib1） | ★★★★★ | ✅ 实测 | CER 7.25%，43 倍实时，无标点 | [wav2vec2-base.md](./wav2vec2-base.md) |
| whisper-small（NICT-Tib1） | ★★★★★ | ✅ 实测 | SWER 18.4%，**必须 9s 分块** | [whisper-small.md](./whisper-small.md) |
| openpecha-general | ★★★☆☆ | ✅ 实测 | 混合域最好（15.5%），许可未声明 | [openpecha-general.md](./openpecha-general.md) |
| tenzin/whisper-small-tibetan | ★★☆☆☆ | ✅ 实测 | SWER 17.9%，**非商用许可** | [tenzin-whisper-small.md](./tenzin-whisper-small.md) |
| whisper-tiny / whisper-base | ★★★☆☆ | ✅ 实测 | 轻量版，需分块 | [whisper-tiny-base.md](./whisper-tiny-base.md) |
| hubert-base | ★★☆☆☆ | 📄 调研 | 词级误差最大，无选用理由 | [hubert-base.md](./hubert-base.md) |

## 统一评测口径（便于横向比较）

- **OCR**：BDRC 协议（botok 规范化 → 去空白/折叠 tsheg/去占位符 → 字符级 Levenshtein，单页封顶 1.0），60 页分层样本（手稿 18 / 数字字体 14 / 木刻 14 / 现代 5 / 金属活字 5 / 打字机 4）。
- **ASR**：NICT-Tib1 统一 300 条（3 位说话人各 100 条，35.4 分钟）；OpenPecha 为 8 领域 × 20 条。
- **硬件**：Apple M4（10 核 GPU）/ 16GB / MPS，batch=1。
- 术语（CER/SWER/RTF/灾难页/pp）见 [指标名词表](../metrics-glossary.md)。
