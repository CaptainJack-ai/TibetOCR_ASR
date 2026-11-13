# 其他藏文 OCR 模型（仅调研，未采用）

> **状态：📄 仅调研｜这些模型在本项目中没有本地实测**，数字来自 BDRC 公开榜单（472 页）。

## 榜单对比

| 模型 | 类型 | 参数量 | 权重可得 | 许可 | mean CER | 未采用原因 |
|---|---|---:|---|---|---:|---|
| BDRC/tibetan-ocr（Yigdzin） | 微调 | 0.8B | ✅ | Apache-2.0 | **3.21%** | —— 已被采用 |
| MITRA-BDRC / bdrc-mitra-ocr-qwen35-0.8b | 微调 | 0.8B | ✅ | 待确认 | 7.32% | 同尺寸但精度差 2.3 倍，仅作备选 |
| Google Vision | 商业 API | - | ❌ 本地 | 商业 | 22.25% | 不可本地部署、贵 |
| DONUT / PechaBridgeOCR v1 | 微调 | ~0.2B | ✅ | 未明确 | 28.01% | 更小但精度不足 |
| PaddleOCR-VL-1.6-0.9B | 通用基座 | 0.9B | ✅ | Apache-2.0 | 37.44% | 未针对藏文微调 |
| BDRC Easter2 系列 | 微调（行级） | 小 | ✅ | 待确认 | 44%–61% | 需自行做行切分，流程复杂且精度低 |
| Transkribus / Tibetan Generic | 微调 | - | ❌ 平台 | 商业 | 55.59%（覆盖 259/472） | 平台服务、免费额度有限 |
| Tesseract + bod | 传统 | - | ✅ | Apache-2.0 | 66.58% | 已在 [tesseract-bod](./tesseract-bod.md) 中实测 |

## 补充说明

- **通用多模态大模型**（Gemini 3 / Claude / GPT-5 / Qwen3.5 / InternVL / DeepSeek-OCR / Mistral OCR / HunyuanOCR）：榜单 CER 39%–100%，**结论是不要用于藏文 OCR**。
- 若 Yigdzin 不可用（例如需要更小模型），**MITRA 0.8B** 是唯一值得再评估的替代；它的接口同样是 Qwen-VL 风格（`AutoModelForImageTextToText`）。
- 行级模型（Easter2 系列）的价值在于**可与自建行切分器组合**，但需要额外投入，精度上限也低于 Yigdzin。
