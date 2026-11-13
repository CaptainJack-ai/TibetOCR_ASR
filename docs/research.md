# 藏语 OCR / ASR 开源模型调研报告

> 调研时间：2026-11-12。目标：找出可在本地（Apple M4 / 16GB，无 CUDA）部署、体量适中的藏语 OCR / ASR 开源模型，并确认公开可用的评测数据集。
> 所有数值均来自模型卡、论文或 BDRC 官方榜单；实测复现结果见 [results.md](./results.md)。

---

## 1. OCR：藏文识别

### 1.1 公开榜单（BDRC，472 页人工转写基准）

BDRC（佛教数字资源中心）维护了一个开放的藏文 OCR 基准：472 页，覆盖卫藏（Uchen）与无头体（Ume）、手稿/木刻/现代印刷/金属活字/打字机等，页面级人工转写。榜单按**整页平均字符错误率（mean CER）**排序，缺失页按 100% 错误计。

| 排名 | 系统 | 类型 | 参数量 | 覆盖率 | mean CER | median CER | 灾难页(>50%) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **BDRC / Yigdzin-v1** | 微调 | 0.8B | 472/472 | **3.21%** | **0.96%** | 0.64% |
| 2 | MITRA-BDRC / qwen35-0.8b | 微调 | 0.8B | 472/472 | 7.32% | 1.92% | 2.97% |
| 3 | Google Vision（闭源） | 商业 API | - | 472/472 | 22.25% | 16.83% | 9.11% |
| 5 | DONUT / PechaBridgeOCR v1 | 微调 | ~0.2B | 472/472 | 28.01% | 20.96% | 22.7% |
| 6 | DONUT / PechaBridgeOCR v2 | 微调 | ~0.2B | 472/472 | 33.67% | 19.88% | 28.6% |
| 7 | PaddleOCR-VL-1.6-0.9B（基座） | 通用 | 0.9B | 472/472 | 37.44% | 20.27% | 30.7% |
| 14 | BDRC / Easter2_woodblock（行级） | 微调 | 小 | 472/472 | 44.55% | 34.86% | 47.2% |
| 19 | Transkribus / Tibetan Generic 0.1 | 微调 | - | 259/472 | 55.59% | 44.59% | 49.4% |
| 21 | Tesseract 5.3 + `bod_uchen` | 规则/传统 | - | 471/472 | 58.29% | 70.52% | 63.3% |
| 26 | Tesseract 5.3 + `bod` | 规则/传统 | - | 471/472 | 66.58% | 79.44% | 71.4% |
| - | Azure Document Intelligence | 商业 API | - | 463/472 | 100% | 100% | 100% |
| - | Qwen3.5-2B / InternVL3.5-2B / deepseek-vl2-tiny 等通用小 VLM | 通用 | 2B 级 | - | 80%~100% | - | - |

关键结论：

1. **通用多模态大模型（Gemini 3 / Claude / GPT-5 / Qwen3.5 / InternVL 等）在藏文上普遍不可用**：榜单 46 个系统中，通用 VLM/云 API 的 mean CER 落在 39%~100%，大量页面整页输出错误或拒答；唯一的例外是 397B 的 Qwen3.5（58.4%，但仍是 Yigdzin 的 18 倍错误率）。
2. **只有专门微调的藏文 OCR 达到实用精度**，且第一名（Yigdzin 1）与第二名差距明显（3.2% vs 7.3%）。
3. 传统 OCR（Tesseract `bod`）虽然只有 8.6MB，但 CER 66.6%，只能做可用性对照，不能生产使用。

### 1.2 首选模型：BDRC/tibetan-ocr（Yigdzin 1）

| 项目 | 内容 |
|---|---|
| 仓库 | https://huggingface.co/BDRC/tibetan-ocr |
| 基座 | `PaddlePaddle/PaddleOCR-VL-1.6`（Apache-2.0） |
| 参数 | ~810M（视觉 ~413M / 26 层解码器 ~360M） |
| 权重 | 1.63 GB（bf16） |
| 许可 | **Apache-2.0**（可商用） |
| 输入 | 整页或行裁剪图；输出纯文本（保留换行） |
| 覆盖 | Uchen / Ume；现代印刷、木刻、手抄；pecha 版式与缩写 |
| 官方精度 | 1,070 页内部基准：median CER 1.42%，clean mean 4.04%，灾难页 3.3%；公开 472 页榜单 mean 3.21% |
| 用法要点 | prompt `Extract all Tibetan text. Preserve line breaks.`；必须把 `mm_token_type_ids` 清零（sequential M-RoPE）；greedy 解码；建议加 DRY 抗重复 |

**重要工程发现（本次实测）：** 该仓库的 `generation_config.json` 里 `use_cache=false`，直接按模型卡示例跑会变成 O(n²) 解码——在 M4 上单页超过 5 分钟。显式传 `use_cache=True` 后，输出逐字节一致，单页从 >300s 降到 **~14s**（60 页样本实测）。

其他注意：
- 需要 `transformers>=5.15`（原生 `paddleocr_vl` 支持），本机用 5.22.2；
- bf16 + MPS 可用；显存/内存峰值约 3GB；
- 对插图、非藏文、破损页面效果差；少数页面会出现重复循环，需 DRY 或温度重采样兜底；
- 只做"给定图 → 文本"，不做版面分析/行检测；整页输入即可。

### 1.3 其他备选（未选原因）

| 模型 | 结论 |
|---|---|
| MITRA-BDRC / bdrc-mitra-ocr-qwen35-0.8b | 同为 0.8B，CER 7.32%，约为 Yigdzin 的 2.3 倍；可作备选 |
| DONUT / PechaBridgeOCR | ~0.2B 更小，但 CER 28%，实用价值低，许可未明确 |
| PaddleOCR-VL-1.6-0.9B 基座 | Apache-2.0，但未针对藏文微调，CER 37% |
| BDRC Easter2 系列（`BDRC/Woodblock_Easter2-Vit` 等） | 行级模型，需要先做行切分，流程复杂且 CER 44%+ |
| Tesseract `bod` / `bod_uchen` | 极轻量基线（8.6MB / 0.5s 每页），但 CER 58%~67% |
| Transkribus | 商业平台，免费额度有限，覆盖页数少 |
| openpecha/Gemma_bo_OCR_4B/12B | 4B/12B 超出"不要太大"的范围，且为 demo 权重 |

---

## 2. ASR：藏语语音识别

### 2.1 公开评测（NICT-Tib1，拉萨藏语朗读新闻）

NICT-Tib1：20 位说话人，15,099 条训练 + 1,547 条测试（3 位说话人，说话人完全不重叠）。以下为 J. Moore 等（IEEE Access 2026）随论文发布的 14 个 checkpoint 中的全量微调结果：

| 模型 | 参数量 | 权重 | 许可 | CER | BoTok-SWER |
|---|---:|---:|---|---:|---:|
| `…-wav2vec2-base` | 94M | 378MB | Apache-2.0 | **7.45%** | 27.47% |
| `…-whisper-small` | 244M | 967MB | Apache-2.0 | 11.85% | **20.42%** |
| `…-hubert-base` | 94M | 378MB | Apache-2.0 | 13.52% | 44.77% |
| `…-whisper-base` | 74M | 290MB | Apache-2.0 | 14.17% | 26.00% |
| `…-whisper-tiny` | 39M | ~150MB | Apache-2.0 | 15.60% | 29.75% |
| `…-whisper-small-lora` | 244M | - | Apache-2.0 | 68.40% | 94.11% |

论文的核心结论：**CER 会低估藏语的真实错误**——wav2vec2 的 CER 最低，但词级 SWER 高达 27%；whisper-small 的 CER 略高，但 SWER 最低。原因是藏文以音节/词为单位书写，字符级替换可能改变词义。评估时应同时报告 CER 与 BoTok-SWER。

### 2.2 其他公开模型

| 模型 | 参数量 | 许可 | 报告精度 | 说明 |
|---|---:|---|---|---|
| `tenzin/whisper-small-tibetan` | 244M | **CC-BY-NC-SA-4.0** | CER 11.67% / SWER 16.84%（自有 300 条混合域测试） | 训练集混合 NICT + 西北民大 + OpenPecha + TIBMD；**非商用** |
| `openpecha/general_stt_base_model` | ~315M (wav2vec2-large) | **未声明** | 社区报告约 12.6% CER | Monlam/OpenPecha 通用模型，混合域（新闻/开示/口语）可能更稳；商用前必须确认许可 |
| `openpecha/Situ_rinpoche_asr` / `Garchen_Rinpoche_stt` | - | 未声明 | 说话人专属 8.08% CER | 单人风格，不可泛化 |
| `facebook/mms-1b-all` / Whisper 原版 zero-shot | 1B / - | 各自许可 | 不可用（输出乱码/高 CER） | 社区报告：通用 wav2vec2 无法映射藏文，通用 Whisper 输出 gibberish |

### 2.3 关键限制：Whisper 的 448 token 上限

本次实测发现：这些 Whisper 微调**沿用原版 byte-level BPE 词表**，藏文每字符约 3 个 token（UTF-8 字节），而 Whisper 解码器最多 448 个位置 → **单次最多输出约 145 个藏文字符**。NICT-Tib1 测试集中位长度 106 字符、P90 175 字符，约 30%~40% 的句子超出上限，会被硬截断。

因此：
- Whisper 类模型的 CER 必须区分"全部句子"和"参考长度 ≤150 字符的子集"来报告（本报告两者都给）；
- CTC 模型（wav2vec2/hubert）按帧输出，不受 448 token 限制，长句表现更稳；
- 生产环境若用 Whisper，需要做长音频切分（按标点/静音切到 <145 字符对应时长）。

---

## 3. 数据集调研

| 用途 | 数据集 | 规模 | 许可 | 说明 |
|---|---|---|---:|---|
| OCR 评测 | `BDRC/tibetan-ocr-benchmark` | 472 页 | **CC0** | 人工转写；Uchen 238 / Ume 193 / 过渡体 21 / 混合 20；手稿 209、数字字体 133、木刻 91、现代 17、金属活字 17、打字机 5；附官方 CER 脚本与全部历史预测 |
| ASR 主评测 | `billingsmoore/nict-tib1` | test 1,547 条 / 185 分钟 / 3 说话人 | **CC-BY-4.0** | 与论文同测试集，可直接复现 CER/SWER |
| ASR 泛化评测 | `openpecha/tibetan-voice-benchmark` | 8,367 条 / 7.36 小时 / 8 领域 | **MIT** | 有声书、儿童、历史、电影、自然口语、新闻、播客、佛法开示；转写经至少一人复核（grade 3 占比高） |
| OCR 训练（可选） | `BDRC/monlamai-transcriptions`、`BDRC/berkeley`、`BDRC/palri-parkhang`、`openpecha/OCR-Tibetan_line_to_text_benchmark` | 3k~12k 页/行 | 各自 | 用于后续微调 |
| ASR 训练（可选） | `openpecha/tibetan_voice`、`pkufool/openslr-tibetan`（CC-BY-SA-4.0）、`lilgoose777/*` 合并集 | - | 各自 | 质量和许可参差，需清洗 |

### 许可与商用提示

- 可商用：Yigdzin（Apache-2.0）、billingsmoore 系列（Apache-2.0）、Tesseract（Apache-2.0）、NICT-Tib1（CC-BY-4.0）、OpenPecha benchmark（MIT）、BDRC OCR benchmark（CC0）。
- 需谨慎：`tenzin/whisper-small-tibetan` 为 **CC-BY-NC-SA-4.0（禁止商用）**；`openpecha/general_stt_base_model` 未声明许可。

---

## 4. 方言与场景覆盖

- 现有开源模型基本只覆盖**卫藏（拉萨）中心方言**的朗读/新闻语音；
- **安多、康巴方言**没有成熟开源模型，效果未知；
- 古典藏语（宗教仪轨、颂经）与口语差异大，需要专门数据；
- OCR 侧 Yigdzin 覆盖 Uchen/Ume 的主流书写风格，但对插图、表格、破损页面、非藏文混排仍会失败。

---

## 5. 选型结论

| 场景 | 推荐 | 理由 |
|---|---|---|
| 藏文 OCR（生产） | **BDRC/tibetan-ocr（Yigdzin 1）** | 唯一达到实用精度的 0.8B 开源模型，Apache-2.0，M4 单页 ~14s |
| 藏文 OCR（极轻量/离线兜底） | Tesseract 5 + `bod` | 8.6MB，速度快，但仅能作为对照/粗筛 |
| 藏语 ASR（精度优先，离线） | **wav2vec2-base（NICT-Tib1）** | CER 最低，94M，M4 上 RTF≈0.02（50 倍实时） |
| 藏语 ASR（综合/词级质量） | **whisper-small（NICT-Tib1）** | SWER 最低，但受 448 token 限制，长句需切分 |
| 藏语 ASR（混合域泛化） | `openpecha/general_stt_base_model` | 领域更全，但许可待确认 |
| 商用合规 | 避开 `tenzin/*`（NC）与许可不明的 openpecha 模型 | - |

---

## 6. 参考链接

- BDRC 藏文 OCR 榜单：https://huggingface.co/spaces/BDRC/tibetan-ocr-leaderboard
- Yigdzin 1：https://huggingface.co/BDRC/tibetan-ocr
- BDRC OCR 基准：https://huggingface.co/datasets/BDRC/tibetan-ocr-benchmark
- BDRC 在线演示：https://ocr.bdrc.io
- NICT-Tib1 测试集：https://huggingface.co/datasets/billingsmoore/nict-tib1
- Tibetan ASR 论文（IEEE Access 2026, doi:10.1109/ACCESS.2026.3709206）与 14 个 checkpoint：https://huggingface.co/billingsmoore
- OpenPecha 藏语语音基准：https://huggingface.co/datasets/openpecha/tibetan-voice-benchmark
- OpenPecha ASR 基准报告：https://forum.openpecha.org/t/tibetan-speech-to-text-model-training-and-benchmark-report/423
