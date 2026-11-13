# BDRC/tibetan-ocr（Yigdzin 1）— 藏文 OCR

> **状态：✅ 本项目实测｜推荐度：★★★★★｜唯一达到实用精度的开源藏文 OCR**
> 一句话：0.8B 的"视觉为主"文档 OCR 模型，给一张整页图，直接输出藏文文本（保留换行）。

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| HuggingFace | `BDRC/tibetan-ocr`（显示名 Yigdzin 1） |
| 基座 | `PaddlePaddle/PaddleOCR-VL-1.6`（Apache-2.0） |
| 架构 | 视觉编码器 + 26 层解码器（SOLAR 深度扩展），`model_type=paddleocr_vl` |
| 参数量 | ~810M（视觉 ~413M / 解码器 ~360M） |
| 权重 | 1.63 GB（bf16） |
| 许可 | **Apache-2.0**（可商用） |
| 任务 | 图像 → 文本（image-text-to-text） |
| 语言 / 文字 | 藏文；Uchen（有头体）、Ume（无头体）、过渡体 |
| 开发者 | Buddhist Digital Resource Center（BDRC） |
| 依赖 | `transformers>=5.15`（原生支持该架构）、torch、torchvision、Pillow |

## 2. 实测表现

### 2.1 公开榜单（BDRC 472 页基准，46 个系统）

| 指标 | 值 | 排名 |
|---|---:|---|
| mean CER | **3.21%** | 1 / 46 |
| median CER | 0.96% | 1 |
| 灾难页（>50%） | 0.64% | 1 |
| 第二名（MITRA 0.8B） | 7.32% | 2 |

### 2.2 本项目实测（60 页分层样本，同口径）

| 指标 | 值 |
|---|---:|
| mean CER | **2.50%** |
| median CER | **1.01%** |
| 灾难页 | **0%** |
| 单页耗时 | 中位 33.3s / 均值 36.4s / 最大 114.7s |
| 内存峰值 | ≈2.9 GB |

分材料：数字字体 0.61%、金属活字 0.33%、木刻 1.23%、现代印刷 1.34%、手稿 3.61%、**打字机 12.77%**。

## 3. 怎么用

### 3.1 CLI（最快）

```bash
./.venv/bin/python scripts/predict_ocr.py mypage.jpg --json
./.venv/bin/python scripts/predict_ocr.py scan.jpg --temperature 0.4 --num-samples 3   # 循环兜底
```

### 3.2 嵌入代码（最小实现，已实测）

```python
import sys, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessorList

sys.path.insert(0, "scripts/vendor")
from dry_logits_processor import make_hf_dry_processor

MODEL_ID, PROMPT, DEVICE = "BDRC/tibetan-ocr", "Extract all Tibetan text. Preserve line breaks.", "mps"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()

def ocr(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: (v.to(DEVICE, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(DEVICE))
              for k, v in inputs.items()}
    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])      # 必须
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=2048, use_cache=True,
            logits_processor=LogitsProcessorList([make_hf_dry_processor(
                prompt_len=inputs["input_ids"].shape[1], multiplier=0.8, base=1.75, allowed_length=12)]))
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
```

## 4. 关键参数（照抄即可）

| 参数 | 值 | 为什么 |
|---|---|---|
| prompt | `Extract all Tibetan text. Preserve line breaks.` | 训练/推理一致的指令，换 prompt 会掉点 |
| `dtype` | `torch.bfloat16` | fp16 未验证；bf16 在 MPS 正常 |
| `use_cache` | **True（必须显式）** | 仓库默认 false，否则解码 O(n²)，单页 >300s |
| `mm_token_type_ids` | **清零（必须）** | 该模型用 sequential 1-D 图像位置；不清零会漏行/串行 |
| `do_sample` | False | 确定性输出 |
| `max_new_tokens` | 2048 | 分词器约 1.26 字符/token；60 页最长 1,562 token。全量 472 页有 4 页 >2048（最长 2,451），跑全量建议 3,072–4,096 |
| DRY | `multiplier=0.8, base=1.75, allowed_length=12` | 官方抗重复配置，不损伤正常经文 |
| `attn_implementation` | `"sdpa"`（可选） | 更快，非必需 |

## 5. 能力边界

**能做**：整页/行条图 → 文本；现代印刷、木刻、手抄、金属活字、打字机；pecha 版式；正字法缩写。

**不能做**：版面分析（分栏/表格/图注/插图定位）、非藏文、破损或极模糊页面、行/区域检测。多栏或图文混排页面需先切块。

## 6. 已知问题

| 现象 | 原因 | 对策 |
|---|---|---|
| 单页 >300s | `use_cache=false` | 显式 `use_cache=True`（实测 83.4s→7.5s @120 token） |
| 漏行、行序错乱 | `mm_token_type_ids` 未清零 | 清零 |
| 整段漏行（长页面） | 多栏/版序 | 切块后逐块识别 |
| 同一句循环 | 重复循环 | DRY；`temperature=0.4, n=3` 取重复度最低 |
| 打字机页 CER 12.8% | 字形结构特殊 | 必须人工校对 |
| 结尾中断 | token 预算用尽 | 提高 `max_new_tokens` |

## 7. 结论

- **这是目前唯一建议用于生产的开源藏文 OCR**；0.8B / 2.9GB 在 16GB 机器上可跑。
- 单进程约 **109 页/小时**（33s/页），不要并发多进程（共享 GPU 会变慢）。
- 按材料决定人工复核比例：数字字体/木刻抽检；手稿 20%；打字机全量校对。
