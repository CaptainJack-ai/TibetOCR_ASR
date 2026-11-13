# Tesseract 5 + tessdata_best `bod` — 极轻量藏文 OCR 基线

> **状态：✅ 本项目实测｜推荐度：★☆☆☆☆（只作对照/预筛，不能生产）**

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 工具 | Tesseract 5.5.2（Homebrew） |
| 模型 | `tessdata_best/bod.traineddata`，8.6 MB |
| 许可 | Apache-2.0 |
| 部署 | 纯 CPU，无 Python 依赖（`tesseract` 命令即可） |
| 备选模型 | `bod_uchen`（有头体专用，榜单略好：58.29% vs 66.58%） |

## 2. 实测表现（BDRC 60 页分层样本）

| 指标 | 值 |
|---|---:|
| mean CER | 53.52% |
| median CER | 56.06% |
| 灾难页（>50%） | 53.3% |
| 耗时 | 3.3 s/页 |

分材料：金属活字 21.4%、现代印刷 27.0%、打字机 32.1%、数字字体 47.2%、木刻 59.2%、手稿 75.1%。
分文字：Uchen 38.5%、过渡体 39.6%、Ume 84.1%。

榜单全量（472 页）：`bod` 66.58%，`bod_uchen` 58.29%。

## 3. 怎么用

```bash
# 单页
tesseract page.png stdout -l bod --tessdata-dir tessdata --psm 6

# 批量（仓库脚本）
./.venv/bin/python scripts/run_ocr.py --engine tesseract --split sample
```

`--psm 6`（假设单块文本）在本项目效果最好；版面复杂时可试 3/4。

## 4. 结论

- 优势：**8.6MB、3 秒/页、零 GPU**，可作为"这页有没有字/是不是藏文"的预筛，或极低资源环境兜底。
- 劣势：CER 53%，一半以上页面不可用；Ume（无头体）尤其差（84%）。
- 与 Yigdzin 相比：60 页里 **59 页更差**，CER 相差约 21 倍。任何有质量要求的场景都应改用 Yigdzin。
