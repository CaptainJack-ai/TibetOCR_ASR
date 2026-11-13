# 真实场景使用手册（Scenarios）

> 前面几篇讲"模型怎么调"，这一篇讲"**手上有一批真实材料，从今天开始怎么跑完**"。
> 命令都可以直接复制；数字来自本项目实测（M4 / 16GB / 单进程）。
>
> 相关：[模型使用说明](./usage.md)（参数细节）｜[单模型文档](./models/README.md)｜[指标名词表](./metrics-glossary.md)

---

## 0. 三种用法，先认领一个

| 用法 | 适合 | 入口 |
|---|---|---|
| **A. 单文件快速验证** | 试效果、几十页以内 | `scripts/predict_ocr.py` / `predict_asr.py` |
| **B. 批处理（推荐）** | 一本 PDF、一个录音目录、一个视频 | **`scripts/pipeline.py`**（断点续跑 + 质量闸门 + 报告） |
| **C. 服务化** | 已有业务系统要调 | 自己包一层 API（§6 有最小实现） |

**第一次用，先按 A 跑 3 个文件**，确认精度和速度符合预期，再上 B；不要一上来就跑一万页。

### 0.1 环境（两条必须知道的）

```bash
cd /Users/zhouchenyu/Desktop/Code/TibetOCR_ASR
# 模型已缓存后，加 --offline（或两个环境变量），否则可能卡 5 分钟在联网校验
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

> **为什么必须加**：模型权重已在本地缓存后，transformers 仍会去 Hub 校验元数据；网络不通时会重试 5 次（实测卡 5 分钟无输出）。加这两个变量后 `import transformers` 从 >120s 降到 0.6s。
> 三个命令行工具（`pipeline.py` / `predict_ocr.py` / `predict_asr.py`）都支持 `--offline` 开关，等价于设置这两个变量。

> **另一条坑**：不要把 `.venv` 放在 iCloud 同步目录（本机 `~/Desktop` 就是）。实测被同步/冲突处理后产生 **7,136 个 `* 2.py` 重复文件**，导致读文件阻塞。本项目的 `.venv` 已迁移到 `~/.venvs/tibetocr` 并用软链接指回来，命令完全不变。

### 0.2 三条命令记住

```bash
# 单文件
./.venv/bin/python scripts/predict_ocr.py page.jpg
./.venv/bin/python scripts/predict_asr.py talk.m4a --model wav2vec2-base

# 批处理（PDF / 图片目录 / 音频目录 / 视频，可混着给）
./.venv/bin/python scripts/pipeline.py ocr book.pdf notes/ --out outputs/myocr
./.venv/bin/python scripts/pipeline.py asr lecture.mp4 rec/ --out outputs/myasr --model whisper-small
```

`pipeline.py` 每次运行产出固定结构，随时可中断、重跑：

```
outputs/myocr/
  pages.jsonl        每页一条记录（追加写，崩溃不丢）
  pages.csv          同上，表格版
  transcripts/       每个输入一个 .txt（PDF 自动按页拼接）
  report.md          总数/耗时/吞吐/**需要人工复核的清单**
  _pages/            PDF 拆出来的中间图
```

---

## 1. 场景 A：扫描书 / PDF → 文本语料

**典型输入**：一本藏文书的扫描 PDF（200–600 页）、或一批手机拍的书页。
**目标**：得到可检索、可校对的纯文本。

### 1.1 步骤

```bash
# ① 先跑 3 页试水（用 --limit）
./.venv/bin/python scripts/pipeline.py ocr book.pdf --out outputs/book --limit 3 --offline

# ② 看报告和第一段文本，判断质量是否可接受
cat outputs/book/report.md
head -c 400 outputs/book/transcripts/book_0001.txt

# ③ 全量（可随时 Ctrl-C，重跑会跳过已完成的页）
./.venv/bin/python scripts/pipeline.py ocr book.pdf --out outputs/book --offline

# ④ 只处理报告里被标记的页（人工复核）
cat outputs/book/report.md
```

### 1.2 输入准备清单

| 项 | 建议 | 为什么 |
|---|---|---|
| PDF 转图 DPI | **250**（`--dpi 250`，默认） | 120dpi 会丢笔画；600dpi 只变慢不变准 |
| 页面方向 | 必须正立 | 倒置/侧转会让 CER 接近 100%，模型不做旋转矫正 |
| 裁边 | 保留完整页，别裁掉正文 | 模型按整页读，裁一半会丢行 |
| 一页一图 | 不要一页跨两张图 | 跨图会漏掉接缝处 |
| 彩色/灰度/二值 | 都行 | 内部统一 `convert("RGB")` |

打光不均、有明显阴影的拍照页，先做一次对比度增强，通常能降 1–3pp CER。

### 1.3 产出与交付

- `transcripts/<书名>.txt`：按页顺序拼接，页间空行。**这是给下游用的主产物。**
- `pages.csv`：逐页记录（字符数、耗时、loop_score），可以进 Excel/数据库。
- 若要保留页边界，直接读 `pages.jsonl` 的 `text` 字段，自己按 `source` 分组。

### 1.4 质量与人工

按材料决定复核比例（本项目实测 CER）：

| 材料 | CER | 建议复核 |
|---|---:|---|
| 数字字体、金属活字 | <0.7% | 抽检 5% |
| 木刻、现代印刷 | ~1.3% | 抽检 10% |
| 手稿 | ~3.6% | 20%，重点看咒语/专名 |
| 打字机 | ~12.8% | **全量校对**（会有整段漏行） |

`report.md` 里的"需要人工复核"清单已按 `loop_score ≥ 6`、空输出、过短自动筛出。

### 1.5 耗时与容量

| 规模 | 单进程耗时 | 说明 |
|---|---:|---|
| 100 页 | ~55 分钟 | 33s/页 |
| 500 页 | ~4.6 小时 | |
| 1,000 页 | ~9.2 小时 | 建议晚上跑 |
| 10,000 页 | ~3.8 天 | 建议拆成多台机器（每台单进程） |

**不要在同一台机器上并发多个 OCR 进程**：共享 GPU 会互相拖慢（实测单页从 30s 涨到 36s），而且内存 2.9GB/进程。要提速就加机器，不是加进程。

---

## 2. 场景 B：讲课 / 会议 / 视频 → 转写稿 + 字幕

**典型输入**：1–2 小时的 mp4/m4a，或一个录音目录。
**目标**：可读的 txt（讲稿/纪要）或 srt（字幕）。

### 2.1 先选模型

| 你的情况 | 选 | 理由 |
|---|---|---|
| 要**字幕/讲稿**，能等 | `--model whisper-small`（默认） | 词级最准（SWER 18.4%），自动 9s 分块出 srt |
| 要**快**，只是检索/初稿 | `--model wav2vec2-base` | 43 倍实时，1 小时音频 83 秒 |
| 音频是影视对白/多人闲聊/儿童 | `--model openpecha-general` | 混合域 CER 15% vs 54% |
| 只是拉萨新闻/朗读 | 上面两个都行 | 都是 NICT 域内 |

> 判断依据：**先看你的音频属于哪个域**。NICT 微调模型在跨域时 CER 从 7% 掉到 54%，这是最大的风险点。

### 2.2 命令

```bash
# 视频直接给，内部用 ffmpeg 抽 16k 单声道
./.venv/bin/python scripts/pipeline.py asr lecture.mp4 --out outputs/lecture --model whisper-small --offline

# 整个录音目录
./.venv/bin/python scripts/pipeline.py asr recordings/ --out outputs/rec --model wav2vec2-base --offline
```

产出：

```
outputs/lecture/transcripts/lecture.txt   # 全文
outputs/lecture/transcripts/lecture.srt   # 分块字幕（whisper 才有）
outputs/lecture/report.md                 # 时长/字符/RTF/需复核清单
```

### 2.3 字幕质量

- srt 的切分点来自**静音检测**（能量 VAD），不是语义断句；通常可读，但长句会被切在中间。
- 想更细：`--chunk-seconds 6`（更碎、更准、更慢）；想更连贯：`--chunk-seconds 12` 但 **CER 会从 9.7% 升到 30%**，不建议超过 9–10s。
- 字幕是**草稿级**：SWER 18% 意味着约每 5 个词错 1 个，发布前必须人工过一遍。

### 2.4 长音频的现实预期

| 音频 | whisper-small | wav2vec2-base |
|---|---:|---:|
| 10 分钟 | ~6.5 分钟 | ~14 秒 |
| 1 小时 | ~39 分钟 | ~83 秒 |
| 10 小时 | ~6.5 小时 | ~14 分钟 |

（RTF 0.653 / 0.023，实测口径）

### 2.5 人工校对流程建议

1. 用 `report.md` 的"需要复核"清单定位异常文件（空输出、字符率 <3 字/秒）。
2. 字幕校对时对照 srt 时间戳听 3–5 段即可判断整体质量，不必全听。
3. 专名/数字（人名、年份）是重灾区，建议建一份"领域词表"做后处理替换。

---

## 3. 场景 C：大规模归档（万页级 / 百小时级）

**目标**：用有限机器稳定跑完，可中断、可复现、可追溯。

### 3.1 分片与并行

```
一台机器 = 一个进程 + 一个输出目录
机器数 ≈ 需要的工作量 / 单机日产能
```

| 任务 | 单机日产能（24h 串行） | 10,000 页 / 100 小时需要 |
|---|---:|---|
| OCR（Yigdzin） | ~2,600 页 | 4 台机器 ≈ 1 天 |
| ASR（whisper-small） | ~37 小时音频 | 3 台机器 ≈ 1 天 |
| ASR（wav2vec2） | ~1,000 小时音频 | 1 台机器 ≈ 2.5 小时 |

按**输入文件**分片（每台机器处理不同的书/目录），不要按页分片（会产生大量小文件与重复加载模型）。

### 3.2 断点续跑

`pipeline.py` 已内建：

- 每条记录完成即写 `pages.jsonl`（`flush()`），进程被杀不丢已完成的部分；
- 重跑时读 jsonl 里 `status == "ok"` 的 id 直接跳过；
- 失败的 item 会在下次重跑时**自动重试**。

所以标准操作就是：**中断了就直接再跑同一条命令**。

### 3.3 失败处理

```bash
# 看有多少失败/待复核
grep -c '"status": "error' outputs/book/pages.jsonl
sed -n '/需要人工复核/,$p' outputs/book/report.md
```

失败通常是：图片损坏、PDF 某页是空白、音频编码异常。单条失败不会中断整批，报告里会列出来。

### 3.4 清单与可追溯

每批保留：

| 文件 | 作用 |
|---|---|
| `pages.jsonl` / `items.jsonl` | 原始记录（不可再生，务必保留） |
| `report.md` | 批次质量快照 |
| 启动命令（写进 README 或脚本） | 复现 |
| `reports/environment.txt` 的副本 | 环境/版本（换机器或升级后结果会变） |

**建议**：每个批次目录里放一个 `manifest.json`，记录输入清单、模型 repo id、git commit（如果有）、参数、开始时间。出了质量问题能倒查。

---

## 4. 场景 D：混合材料（手稿 + 印刷 + 打字机）

**问题**：一本书里混了木刻、现代印刷、手写批注；或一批档案什么都有。统一参数会浪费人力。

**做法：先分类，再分流。**

```bash
# ① 全量跑（模型本身对材料不敏感，先出结果）
./.venv/bin/python scripts/pipeline.py ocr mixed/ --out outputs/mixed --offline

# ② 按 report + 逐页指标分层
./.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/mixed/pages.csv")
df["need_review"] = df.loop_score >= 6
print(df.groupby("need_review").size())
print(df[df.need_review][["id", "chars", "loop_score"]].head(20))
PY
```

- `loop_score` 高、字符数异常、空白 → 优先人工；
- 若材料类型可自动判断（例如用 BDRC 的文字体系分类器），可以给不同材料设不同的复核比例与 `--max-new-tokens`。

**先验经验**：手稿页平均 CER 是印刷页的 3–5 倍；打字机页必须全量校对。所以"混合材料全量人工"是浪费，按材料分层才是正确成本。

---

## 5. 场景 E：一轮完整的人工-机器协同流程

适合"要交付质量"的项目（出版、字幕上线）：

```
① 机器跑全量（pipeline.py）
        ↓
② 自动闸门筛出可疑项（report.md）
        ↓
③ 分层抽样人工转写 50–100 条 → 算你自己的 CER/SWER 基线
        ↓
④ 按基线决定复核比例（本项目给的 CER→比例表见 §1.4）
        ↓
⑤ 人工校对（编辑器里对照原图/音频）
        ↓
⑥ 把校对后的文本回灌，作为下次的回归测试集
```

关键点：**第 ③ 步不能省**。别人报告的 CER 不能直接套到你的数据上（域不同差 7 倍）。用 50–100 条自有样本定标，成本很低，但能决定后面几万页的复核预算。

回归集可以这样固化：

```bash
mkdir -p eval/mine && cp transcripts/page_0001.txt eval/mine/   # 校对后的正确文本
# 以后每次换模型/参数，重跑这批 page_id，用 scripts/score_ocr.py 的口径算 CER
```

---

## 6. 场景 F：接入自己的系统（最小 API）

单机、单进程、串行队列。**不要**为 OCR/ASR 起多 worker（MPS 会互相拖慢）。

```python
# app.py —— 最小实现，够内部用
from fastapi import FastAPI, UploadFile, BackgroundTasks
from pathlib import Path
import shutil, sys, uuid
sys.path.insert(0, "scripts")
from pipeline import cmd_ocr  # noqa: E402  也可直接用 YigdzinEngine

app = FastAPI()
JOBS = {}

@app.post("/ocr")
async def ocr(file: UploadFile):
    job = uuid.uuid4().hex
    dst = Path("outputs/api") / job
    dst.mkdir(parents=True, exist_ok=True)
    src = dst / file.filename
    src.write_bytes(await file.read())
    JOBS[job] = {"status": "queued", "out": str(dst)}

    def run():                      # 实际项目里换成队列（Redis/RQ/Celery）
        import subprocess
        JOBS[job]["status"] = "running"
        subprocess.run(["./.venv/bin/python", "scripts/pipeline.py", "ocr",
                        str(src), "--out", str(dst)], check=False,
                       env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
        JOBS[job]["status"] = "done"

    BackgroundTasks().add_task(run)
    return {"job": job}

@app.get("/jobs/{job}")
def status(job: str):
    return JOBS.get(job, {"status": "unknown"})
```

工程注意：

| 项 | 建议 |
|---|---|
| 并发 | 1（GPU 共享，多请求反而更慢）；用队列排队 |
| 超时 | OCR 单页 30–120s；API 超时设 180s 或改成异步任务 |
| 模型常驻 | 进程启动时加载一次，别每个请求都 `from_pretrained`（加载 6s） |
| 缓存 | 同一文件（md5）重复提交直接返回旧结果 |
| 输入校验 | 拒绝 >50MB 图片 / >4h 音频，避免 OOM |
| 返回 | 文本 + `loop_score` + `flags`，让前端知道要不要提示"需复核" |

---

## 7. 统一的质量闸门（直接抄）

| 任务 | 信号 | 阈值 | 动作 |
|---|---|---|---|
| OCR | 输出为空 / <10 字符 | - | 人工或重拍 |
| OCR | `loop_score`（12-gram 最大重复次数） | ≥6 | 进复核队列（≥8 自动温度重采样） |
| OCR | 字符数远低于同批中位数 | - | 检查是否空白页/裁切错误 |
| OCR | 单页耗时 >3× 中位数 | - | 检查是否超长页/循环 |
| ASR | 输出为空 | - | 人工 |
| ASR | 字符率 <3 字/秒 | - | 音频可能无声/纯音乐/语种不符 |
| ASR | 输出非藏文字符占比高 | >10% | 音频可能不是藏语 |

> 注意：`loop_score` 对**经文中合法重复**也会给出 3–5，所以阈值设在 6；真正的退化循环通常几十上百。

---

## 8. 真实踩坑清单

| 症状 | 原因 | 解决 |
|---|---|---|
| 命令跑起来 5 分钟没输出 | transformers 联网校验元数据超时 | 加 `--offline` 或 `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| `import transformers` 卡住、CPU 0% | venv 在 iCloud/网盘同步目录，文件读取被阻塞；或产生大量 `* 2.py` 副本 | venv 放本地（本项目用 `~/.venvs/tibetocr` + 软链接） |
| OCR 单页 >5 分钟 | 忘了 `use_cache=True` | `pipeline.py`/`predict_ocr.py` 已内置 |
| OCR 漏行、行序乱 | `mm_token_type_ids` 未清零 | 已内置 |
| 转写字数明显偏少 | Whisper 448 token 截断 | 用 `pipeline.py asr`（默认 9s 分块）或调小 `--chunk-seconds` |
| 长音频内存暴涨 | CTC 模型一次吃整个音频 | `pipeline.py` 对 CTC 默认 30s 分块 |
| 中文/英文文件名导致路径怪 | 文件名被清洗成 `_` | 已用 `re.sub` 清洗并保留 `source` 原路径 |
| 同一批数据两次结果不同 | 温度/设备/dtype 变了 | 固定 `--device mps --dtype fp16`（ASR）/`bf16`（OCR），记录在 report |
| `git push` 报 `Connection reset by peer`，但浏览器能开 GitHub | git 不读 macOS 系统代理（浏览器读），直连被重置 | 运行 `./scripts/fix_git_proxy.sh`（自动读取系统代理并写入 `~/.gitconfig`）|

---

## 9. 不要做的事

- ❌ 用通用大模型（Gemini/GPT/Claude/Qwen-VL）跑藏文 OCR：CER 39%–100%。
- ❌ 用原始 Whisper / MMS 直接转藏语：输出乱码。
- ❌ 把 NICT 微调的 ASR 用在影视/儿童/方言音频上：CER 会到 54%。
- ❌ 同一台机器起多个 OCR 进程"并行提速"：共享 GPU 反而更慢。
- ❌ 关闭 Whisper 的 9s 分块处理长音频：CER 从 9.7% 涨到 45.7%。
- ❌ 不做定标就用别人的 CER 估自己的复核量：跨域可差 7 倍。
- ❌ 只保留最终 txt 而删掉 `pages.jsonl`：无法定位坏页、无法复现。

---

## 10. 一页速查

```bash
# OCR：PDF/图片/目录
cd /Users/zhouchenyu/Desktop/Code/TibetOCR_ASR
./.venv/bin/python scripts/pipeline.py ocr book.pdf shots/ --out outputs/book --dpi 250 --offline

# ASR：视频/音频/目录
./.venv/bin/python scripts/pipeline.py asr lecture.mp4 rec/ --out outputs/lecture --model whisper-small --offline

# 看结果
cat outputs/book/report.md
ls outputs/book/transcripts/
cat outputs/lecture/transcripts/lecture.srt | head

# 中断后继续：再跑一遍同样的命令即可
```
