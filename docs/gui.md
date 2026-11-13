# 本地评测 GUI 使用说明

> 一个跑在本机的浏览器界面，用于**看模型到底在哪些字上出错**：选样例（自带标准答案）或上传自己的藏文图片/语音，跑一次推理，得到逐字着色的 diff 与指标。
> 实现：`gui/server.py`（标准库 HTTP 服务）+ `gui/static/*`（原生 HTML/CSS/JS）+ `gui/diffrender.py`（对齐与着色）。**不新增任何依赖。**

---

## 1. 启动

```bash
cd /Users/zhouchenyu/Desktop/Code/TibetOCR_ASR
./scripts/gui.sh                 # 打开 http://127.0.0.1:8765
./scripts/gui.sh --port 9000     # 换端口
./scripts/gui.sh --no-browser    # 不自动开浏览器
```

- 只监听 `127.0.0.1`，不对外暴露。
- 启动时打印可用样例数量；模型**首次使用时才加载**（Yigdzin 约 6 秒），之后常驻。
- 用 `--offline`（`gui.sh` 已默认）跳过 HuggingFace 联网校验，避免无网络时卡 5 分钟。

---

## 2. OCR 评测

### 2.1 用样例（有标准答案）

1. 左栏「样例」→ 按 **材料 / 文字类型 / 清晰度** 筛选；
2. 点列表里的一项 → 中间显示原图与元数据，右侧准备运行；
3. 「运行识别」→ 状态栏显示已用时（整页约 10–35 秒）；
4. 完成后：
   - **指标卡**：CER、字符正确率、替换/漏识别/多识别计数、参考长度、预测字符数、耗时、字符/秒；
   - **diff 区**：预测文本按字符着色（见 §4）；
   - 计数行：`匹配 905 · 替换 0 · 漏识别 0 · 多识别 0 · 参考 905 字符`。

### 2.2 用自己的图片

1. 左栏切到「上传新图片」→ 选文件（png / jpg / tif，≤40MB）；
2. 若有标准答案，粘贴到文本框（**这一栏决定是否计算指标**）；
3. 点「识别上传图片」。

> 判定标准答案的两条路径：样例会自动读 `ground_truth.csv`；上传则读你粘贴的文本。都不提供时只显示预测文本。

### 2.3 视图切换

| 按钮 | 显示 |
|---|---|
| 行内 | 单栏，预测文本着色；悬停红/灰片段可看参考原文 |
| 并排 | 左「参考」右「预测」两栏对齐，错误同时标在两侧 |

---

## 3. ASR 评测

1. 左栏选数据集（**NICT-Tib1** 拉萨朗读新闻 / **OpenPecha** 8 领域）→ 选领域 → 点一条样例；
2. 中间是播放器（支持拖动，服务端实现了 Range 请求）+ 元数据 + **分段表**（每段 start/end/字符数/文本）；
3. 右侧选模型与分块长度：
   - `分块 = 0` 表示 **auto**：Whisper 类 9 秒、CTC 类 30 秒（与结论一致，见下）；
   - Whisper 单次最多约 145 字符（448 token），**不设分块时 >150 字的句子会被截断**；
4. 「开始转写」→ 完成后得到 CER / SWER / 字符正确率 / 词正确率 / 时长 / 字符数 / RTF，以及字符级与**词级（BoTok 分词）**两套 diff。

用自己的音频/视频：切「上传音频 / 视频」，支持 wav/mp3/m4a/mp4 等，服务端用 ffmpeg 转 16k 单声道。

---

## 4. 颜色语义（唯一一套）

| 语义 | 说明 | 视觉 |
|---|---|---|
| 正确 | 与参考完全一致 | 正常墨色 |
| **替换** | 参考有这个字，预测成了别的字 | 红底 `#ffd7d5`，字 `#a40e26` |
| **漏识别**（删除） | 参考有、预测没有 | 灰底 + 删除线 |
| **多识别**（插入） | 预测多出来的字 | 琥珀底 + 虚线下划线 |
| 空格 / 换行 | 便于发现空白差异 | `␣` / `¶` 浅灰 |

- 图例固定显示在 diff 上方；
- 计数行给出四类数量，与指标卡一致；
- 藏文行高 2.15、字号 18px（藏文有上下加字，行距小会裁字形）。

---

## 5. 批量抽检

「批量抽检」页：选任务（OCR/ASR）→ 条数 N（≤50）→ 随机种子 → 条件（材料 / 领域）→ 运行。

- 串行执行，状态栏显示 `i/n` 进度；
- 完成后给 **mean / median / p90 CER、灾难率（CER>50%）**，ASR 另有 mean SWER；
- 表格按 CER 从高到低排序，方便先看最差的；每行含预测前 200 字。

用途：换模型/参数后，抽 20–50 条判断整体水平，不必跑全量。

---

## 6. 指标定义

| 指标 | 定义 | 备注 |
|---|---|---|
| **CER** | `Levenshtein(归一化预测, 归一化参考) / len(参考)`，单条封顶 1.0 | 与 BDRC 官方脚本一致 |
| **字符正确率** | `1 − CER` | 直观展示用 |
| **SWER** | BoTok 分词后按词算 WER | 藏文一个词约 4.6 字符，通常 SWER ≈ 3–4 × CER |
| **词正确率** | `1 − SWER` | |
| **RTF** | 推理耗时 / 音频时长 | <1 为快于实时 |
| **灾难率** | 单条 CER > 50% 的比例 | 批处理里最能反映"坏样本" |

所有计算调用 `scripts/common.py`，与 `reports/ocr_results.csv`、`reports/asr_results.csv` 同源，可交叉验证。

---

## 7. 后端接口（便于脚本化）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 设备、torch 版本、已加载模型 |
| GET | `/api/samples/ocr?technology=&script=&legibility=&limit=` | 样例列表 |
| GET | `/api/samples/ocr/<id>` | 单页（图 URL + 参考 + 元数据） |
| GET | `/api/image/<id>` | 原图 |
| GET | `/api/samples/asr?dataset=&dept=&limit=` | 语音样例列表 |
| GET | `/api/samples/asr/<dataset>/<id>` | 单条（音频 URL + 参考） |
| GET | `/api/audio/<dataset>/<id>` | 音频（支持 Range） |
| POST | `/api/sample/ocr` `{id, engine}` | 跑本地样例 |
| POST | `/api/sample/asr` `{dataset, id, model, chunk}` | 跑本地样例 |
| POST | `/api/infer/ocr` | body=图片字节；头 `X-Filename` `X-Engine` `X-Ref`(URL 编码) |
| POST | `/api/infer/asr` | body=音频字节；头 `X-Filename` `X-Model` `X-Chunk` `X-Ref` |
| POST | `/api/batch/ocr` `{n, seed, engine, technology}` | 批量 |
| POST | `/api/batch/asr` `{n, seed, dataset, model, dept}` | 批量 |
| GET | `/api/job/<id>` | `{status, elapsed, progress, result, error}` |

任务都是异步的：POST 返回 `{job}`，随后轮询 `/api/job/<id>`（页面 500ms 一次）。

示例：

```bash
J=$(curl -s -X POST -H 'Content-Type: application/json' \
      -d '{"id":"0039","engine":"yigdzin"}' \
      http://127.0.0.1:8765/api/sample/ocr | python3 -c 'import json,sys;print(json.load(sys.stdin)["job"])')
curl -s http://127.0.0.1:8765/api/job/$J | python3 -m json.tool | head -20
```

---

## 8. 设计取舍

| 决定 | 原因 |
|---|---|
| 标准库 `http.server` 而不是 FastAPI/Gradio | 零新增依赖、离线可用；样式与 diff 渲染完全可控 |
| 全局一把推理锁 | MPS 共享，多进程/多线程并发会互相拖慢（实测单页 30s→36s） |
| 异步 job + 轮询 | OCR 单页 10–35s、批量更久，避免 HTTP 超时 |
| 上传用原始 body + 头部传元数据 | 不必手写 multipart 解析，减少出错面 |
| 指标全部走 `scripts/common.py` | 避免 GUI 与报告口径漂移 |

---

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| 首次运行卡在「加载模型」 | 正常，Yigdzin 约 6 秒；之后常驻 |
| 页面打不开 | 端口被占：`./scripts/gui.sh --port 9001` |
| 一直显示"运行中"很久 | OCR 长页面可达 60–115 秒；看状态栏计时 |
| 藏文字形显示不全 | macOS 自带 Kailasa；其他系统建议安装 Noto Serif Tibetan |
| 想离线跑 | `gui.sh` 已带 `--offline`，不需要网络 |
| 上传报"文件过大" | 限制 40MB；大 PDF 请先用 `pipeline.py ocr` 拆页 |
| 想批量跑上千页 | 用 `scripts/pipeline.py`（断点续跑、CLI），GUI 面向逐条检查 |
