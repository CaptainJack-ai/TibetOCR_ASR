# 数据集说明：类型、格式、字段与磁盘组织

> 本文说明本项目**用了哪些数据、原始格式长什么样、我们把它变成了什么、放在磁盘什么位置、谁读写谁**。
> 指标口径见 [指标名词表](./metrics-glossary.md)，运行记录见 [../reports/runlog.md](../reports/runlog.md)。
> 所有路径均相对于项目根目录 `/Users/zhouchenyu/Desktop/Code/TibetOCR_ASR`。

---

## 0. 总览

| 数据集 | 用途 | 规模 | 许可 | 原始格式 | 落盘位置 |
|---|---|---|---|---|---|
| **BDRC Tibetan OCR Benchmark** | OCR 评测（参考+图） | 472 页 / 122MB 图像 | CC0 | 1 个 parquet（图像字节内嵌） | `data/ocr/` |
| **NICT-Tib1 (test split)** | ASR 主评测（拉萨朗读） | 1,547 条 / 185.5 分钟 / 3 说话人 | CC-BY-4.0 | 1 个 parquet（音频字节内嵌） | `data/asr/nict_test/` |
| **OpenPecha / Monlam 语音基准** | ASR 跨域泛化（8 领域） | 8,367 条全量；本项目抽 160 条 / 8.9 分钟 | MIT | 1 个 CSV（音频是外链 URL） | `data/asr/openpecha_bench/` |

三个数据集有一个共同特点：**元数据与媒体内容被"打包在同一个文件里"**（parquet 的 `struct<bytes, path>` 字段），或者"元数据在仓库、媒体在外部 CDN"（OpenPecha）。因此数据准备脚本必须先做一次**解包/转码**，把训练与推理代码需要的"文件系统形态"建出来。

---

## 1. 磁盘布局

```
TibetOCR_ASR/                                        总量 ≈ 2.5GB（含 .venv）
├── data/                                            ≈ 960MB   数据集（.gitignore）
│   ├── ocr/                                         ≈ 249MB   ← 数据集 1
│   │   ├── hf/                                      ≈ 123MB    原样下载（HuggingFace 本地快照）
│   │   │   ├── data/test-00000-of-00001.parquet      121,619,456 B   472 页 + 图像 + 转写
│   │   │   ├── runs/compute_cer.py                         5,324 B   官方打分脚本
│   │   │   ├── runs/ground_truth.csv                   1,257,403 B   官方 GT（botok 已规范化）
│   │   │   ├── runs/requirements.txt                          16 B
│   │   │   └── .cache/huggingface/…                                  下载锁 / etag / 校验元数据
│   │   ├── images/                                   ≈ 122MB    472 个 {id}.png（从 parquet 解出）
│   │   ├── ground_truth.csv                            1,257,403 B   472 行 id,transcription
│   │   ├── metadata.csv                                   39,035 B   472 行，8 列分桶信息
│   │   ├── official/                                  ≈ 1.3MB    官方口径副本（只读参考）
│   │   │   └── compute_cer.py / ground_truth.csv / requirements.txt
│   │   ├── sample_ids.json                                    829 B  pilot 6 页 / sample 60 页
│   │   └── stats.json                                         335 B  技术/文字类型分布
│   └── asr/                                          ≈ 712MB   ← 数据集 2 + 3
│       ├── hf/                                       ≈ 338MB
│       │   ├── data/test-00000-of-00001.parquet      334,751,877 B   NICT 1,547 条
│       │   └── test.csv                                2,964,580 B   OpenPecha 全量清单 8,367 行
│       ├── nict_test/                                ≈ 357MB   ← 数据集 2
│       │   ├── wav/                                  ≈ 356MB   1,547 × nict_XXXXX.wav
│       │   ├── metadata.csv                              646,608 B   1,547 行
│       │   └── sample_ids.json                            33,281 B   sample 300 / all 1547
│       └── openpecha_bench/                           ≈ 17MB   ← 数据集 3
│           ├── wav/                                    ≈ 17.2MB  160 个原始文件名 .wav
│           ├── metadata.csv                               45,072 B   160 行
│           └── sample_ids.json                             7,094 B
├── outputs/                                         ≈ 9MB     每次运行的预测与记录（.gitignore）
├── reports/                                         ≈ 3MB     汇总表 + 样例 + 本文
├── tessdata/bod.traineddata                          8,623,846 B  Tesseract 藏文模型（.gitignore）
├── docs/                                            ≈ 60 KB
├── scripts/                                         ≈ 130 KB
└── .venv/                                           ≈ 1.6GB   Python 依赖（与数据无关）
```

> 注意：本文按**十进 MB** 写（1 MB = 10⁶ B），而 `du -sh` 显示的是 MiB（1 MiB = 1.048 MB）并按 4KB 块取整，所以 `du` 会显示 `data/ocr ≈ 237M`、`data ≈ 920M`。二者相差约 5%，属正常。
> `data/`、`outputs/`、`tessdata/`、`.venv/` 都在 `.gitignore` 里，**不会进版本库**。

---

## 2. 数据集 1：BDRC 藏文 OCR 基准

### 2.1 来源

| 项目 | 内容 |
|---|---|
| 仓库 | https://huggingface.co/datasets/BDRC/tibetan-ocr-benchmark |
| 版本 | 主分支快照（2026-11-07 changelog：47 页 synthetic 重生成、11 页 Uchen/Ume 修正） |
| 许可 | **CC0-1.0**（可自由使用，含商用） |
| 规模 | 472 页（另有 1,070 页私有集不公开） |
| 标注方 | Dharmaduta |
| 配套 | 46 个系统的历史预测（`runs/predictions/*.csv`）+ 官方 CER 脚本 |

### 2.2 原始格式：单个 parquet

```
data/ocr/hf/data/test-00000-of-00001.parquet   121,619,456 B   472 行 × 9 列   2 个 row group
schema:
  id             string
  transcription  string
  technology     string
  script         string
  script_4       string
  period         string
  rarity         string
  legibility     string
  image          struct<bytes: binary, path: string>
```

**`image` 是嵌套结构体**（HuggingFace `datasets` 的 Image 特性约定）：`bytes` 是完整图像文件字节（原样，未解码），`path` 是匿名化文件名（就是 4 位页码，无扩展名）。用 pandas 读出来是 `dict`：

```python
>>> df.iloc[0]["image"].keys()
dict_keys(['bytes', 'path'])
>>> len(df.iloc[0]["image"]["bytes"]), df.iloc[0]["image"]["path"]
(195521, '0002')
```

这就是为什么数据准备脚本要用 `img_path.write_bytes(row.image["bytes"])` 而不是"复制文件"。

### 2.3 字段字典（原始 9 列）

| 列 | 类型 | 含义 | 取值 / 示例 | 为什么有用 |
|---|---|---|---|---|
| `id` | string | 匿名化页码，4 位补零 | `"0002"` … `"1075"`（**不连续**，472 个里有很多空洞） | 与官方 `ground_truth.csv`、`predictions/*.csv` 对齐的唯一键 |
| `transcription` | string | 人工转写，仅正文框（不含页边、页码） | `༄༅། །སྔོ་ལྗང་མྱུ་གུ་ཡིས་ཁྱབ་ཅིང་། …` | 评测参考；**已在数据集构建时做过 botok 规范化** |
| `technology` | string | 页面制作工艺 | `manuscripts`(209) / `digital_fonts`(133) / `blockprints`(91) / `modern`(17) / `metal_types`(17) / `typewriters`(5) | 分层抽样与分桶报告的主维度 |
| `script` | string | 粗粒度文字体系 | `Uchen`(238) / `Ume`(193) / `Transitional`(21) / 混排 `Uchen+Ume` 等(20) | 判断字形体系难度 |
| `script_4` | string | 细粒度书写风格 | `Uchen` / `Pedri` / `Gyuyig` / `Druma` / `Tsugdri` / `Danyig` / `Non-applicable` | 更细的风格分析（本项目未用于报告） |
| `period` | string | 古文字学年代 | `Medieval (15)-present` | 判断历史手稿难度 |
| `rarity` | string | 该书写风格的常见度 | `4 - very common` … `1 - rare` | 判断指标是否被稀有风格左右 |
| `legibility` | string | 清晰度评级 | `4 - Excellent`(293) / `3 - Good`(124) / `2 - Fair`(42) / `1 - Poor`(12) / `0 - Illegible`(1) | 解释失败原因；抽样时避开最差页做 pilot |
| `image` | struct | 图像字节 + 文件名 | `{bytes: 195521 B, path: "0002"}` | 实际输入 |

**转写约定**（影响 CER，必须知道）：只转写正文框；保留正字法缩写；以下占位符出现在 GT 中——

| 符号 | 含义 | 本项目处理 |
|---|---|---|
| `I` | 不可辨认 | 打分前删除（BDRC 协议） |
| `B` | 破损纸面 | 删除 |
| `S` | 被划掉 | 删除 |
| `D` / `K` | 空行母文字（dakini script） | 删除 |
| `O` | 装饰性图案 | 删除 |
| `( )` | 抄写者后补内容 | 按普通字符保留 |

即 `PLACEHOLDERS = frozenset("KOBIS")` 这个集合的来源。

### 2.4 我们导出的形态

| 文件 | 行/个 | 内容 | 说明 |
|---|---:|---|---|
| `images/{id}.png` | 472 | 单页图像 | **原样字节**，未转码（见 2.5） |
| `ground_truth.csv` | 472 | `id,transcription` | 与官方 `runs/ground_truth.csv` **md5 完全相同**（`ed10b58b…`），因为 parquet 里的转写本来就是规范化过的 |
| `metadata.csv` | 472 | `id,technology,script,script_4,period,rarity,legibility,n_chars` | 供分桶统计与抽样；`n_chars` 是本项目新增列（转写字符数，用来解释耗时与 CER 长度效应） |
| `sample_ids.json` | - | `{"pilot":[6 个 id], "sample":[60 个 id]}` | 抽样结果，随机种子 42，可复现 |
| `stats.json` | - | 技术/文字类型计数 | 快速核对分布 |
| `official/` | 3 个文件 | 官方脚本 + GT + 依赖 | **只读口径基准**，`score_ocr.py` 复刻其逻辑，`verify_data.py` 用它校验 |

`ground_truth.csv` 样例（一行）：

```csv
id,transcription
0002,"༄༅། །སྔོ་ལྗང་མྱུ་གུ་ཡིས་ཁྱབ་ཅིང་། །བཏབ་པའི་ལོ་ཏོག་ལེགས་པའི་རྩ་བདེ་ཚུགས་བྱུང་། …"
```

> 文件是 **UTF-8 无 BOM**（`csv.DictWriter` 默认行为），字段带引号是因为转写里有换行与逗号。

`metadata.csv` 样例：

```csv
id,technology,script,script_4,period,rarity,legibility,n_chars
0002,blockprints,Uchen,Uchen,Medieval (15)-present,4 - very common,2 - Fair,965
0003,digital_fonts,Uchen,Uchen,Medieval (15)-present,4 - very common,4 - Excellent,811
```

### 2.5 图像本身的性质（实测）

| 项目 | 数值 |
|---|---|
| 数量 | 472 |
| 分辨率 | 宽中位 **2220 px**（526–11712）、高中位 **881 px**（227–5957） |
| 面积 | 中位 **1.86 MP**（最大约 70 MP 的长卷） |
| 色彩模式 | `RGB` 389、`1`（二值）48、`L`（灰度）34、`LA` 1 |
| **真实编码** | **JPEG 415 个、PNG 57 个** |

⚠️ **容器扩展名与真实编码不一致**：所有文件都命名为 `{id}.png`，但其中 415 个实际是 JPEG 字节（因为脚本按原样保存 `image["bytes"]`，而数据集里的原图多数是 JPEG，只是 `path` 字段不含扩展名）。

- **影响**：无。PIL 与 Tesseract 都按文件头魔数识别格式，不看扩展名；`verify_data.py` 已抽查解码通过。
- 若确实需要真 PNG（例如某些只认扩展名的工具），重新编码即可：

```bash
./.venv/bin/python - <<'PY'
from PIL import Image; from pathlib import Path
for p in Path("data/ocr/images").glob("*.png"):
    im = Image.open(p)
    if im.format != "PNG":
        im.save(p, format="PNG")
PY
```

### 2.6 抽样文件结构

```json
{
  "pilot":  ["0567", "0042", "0187", "0097", "0725", "0706"],
  "sample": ["0011", "0029", "0039", "… 共 60 个 …"]
}
```

- `pilot`：每种制作工艺各 1 页、优先短而清晰的页面，用于跑通链路 + 测速。
- `sample`：按 `technology` 配额 18/14/14/5/5/4，层内交替取 Uchen/Ume，共 60 页。
- `run_ocr.py --split pilot|sample|all` 直接读这个文件；`all` 表示全部 472 页。

---

## 3. 数据集 2：NICT-Tib1（test split）

### 3.1 来源

| 项目 | 内容 |
|---|---|
| 仓库 | https://huggingface.co/datasets/billingsmoore/nict-tib1 |
| 原始语料 | NICT-Tib1（Soky, Gong & Li, 2022），拉萨藏语朗读新闻，20 位说话人 |
| 许可 | **CC-BY-4.0** |
| 划分 | train 15,099 条（17 人）/ **test 1,547 条（3 人，与训练说话人不重叠）** |
| 为什么用它 | 与 Tibetan ASR 论文（IEEE Access 2026）的 14 个 checkpoint **同一测试集**，可直接对比论文数字 |

### 3.2 原始格式：单个 parquet

```
data/asr/hf/data/test-00000-of-00001.parquet   334,751,877 B   1,547 行 × 3 列   16 个 row group
schema:
  audio          struct<bytes: binary, path: string>
  transcription  string
  speaker_id     string
```

| 列 | 含义 | 示例 |
|---|---|---|
| `audio.bytes` | 完整 wav 文件字节（16 kHz / 单声道 / PCM_16） | 平均 ~230KB（总 356MB） |
| `audio.path` | 原始文件名 `{speaker}_{utt}.wav` | `074_344.wav` |
| `transcription` | 逐句转写（保留 tsheg/shad，含数字与专名） | `ཉིས་སྟོང་བཅུ་བདུན་ལོའི་རྒྱལ་སྤྱིའི་…` |
| `speaker_id` | 说话人编号 | `074` / `019` / `070` |

### 3.3 落盘格式与命名

| 项目 | 内容 |
|---|---|
| 路径 | `data/asr/nict_test/wav/{id}.wav` |
| 命名 | **`nict_{parquet 行号:05d}.wav`**：`nict_00000.wav` … `nict_01546.wav` |
| 编码 | 16 kHz、单声道、`PCM_16`（RIFF/WAV） |
| 为什么重命名 | 原始名 `074_344.wav` 只在同一说话人内唯一，跨 split/跨版本可能重复或变号；顺序索引 + 保留原始名到 parquet，两边都能追溯 |
| 代价 | 索引依赖 **parquet 行序**。`metadata.csv` 是唯一的"索引 → 内容"映射，请勿只保留 wav |
| 转码 | 用 `librosa.load(..., sr=16000, mono=True)` 后再写回，保证采样率/声道统一（本数据集原本就是 16k，属幂等操作） |

### 3.4 我们导出的 `metadata.csv`（7 列，1,547 行）

| 列 | 含义 | 示例 |
|---|---|---|
| `id` | 顺序索引 | `nict_00000` |
| `audio_path` | **相对项目根**的音频路径 | `data/asr/nict_test/wav/nict_00000.wav` |
| `transcription` | 参考转写 | `ཉིས་སྟོང་བཅུ་བདུན་ལོའི་…` |
| `speaker` | 说话人 | `074` |
| `duration` | 时长（秒，读取音频头得到） | `9.24` |
| `dataset` | 数据集标记 | `nict` |
| `dept` | 领域标记（恒为 `news`） | `news` |

`sample_ids.json`：

```json
{"sample": ["nict_00019", "… 共 300 个 …"], "all": ["nict_00000", "… 共 1547 个 …"]}
```

- `sample`：按说话人各抽 100 条（`800/400/347` → 各 100），共 300 条。所有 Whisper 类模型都跑这个子集；
- `all`：全量，CTC 模型（很快）跑全量；
- `summarize_asr.py` 会把全量结果过滤到 `sample`，保证所有模型在同一 300 条上比较。

### 3.5 统计

| 项目 | 数值 |
|---|---|
| 条数 / 总时长 | 1,547 条 / **185.5 分钟** |
| 单条时长 | 均值 7.19s、中位 6.56s、p90 10.51s、最大 18.87s |
| 说话人 | `074`: 800、`019`: 400、`070`: 347 |
| 参考长度 | 中位 106 字符、p90 175、最大 296 |
| 总字节 | wav 356MB（parquet 里压缩为 335MB） |

> 关键含义：单条最长约 19 秒，但参考长度可达 296 字符 → **Whisper 的 448 token 上限（约 145 字符）会截断约 1/5 的句子**，这是数据本身带来的限制，不是模型 bug。

---

## 4. 数据集 3：OpenPecha / Monlam 语音基准

### 4.1 来源

| 项目 | 内容 |
|---|---|
| 仓库 | https://huggingface.co/datasets/openpecha/tibetan-voice-benchmark |
| 制作方 | Monlam AI × OpenPecha（2024-07-15 取数） |
| 许可 | **MIT** |
| 规模 | 8,367 条 / 7.36 小时 / 8 个领域 |
| 转写质量 | 每条至少一名复核者；`grade` 3 表示通过质检团队 |
| 音频位置 | **不在仓库里**，而是 CloudFront 直链（`https://d38pmlk0v88drf.cloudfront.net/wav16k/{file_name}.wav`） |

### 4.2 原始格式：单个 CSV（含外链）

```
data/asr/hf/test.csv   2,964,580 B   8,367 行 × 8 列
```

| 列 | 含义 | 示例 |
|---|---|---|
| `file_name` | 唯一标识（也是音频 URL 的文件名） | `STT_AB00277_0455_2502107_to_2503882` |
| `uni` | 藏文 Unicode 转写（评测用） | `གཟུགས་སྐུ་སྤྱི་ཙམ་ལ་སྤྲུལ་སྐུ་དང་།` |
| `wylie` | 威利转写（拉丁字母，本项目未用） | `gzugs sku spyi tsam la sprul sku dang /__` |
| `url` | 音频直链 | `https://d38pmlk0v88drf.cloudfront.net/wav16k/….wav` |
| `dept` | 领域代码（见下表） | `STT_AB` |
| `grade` | 转写质量等级 1–3 | `3` |
| `char_len` | 转写字符数 | `36` |
| `audio_len` | 音频时长（秒） | `1.775` |

这是与前面两个数据集**结构上最大的不同**：元数据在 HF 仓库，媒体在外部 CDN。所以准备脚本要 `pandas` 读 CSV → `requests.get(url)` 下载 → 落盘。

### 4.3 下载与抽样流程

```
test.csv  ──按 dept 分组──> 每组优先取 grade≥2 ──随机抽 20 条──> 并发下载 wav(8 线程)
          └─> data/asr/openpecha_bench/wav/{file_name}.wav   （保留原始文件名）
          └─> data/asr/openpecha_bench/metadata.csv
```

- 抽样：8 领域 × 20 条 = **160 条**（随机种子 42）。
- 命名：**保留原始 `file_name`**（全局唯一，含领域前缀 `STT_AB…`）。
- 若单条下载失败只跳过该条并告警（`status` 会在后续打分中显露为缺失）。
- 需要全量时：`--openpecha-per-dept 2000`（会下载全部 8,367 条、约 7.4 小时音频）。

### 4.4 领域代码对照（`dept`）

| 代码 | 英文 | 中文 | 本样本条数 / 总时长 | 平均转写字符数 |
|---|---|---|---:|---:|
| `STT_AB` | Audio book | 有声书 | 20 / 49.5s | 33.5 |
| `STT_CS` | Children Speech | 儿童语音 | 20 / 82.6s | 46.7 |
| `STT_HS` | History | 历史 | 20 / 65.0s | 52.9 |
| `STT_MV` | Tibetan Movies | 电影 | 20 / 109.0s | 47.4 |
| `STT_NS` | Natural Speech | 自然口语 | 20 / 60.4s | 57.4 |
| `STT_NW` | News | 新闻 | 20 / 69.7s | 54.4 |
| `STT_PC` | Podcast | 播客 | 20 / 58.9s | 59.8 |
| `STT_TT` | Tibetan Teachings | 佛法开示 | 20 / 41.6s | 44.2 |

### 4.5 我们导出的 `metadata.csv`（8 列，160 行）

| 列 | 含义 | 示例 |
|---|---|---|
| `id` | 原始 `file_name` | `STT_AB00189_0080_275947_to_278950` |
| `audio_path` | 相对项目根路径 | `data/asr/openpecha_bench/wav/STT_AB00189_….wav` |
| `transcription` | `uni` 列 | `ཉན་རང་གི་ཐེག་པ་ཞེས་སྤྱད་ན` |
| `speaker` | 空（该数据集不提供说话人标注） | `` |
| `duration` | 音频头里的真实时长 | `3.003` |
| `dataset` | 标记 | `openpecha` |
| `dept` | 领域代码 | `STT_AB` |
| `grade` | 转写质量等级 | `3` |

统计：160 条 / **8.94 分钟** / 全部 `grade=3`（本样本恰好都命中质检通过的条目）。

---

## 5. 文件格式速查

| 格式 | 出现在哪 | 关键特性 | 读取方式 |
|---|---|---|---|
| **Parquet** | `data/{ocr,asr}/hf/data/*.parquet` | 列式、Schema 内嵌、可存二进制；`struct<bytes,path>` 承载媒体 | `pandas.read_parquet` |
| **CSV（UTF-8，无 BOM）** | `ground_truth.csv`、`metadata.csv`、`test.csv` | 文本含换行/逗号 → 必须加引号 | `pandas.read_csv(dtype={"id": str})` |
| **WAV（RIFF/PCM_16）** | `data/asr/*/wav/*.wav` | 16 kHz、单声道、16 bit；无损、可随机定位 | `soundfile.info` / `librosa.load(sr=16000)` |
| **PNG/JPEG（字节）** | `data/ocr/images/*.png` | **扩展名不等于真实编码**（415 JPEG + 57 PNG） | `PIL.Image.open` |
| **JSON** | `sample_ids.json`、`stats.json`、`run.json` | 结构与参数快照 | `json.load` |
| **Traineddata** | `tessdata/bod.traineddata` | Tesseract LSTM 模型包 | `--tessdata-dir tessdata` |

**CSV 的引号规则很重要**：OCR 输出保留换行（prompt 要求 "Preserve line breaks"），一行预测可能跨多行文本。因此 `run_ocr.py` 写 `predictions.csv` 时用 `csv.QUOTE_ALL`，否则重读会错行。

---

## 6. 数据流水线：谁生产、谁消费

```
                      ┌──────────────────────────────────────────────┐
 HuggingFace Hub ────>│ prepare_ocr_data.py / prepare_asr_data.py    │
 (hf_hub_download     │  解包 parquet / 下载 CDN / 转码 / 抽样        │
  local_dir=data/*/hf)└───────────────┬──────────────────────────────┘
                                      │
        ┌─────────────────────────────┴───────────────────────────┐
        ▼                                                         ▼
 data/ocr/images + ground_truth.csv + metadata.csv          data/asr/{nict_test,openpecha_bench}/
 + sample_ids.json + stats.json + official/                  wav/ + metadata.csv + sample_ids.json
        │                                                         │
        ▼                                                         ▼
   run_ocr.py  (yigdzin | tesseract)                        run_asr.py (6 个模型)
        │                                                         │
        ▼                                                         ▼
 outputs/ocr/<run_id>/                                     outputs/asr/<run_id>/
   predictions.csv  (id,transcription)                       predictions.csv
   details.csv      (+seconds,status,technology,script)      run.json
   run.json         (参数/设备/耗时/错误数)                    │
        │                                                         │
        ▼                                                         ▼
   score_ocr.py                                             score_asr.py
        │                                                         │
        ├─> details_scored.csv (+gt,cer,catastrophic)             ├─> predictions_scored.csv (+cer,swer,rtf)
        └─> reports/ocr_results.csv                               ├─> reports/asr_results.csv
                                                                  └─> summarize_asr.py → reports/asr_final_*.csv
        │                                                         │
        └───────────────> make_samples.py ────────────────────────┘
                          reports/samples/{ocr,asr}/ + *.md
```

| 文件 | 生产者 | 消费者 | 备注 |
|---|---|---|---|
| `data/ocr/hf/**` | `hf_hub_download` | `prepare_ocr_data.py` | 原始快照，不直接给模型用 |
| `data/ocr/images/` | `prepare_ocr_data.py` | `run_ocr.py` | 按 `id` 命名 |
| `data/ocr/ground_truth.csv` | `prepare_ocr_data.py` | `score_ocr.py`、`verify_data.py` | 与官方一致 |
| `data/ocr/metadata.csv` | 同上 | `run_ocr.py`（分桶字段）、`score_ocr.py` | 分技术与文字的统计来源 |
| `data/ocr/sample_ids.json` | 同上 | `run_ocr.py --split` | 可复现抽样 |
| `data/ocr/official/**` | `hf_hub_download` | `verify_data.py`、人 | 口径基准（只读） |
| `data/asr/hf/**` | `hf_hub_download` | `prepare_asr_data.py` | NICT parquet + OpenPecha CSV |
| `data/asr/*/wav/` | `prepare_asr_data.py` | `run_asr.py` | 16k mono PCM_16 |
| `data/asr/*/metadata.csv` | 同上 | `run_asr.py`、`score_asr.py`、`make_samples.py` | 含 `audio_path` |
| `data/asr/*/sample_ids.json` | 同上 | `run_asr.py --limit`、`summarize_asr.py` | 统一子集 |
| `outputs/ocr/<run>/predictions.csv` | `run_ocr.py` | `score_ocr.py` | `id,transcription`，QUOTE_ALL |
| `outputs/ocr/<run>/details*.csv` | `run_ocr.py` / `score_ocr.py` | 人、`make_samples.py` | 逐页记录 |
| `outputs/asr/<run>/predictions*.csv` | `run_asr.py` / `score_asr.py` | 人、`summarize_asr.py`、`make_samples.py` | 逐条记录 |
| `outputs/logs/*.log` | 各脚本重定向 | 人 | 原始 stdout |

---

## 7. 模型权重放在哪（不在项目目录里）

模型**没有**下载到 `data/`，而是走 HuggingFace 默认缓存：

```
~/.cache/huggingface/
├── hub/
│   ├── models--BDRC--tibetan-ocr/
│   │   ├── snapshots/<commit-sha>/model.safetensors -> ../../blobs/dc/dca65…   ← 软链接
│   │   └── refs/main
│   └── models--<org>--<name>/…
├── hub/blobs/XX/<sha256>           ← 真实权重（按内容寻址，去重）
└── xet/                            ← xet 分块传输缓存（下载加速用，可删）
```

| 模型 | 权重文件 | 大小 |
|---|---|---:|
| `BDRC/tibetan-ocr`（Yigdzin） | `model.safetensors` | **1.63 GB** |
| `openpecha/general_stt_base_model` | `model.safetensors` | 1.26 GB |
| `billingsmoore/…-whisper-small` | `model.safetensors` | 967 MB |
| `tenzin/whisper-small-tibetan` | `model.safetensors` | 967 MB |
| `billingsmoore/…-wav2vec2-base` | `model.safetensors` | 378 MB |
| `billingsmoore/…-whisper-base` | `model.safetensors` | 290 MB |
| `billingsmoore/…-whisper-tiny` | `model.safetensors` | 151 MB |
| `openai/whisper-{tiny,base,small}` | 仅 tokenizer/feature-extractor | 各 ~4 MB |
| 合计（`~/.cache/huggingface`） | | **≈ 6.5 GB** |

**为什么会疑惑"模型目录只有 460KB"**：`snapshots/` 里都是软链接，真实字节在 `hub/blobs/`。`du -sh` 默认不跟随软链接，所以看起来很小。检查真实占用用：

```bash
du -sh ~/.cache/huggingface/hub/blobs
```

**换位置**：设置 `HF_HOME=/path/to/big-disk`（或 `HF_HUB_CACHE`）后再跑脚本即可，代码无需修改。

---

## 8. 路径约定（容易踩的三个点）

1. **`metadata.csv` 里的 `audio_path` 是相对项目根目录**（如 `data/asr/nict_test/wav/x.wav`），不是相对 csv 所在目录。
   `run_asr.py` 用 `ASR_DIR.parent.parent`（= 项目根）拼接绝对路径；若移动数据目录，这两个常量要一起改（`scripts/common.py` 顶部）。
2. **OCR 侧只用 `id`，不存图片路径**。`run_ocr.py` 按 `data/ocr/images/{id}.png` 约定拼路径，所以**改文件名 = 改约定**，需同步改代码。
3. **`sample_ids.json` 是"数据与实验的契约"**。改了抽样必须重跑所有模型，否则跨模型对比失效（`summarize_asr.py` 也会过滤不到旧样本）。

---

## 9. 存储核算

| 项目 | 大小 | 可删？ |
|---|---:|---|
| `data/`（三套数据集 + 中间产物） | 960 MB | 可删，用脚本重下 |
| `outputs/`（14 个 run 的预测与日志） | 9 MB | 建议保留（测试记录） |
| `reports/`（汇总表、样例、文档） | 3 MB | 建议保留 |
| `tessdata/`（Tesseract bod） | 8.6 MB | 可删，重下 1 条命令 |
| `.venv/`（依赖） | 1.6 GB | 可删，`pip install -r requirements.txt` |
| `~/.cache/huggingface/`（模型） | 6.5 GB | 可删，首次运行会重下 |
| **合计** | **≈ 9.1 GB** | - |

只保留"结论可复现"的最小集：`scripts/` + `requirements.txt` + `reports/` + 文档（约 3.2 MB）。

---

## 10. 重建与完整性校验

### 10.1 从零重建

```bash
# 只重建 OCR（约 10 秒 + 121MB 下载）
./.venv/bin/python scripts/prepare_ocr_data.py --force

# 只重建 ASR（约 1 分钟 + 335MB + CDN 下载）
./.venv/bin/python scripts/prepare_asr_data.py

# Tesseract 模型
curl -L -o tessdata/bod.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/bod.traineddata
```

- `prepare_ocr_data.py --force` 会覆盖已导出的图片（不加则跳过已存在文件，只补缺）。
- 抽样可复现：`--seed 42`（默认），若改种子会生成不同的 `sample_ids.json`，需要重跑实验。
- OpenPecha 的音频来自外部 CDN，**若链接失效，脚本会跳过该条并告警**；重新运行时只会补下缺失文件。

### 10.2 校验

```bash
./.venv/bin/python scripts/verify_data.py
```

实际输出（2026-11-12，全部通过）：

```
--- ocr ---
PASS  OCR: 472 page images  (472 files)
PASS  OCR: ground_truth ids unique
PASS  OCR: ground_truth ids == metadata ids
PASS  OCR: image files == ground_truth ids
PASS  OCR: every page has transcription
PASS  OCR: sample images decodable  (first 40 checked, 0 broken)
PASS  OCR: prepared GT == official normalised GT  (0/472 differ)
PASS  OCR: official scorer present
PASS  OCR: tesseract bod model present
PASS  OCR: sample ids  (pilot 6 / sample 60)
PASS  OCR: sample ids unique and in GT
--- nict ---
PASS  NICT: 1547 test utterances  (1547)
PASS  NICT: 1547 wav files  (1547)
PASS  NICT: ids unique
PASS  NICT: wav names == metadata ids
PASS  NICT: 16 kHz  (50 checked)
PASS  NICT: mono  (50 checked)
PASS  NICT: metadata duration matches audio  (max diff 0.0000s)
PASS  NICT: every utterance has transcription
PASS  NICT: 3 speakers (disjoint from train)  (019, 070, 074)
PASS  NICT: sample ids  (sample 300 / all 1547)
PASS  NICT: sample ids in metadata
--- openpecha ---
PASS  OpenPecha: 160 sampled clips  (160)
PASS  OpenPecha: wav per clip  (160)
PASS  OpenPecha: ids unique
PASS  OpenPecha: 8 domains x 20
PASS  OpenPecha: all grade-3 (QC) transcriptions
PASS  OpenPecha: every clip has transcription
PASS  OpenPecha: 16 kHz mono  (20 checked)
============================================================
all checks passed
```

它检查的是**"我们的导出"与"官方/源数据"是否一致**，而不是模型精度：

| 检查项 | 目的 |
|---|---|
| id 集合三方一致（图片 / GT / metadata） | 防止解包时漏页或错位 |
| 我们的 GT vs 官方 GT（逐条 prepare 后比较） | 确认没有改动参考文本（实测 0/472 差异） |
| 音频采样率/声道/时长与 metadata 一致 | 防止转码错误导致"看起来能跑但指标错" |
| 抽样 id 是子集且数量正确 | 防止改抽样后实验不可比 |
| OpenPecha 全为 grade=3、8 领域各 20 | 防止抽样逻辑漂移 |

---

## 11. 已知限制与坑

| 项 | 说明 | 影响 |
|---|---|---|
| **OCR 图像扩展名 ≠ 真实编码** | 415 个 JPEG 命名为 `.png` | 无（按魔数识别）；需真 PNG 见 2.5 |
| **OCR id 不连续** | 0002–1075 之间大量空洞（匿名化） | 不能用 `range()` 遍历，必须读 `metadata.csv` |
| **OCR 仅公开 472/1070 页** | 私有 holdout 未公开 | 与榜单全量数字只能比"同一子集" |
| **NICT 仅 3 位说话人** | 官方设计（防止说话人泄漏） | 结论对"新说话人"的泛化置信度有限 |
| **NICT id 依赖 parquet 行序** | `nict_00000` 是行号，不是原文件名 | 必须保留 metadata.csv；原文件名在 parquet `audio.path` |
| **Whisper 448 token 上限** | 1/5 句子参考长度 >145 字符 | 长句截断，需报告 ≤150 字子集（见指标名词表） |
| **OpenPecha 音频走外部 CDN** | CloudFront 直链可能变动/限速 | 下载失败会跳过；重建时补下 |
| **OpenPecha 与 openpecha 训练集可能重叠** | 同源数据 | 该数据集上的分数应视为乐观上界 |
| **参考转写已规范化** | parquet 里的 GT 是 botok-normalized | 自己再跑一遍规范化反而可能引入 1 处差异（实测 1/472），所以 ref 侧只做 `prepare_text` |
| **方言覆盖** | 三套数据都是卫藏（拉萨）中心方言 | 安多/康巴结论不可外推 |
