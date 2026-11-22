# 中英神经机器翻译项目计划（plan.md）

> 项目：基于 Transformer 的中英神经机器翻译
> 数据：IWSLT 2017 TED Talks（zh-en）
> 环境：本地写代码 / GitHub 同步 / Kaggle GPU 训练
> 报告模板：`translation/项目实践报告-模版.doc`
> 版本：v0.1（2026-11-21）

---

## 0. 一句话目标

在 IWSLT17 中英数据上，从零实现并训练 Transformer，完成：

1. **3 个主模型对比**：LSTM + Attention、基础 Transformer、改进 Transformer；
2. **2 个消融实验**：去掉位置编码、单头注意力；
3. **1 个对比分析**：在测试集上比较 BLEU / chrF / 参数量 / 训练时间；
4. 按模板完成项目实践报告。

> 说明：**神经机器翻译（NMT）是一类任务/方法，不是某一个模型。** LSTM+Attention 和 Transformer 都属于 NMT 模型。

---

## 1. 报告要求拆解

模板目录如下，最终报告严格按这个结构写：

```text
1 引言
  1.1 项目背景
  1.2 项目研究的主要内容
  1.3 主要问题与方法
2 数据集
  2.1 开源数据集介绍（二选一）
  2.2 自己数据集设计（二选一）
3 网络设计
  3.1 基础网络
  3.2 改进
4 训练与测试
  4.1 环境配置
  4.2 训练过程
  4.3 测试过程
  4.4 结果可视化
  4.5 消融实验和对比实验
5 总结与展望
  5.1 项目收获
  5.2 项目展望
  5.3 团队分工及协作
```

### 1.1 模板中的硬性要求

- 全文中文宋体小四，西文 Times New Roman，1.5 倍行距；
- 一级标题四号加粗，二级标题小四加粗，正文小四；
- 每章另起一页；
- 报告中的代码一律用**截图**展示，浅色背景、深色文字；
- 图居中，图题格式为“图4-1”这类“章节号-图序号”；
- 正文中要有“如图4-1所示”的指向性文字；
- 完成后更新目录；
- 删除模板里的红色说明文字和格式说明；
- 打印归档时封面、目录、评语表单面，正文双面。

### 1.2 对翻译任务的适配

模板里写的是“准确度”，翻译任务应改为：

- **BLEU**：主指标；
- **chrF**：辅助指标，中文目标时更稳；
- 参数量、训练时间、推理速度作为工程指标。

报告里要说明：翻译是序列生成任务，不用分类准确率。

### 1.3 数据集章节二选一

本项目使用公开数据集，因此：

- 保留 **2.1 开源数据集介绍**；
- **2.2 自己数据集设计** 写“本项目未自建数据集”，或按老师要求删除该小节。

---

## 2. 数据集

### 2.1 数据集选择

| 项目 | 内容 |
|---|---|
| 数据集 | IWSLT 2017 TED Talks |
| 语言方向 | **仅 zh→en（中文翻译为英文）** |
| 训练集 | 231,164 对（原始 231,266 行，剔除 1 个中英行数不一致的 talk，共 102 对） |
| 验证集 | 879 对 |
| 测试集 | 8,549 对 |
| 领域 | TED 演讲，口语化 |
| 许可证 | CC BY-NC-ND 4.0，课程/研究使用 |

### 2.2 数据位置

**本地（当前采用）**：六个文件放在仓库内，已被 `.gitignore` 忽略：

```text
D:\project\transformer-nmt\data\iwslt17_zh_en\
├── train.zh  train.en     # 231,164 对
├── valid.zh  valid.en     #     879 对
└── test.zh   test.en      #   8,549 对
```

数据来源是 WIT3 的 IWSLT17 zh-en 原始压缩包（HuggingFace 镜像，公开可下载）：

```text
页面：https://huggingface.co/datasets/IWSLT/iwslt2017   （config: iwslt2017-zh-en）
文件：https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/zh/en/zh-en.zip
```

一条命令完成下载 + 解压 + 转写 + 数量校验（约 27 MB）：

```bash
python scripts/download_iwslt17.py
```

> 国内网络慢时加镜像：`python scripts/download_iwslt17.py --mirror https://hf-mirror.com`。

**Kaggle（旧方案，仍可保留）**：

```text
/kaggle/input/datasets/hermionegrangerislet/translationdataset/data/
```

这是 Kaggle notebook 的挂载路径，只在挂载了该数据集的 notebook 内有效；对应页面
`https://www.kaggle.com/datasets/hermionegrangerislet/translationdataset` 匿名访问返回 404（私有或已改名），
所以本地训练改为直接下载上面的公开原始数据。

代码中统一使用 `--data-dir` 传入，启动时检查六个文件是否存在。

### 2.3 预处理

0. **按 TED talk 对齐**：原始 `train.tags` 把 talk 元数据（`<url>`、`<reviewer>` 等）和句子混排，且中英文件的元数据位置不完全一致，不能逐行硬拼；`download_iwslt17.py` 先按 `<url>` 切块、再在块内配对，并剔除中英行数不一致的 talk（1 个，102 对）；
1. Unicode NFKC 归一化；
2. 去首尾空白、去零宽字符；
3. 过滤空句；
4. 过滤长度异常：分词后源/目标长度比 > 2.5～3.0；
5. 句对去重；
6. 训练集/验证集/测试集之间做重叠检查；
7. 中文标点保留全角，不做过度清洗。

### 2.4 分词

- 工具：SentencePiece；
- 类型：BPE；
- 词表：32,000，中英共享；
- 训练数据：只使用 `train.zh` + `train.en`；
- 特殊符号：`pad=0, unk=1, bos=2, eos=3`；
- 验证集和测试集只做 encode，不参与训练。

产物：

```text
spm_zh_en.model
spm_zh_en.vocab
```

### 2.5 数据统计要写进报告

- 句对数量、平均长度、P50/P95/P99、最大长度；
- 空行率、重复率、过滤前后数量；
- `<unk>` 比例；
- 训练/验证/测试的长度分布图。

---

## 3. 代码结构

```text
transformer-nmt/
├── configs/
│   ├── lstm.yaml
│   ├── transformer.yaml
│   ├── transformer_improved.yaml
│   ├── ablation_no_pe.yaml
│   ├── ablation_1head.yaml
│   └── ablation_no_tie.yaml
├── data/
│   ├── prepare.py          # 数据检查、统计、导出
│   ├── tokenizer.py        # SentencePiece / char tokenizer
│   └── dataset.py          # Dataset、collate、长度分桶
├── models/
│   ├── transformer.py      # Transformer、MHA、PE、mask
│   ├── lstm.py             # LSTM + Bahdanau attention
│   └── factory.py          # 根据 config 建模型
├── train.py                # 训练、验证、checkpoint、resume
├── evaluate.py             # greedy / beam、BLEU、chrF
├── analyze.py              # 曲线、结果表、错误分析
├── plot_attention.py       # 注意力热图
├── utils.py                # seed、logging、device、路径检查
├── scripts/
│   ├── prepare.sh          # 训练 tokenizer
│   ├── run_all.sh          # 5 组训练
│   ├── evaluate_all.sh     # 5 组测试
│   └── kaggle_all.sh       # 一键跑完
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 3.1 模块职责

| 文件 | 写什么 |
|---|---|
| `configs/*.yaml` | 每组实验的超参数，保证可复现 |
| `data/prepare.py` | 统计、过滤、去重、导出文本 |
| `data/tokenizer.py` | 训练/加载 SentencePiece，统一特殊符号 |
| `data/dataset.py` | `TranslationDataset`、`collate_fn`、长度分桶 |
| `models/transformer.py` | Multi-Head Attention、Encoder/Decoder、Positional Encoding、mask |
| `models/lstm.py` | BiLSTM 编码器 + LSTM 解码器 + Bahdanau attention |
| `models/factory.py` | `build_model(cfg, vocab_size)` |
| `train.py` | AMP、梯度裁剪、warmup、验证 BLEU、保存 checkpoint、resume |
| `evaluate.py` | greedy/beam 解码，输出 BLEU/chrF |
| `analyze.py` | loss/BLEU 曲线、结果表、错误样例 |
| `plot_attention.py` | 注意力热图 |
| `scripts/*.sh` | tokenizer、5 组训练、测试、一键流水线 |
| `README.md` | 环境、数据路径、训练命令、结果表 |

### 3.2 训练脚本接口

```bash
python train.py \
  --config configs/transformer.yaml \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --sp-model /kaggle/input/datasets/hermionegrangerislet/translationdataset/spm_zh_en.model \
  --epochs 1 \
  --batch-size 32 \
  --run-name smoke_transformer
```

必须支持：

```text
--config
--data-dir
--sp-model
--epochs
--batch-size
--resume
--run-name
--device
--max-steps       # 只跑若干步，用于 smoke test
```

### 3.3 每次实验的输出

```text
runs/<run-name>/
├── config.json
├── train.log
├── loss.csv
├── best.pt
├── last.pt
├── test_bleu.json
└── predictions.txt
```

报告里的表格和曲线全部从这些文件生成，不靠手抄。

---

## 4. 模型设计

### 4.1 M1：LSTM + Bahdanau Attention（基线）

- 词嵌入：256；
- 编码器：1 层双向 LSTM，hidden=256；
- 解码器：1 层 LSTM，hidden=256；
- 注意力：Bahdanau 加性注意力；
- dropout：0.1；
- 解码：greedy；
- 训练：Adam，lr=1e-3，batch=32，epochs=10。

作用：作为“传统 NMT”基线，证明 Transformer 的收益。

### 4.2 M2：基础 Transformer（原始模型）

- `d_model=256`；
- `nlayers=4`；
- `nheads=8`；
- `d_ff=1024`；
- dropout=0.1；
- 正弦位置编码；
- Post-LN；
- 无 weight tying；
- 无 label smoothing；
- 解码：greedy；
- 优化器：Adam(β1=0.9, β2=0.98, eps=1e-9)；
- warmup=4000；
- batch=32；
- max_len=128；
- epochs=10。

> 为了控制变量，M2 是“简化版基础 Transformer”，不启用 weight tying 和 label smoothing；M3 再加改进。报告中要说明这一点，避免和论文完全复现混淆。

### 4.3 M3：改进 Transformer（你的模型）

在 M2 基础上加：

1. **Weight tying**：输出层与目标词嵌入共享权重；
2. **Label smoothing=0.1**；
3. **Beam search=4 + length penalty=0.6**。

其余结构与 M2 完全一致，保证对比公平。

### 4.4 模型对比关系

```text
M1 LSTM + Attention
        ↓
M2 基础 Transformer
        ↓
M3 改进 Transformer
```

---

## 5. 实验矩阵

### 5.1 主实验（3 个模型）

| 编号 | 实验 | 说明 | 是否重训 |
|---|---|---|---|
| E1 | LSTM + Attention | 基线 | 是 |
| E2 | 基础 Transformer | 原始模型 | 是 |
| E3 | 改进 Transformer | 你的模型 | 是 |

### 5.2 消融实验（2 个）

| 编号 | 实验 | 改什么 | 验证什么 |
|---|---|---|---|
| A1 | E3 w/o Positional Encoding | 去掉位置编码 | 位置编码是否必要 |
| A2 | E3 with 1 Head | `nheads=1` | 多头注意力是否有效 |

消融做 **A1 + A2**。

### 5.3 解码对比（不用重训）

| 编号 | 实验 | 说明 |
|---|---|---|
| D1 | Greedy | 直接取 argmax |
| D2 | Beam=4 + length penalty | 用 E3 的 checkpoint 重新解码 |

### 5.4 对比实验

不是再训练一个模型，而是把 E1/E2/E3 + 消融结果放在一张表里：

| 模型 | 参数量 | 训练时间 | Dev BLEU | Test BLEU | chrF |
|---|---|---|---|---|---|
| LSTM + Attention | 待填 | 待填 | 待填 | 待填 | 待填 |
| 基础 Transformer | 待填 | 待填 | 待填 | 待填 | 待填 |
| 改进 Transformer | 待填 | 待填 | 待填 | 待填 | 待填 |
| w/o Positional Encoding | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 Head | 待填 | 待填 | 待填 | 待填 | 待填 |
| w/o Weight Tying | 待填 | 待填 | 待填 | 待填 | 待填 |

### 5.5 公平对比规则

1. 同一份 train/valid/test；
2. 同一 tokenizer（分词消融除外）；
3. 同一 `max_len`、batch size、epoch 数、随机种子；
4. 用验证集选模型，测试集只跑一次；
5. 报告参数量、训练时间、BLEU、chrF；
6. 主对比建议跑 2–3 个 seed，报告均值±标准差；时间不够则固定 seed=42，并在报告中说明。

### 5.6 预计计算量

统一使用 `d_model=256, nlayers=4`，IWSLT17 全量：

- **本地 RTX 4060 Laptop 8GB 实测**（batch=32、AMP、max_len=128）：
  - 约 7 s / 100 步（≈14 步/s），7,223 步/epoch ≈ 8–9 分钟；
  - 单组 10 epoch ≈ 1.5 小时（含每轮 879 句验证）；
  - 5 组训练 ≈ 7–8 小时；评测约 1–2 小时。
  - batch=64 实测更慢（22 s/100 步），保持 batch=32。
- 参考值（T4）：单组训练约 1.5–2 小时（10 epoch，含每轮验证）；
- E1/E2/E3 + A1/A2 共 5 次训练，本地约 7–8 小时；
- 评测另算，约 2–3 小时；
- D1/D2 只解码，不重训。

如果时间不够：

- 消融可只做 A1/A2；
- 或把消融训练轮数减到 5 轮，但要与主实验说明差异；
- 不要削减测试集。

---

## 6. 训练与评测流程

### 6.1 本地（Windows + RTX 4060 Laptop 8GB，承担完整训练）

本地现在做完整训练，不再只是 smoke test。依赖（Windows 上 sentencepiece 0.2.x 缺字符映射，固定 0.1.99）：

```powershell
python -m pip install -r requirements.txt
python -m pip install sentencepiece==0.1.99

# 1) 下载并转写数据（约 27 MB，校验 231,164 / 879 / 8,549）
python scripts/download_iwslt17.py

# 2) 训练 tokenizer（32k BPE，中英共享）
python -m data.prepare --data-dir data/iwslt17_zh_en --out-dir prepared --vocab-size 32000

# 3) 20 步 smoke test：确认 forward/backward、显存、速度
powershell -File scripts/train_local.ps1 -RunName smoke -Epochs 1 -MaxSteps 20

# 4) 代码静态检查
python -m py_compile config.py utils.py train.py evaluate.py analyze.py data/prepare.py data/tokenizer.py data/dataset.py models/transformer.py models/lstm.py models/factory.py
python -m unittest tests.test_smoke -v
```

显存：`d_model=256 / nlayers=4 / batch=32 / max_len=128 / AMP` 在 8GB 上余量较大，可尝试 batch=64 看速度是否更好。
训练速度：先跑 1 epoch 实测耗时，再按 5.6 决定总轮数（不要一上来就排 5×10 轮）。

### 6.2 Kaggle CPU

1. 挂载数据；
2. 检查六个文件；
3. 跑一个 batch，确认 `src.shape`、`tgt.shape`；
4. 跑 `--max-steps 20`，确认 loss 能反传。

### 6.3 Kaggle GPU

1. 先跑 1 epoch，记录耗时和验证 BLEU；
2. 确认 checkpoint 正常保存；
3. 再跑完整 10 epochs；
4. 每轮保存 `last.pt`，最佳保存 `best.pt`；
5. 用 `best.pt` 在 test 上评测一次；
6. 跑 beam search 对比；
7. 把 `runs/` 结果下载或保存到 Kaggle Dataset。

### 6.4 评测

```text
sacreBLEU：
  zh→en: tokenize=13a
chrF：
  sacreBLEU 的 chrF
```

报告必须写清楚：

- 用的 tokenizer；
- 是否 case-sensitive；
- beam size；
- 测试集版本；
- 是否 ensemble。

---

## 7. 结果可视化

至少做以下图：

1. **训练/验证 loss 曲线**（图4-1）；
2. **验证 BLEU 随 epoch 变化**（图4-2）；
3. **不同长度区间的 BLEU**：1–10、11–20、21–30、31–40、40+（图4-3）；
4. **注意力热图**：取一个中英样本，画 encoder-decoder cross-attention（图4-4）；
5. **错误分析表**：漏译、重复、术语、语序、未登录词，各举 2–3 例。

可选：

- 不同 beam size 的 BLEU；
- 不同模型参数量/训练时间对比；
- 训练集/验证集 loss 与 BLEU 的双轴图。

---

## 8. 报告章节与材料映射

| 报告章节 | 写什么 | 对应代码/结果 |
|---|---|---|
| 1.1 背景 | 中英翻译需求、NMT 发展 | 参考文献 01/02/08 |
| 1.2 主要内容 | 实现 Transformer、对比、消融 | 本计划 |
| 1.3 主要问题与方法 | OOV、长句、语序、数据质量；BPE、注意力、Transformer | 参考文献 06/09/10 |
| 2.1 开源数据集 | IWSLT17 介绍、统计、预处理 | `prepare.py`、`data_stats.json` |
| 3.1 基础网络 | LSTM+Attention、Transformer 结构 | `models/*.py`、结构图 |
| 3.2 改进 | weight tying、label smoothing、beam | `models/transformer.py`、`evaluate.py` |
| 4.1 环境配置 | Python、PyTorch、Kaggle T4、Git | `requirements.txt`、README |
| 4.2 训练过程 | 超参、loss、耗时、问题 | `train.log`、loss 曲线 |
| 4.3 测试过程 | BLEU/chrF、greedy/beam | `test_bleu.json` |
| 4.4 结果可视化 | loss/BLEU/长度/注意力/错误 | `analyze.py`、图表 |
| 4.5 消融与对比 | 5 组结果表 | `runs/*` |
| 5.1 收获 | 遇到的问题与解决 | 开发记录 |
| 5.2 展望 | 数据增强、回译、预训练模型 | 参考文献 11/12/13 |
| 5.3 分工 | 按个人独立完成撰写 | 开发记录 |

### 8.1 建议的图表编号

```text
图3-1  LSTM + Attention 结构图
图3-2  Transformer 结构图
图4-1  训练/验证 loss 曲线
图4-2  验证 BLEU 曲线
图4-3  不同长度区间 BLEU
图4-4  中英注意力热图
表4-1  主模型对比
表4-2  消融实验
表4-3  错误分析示例
```

### 8.2 代码截图清单

模板要求代码截图，建议只截核心片段：

1. `data/dataset.py` 的 `__getitem__` + `collate_fn`；
2. `models/transformer.py` 的 Multi-Head Attention；
3. `models/transformer.py` 的 Encoder/Decoder forward；
4. `train.py` 的训练循环（shift、mask、loss）；
5. `evaluate.py` 的 beam search。

不要截整个文件，也不要截大段无关代码。

---

## 9. 时间安排

按 7 个工作日估算：

| 阶段 | 工作 | 产出 |
|---|---|---|
| Day 1 | 建仓库、搬当前 Notebook 代码、模块化 | 可 import 的项目骨架 |
| Day 2 | 数据检查、SentencePiece、DataLoader | 数据统计、tokenizer、一个 batch |
| Day 3 | 训练脚本、smoke test、M2 跑 1 epoch | loss 曲线、checkpoint |
| Day 4 | E1/E2/E3 主实验 | 3 组结果 |
| Day 5 | A1/A2 消融 + beam | 消融表、beam 表 |
| Day 6 | 评测、画图、错误分析 | 报告 4.3–4.5 材料 |
| Day 7 | 写报告、排版、截图、更新目录 | 最终报告 |

项目按“个人独立完成”推进，不设固定截止日期；训练可以分多次 Kaggle session 完成，代码必须支持断点续训。

---

## 10. 风险与应对

| 风险 | 应对 |
|---|---|
| Kaggle 12 小时会话限制 | 每轮存 `last.pt`，支持 `--resume` |
| GPU 额度不够 | 消融只做 2 个；模型统一 256/4；不跑 WMT 全量 |
| LSTM 基线训练慢/不收敛 | 用 1 层 LSTM、lr=1e-3、梯度裁剪；只作为基线 |
| BLEU 太低 | 先检查数据对齐、mask、shift；再增加 epoch/调 lr |
| 包版本冲突 | `requirements.txt` 固定版本；Kaggle 上先 `pip install -e .` 或按 README |
| 数据路径变化 | 所有路径用 `--data-dir` / `--sp-model` 传入，启动时检查六个文件 |
| 代码截图太多 | 只截核心 5 段，其余放 GitHub 链接 |
| 报告格式不符 | 直接套模板样式，最后更新目录、删红色说明 |

---

## 11. 已确认约束与仍需确认

### 11.1 已确认（不再变更）

1. **项目按个人独立完成推进**，报告 5.3 写个人完成；不拆分团队任务。
2. **不设固定截止日期**，按质量优先推进；训练可分多次 Kaggle session 完成。
3. **固定 5 次训练**：
   - E1 LSTM + Attention
   - E2 基础 Transformer
   - E3 改进 Transformer
   - A1 去掉位置编码
   - A2 单头注意力
   - beam search 只解码，不增加训练次数。
4. **只做 zh→en（中文翻译为英文）**，不做 en→zh。
5. **Kaggle 数据路径已确认**：
   `/kaggle/input/datasets/hermionegrangerislet/translationdataset/data`
6. 本地（RTX 4060 Laptop 8GB）承担正式训练；Kaggle 作为备选（显卡更强、可长时间跑），两条路都用同一套代码和 `--data-dir` / `--sp-model` 参数。

### 11.2 仍需确认（不影响现在开工，可后续补充）

1. **Kaggle 每周 GPU 额度还剩多少？**
   - 5 次训练按 1.5–2 小时/次估算；
   - 如果额度紧张，优先保证 E1/E2/E3，再压缩消融 epoch。

2. **“改进”是否必须是算法创新？**
   - 当前按工程改进设计：weight tying、label smoothing、beam search；
   - 如果老师明确要求算法创新，再加 relative position / copy mechanism / back-translation。

3. **是否允许使用预训练模型？**
   - 当前按从零训练；
   - 如果允许，可加 OPUS-MT / mBART 微调作为额外对比，但不计入 5 次训练。

4. **报告是否需要摘要/英文摘要？**
   - 模板没有摘要页；如学校另有要求再补。

5. **目标 BLEU 是多少？**
   - 当前按 256/4 配置设定合理目标：验证 BLEU 18–25、测试 BLEU 15–22；
   - 如果要求 25+，需要 512/6、更多 epoch 或回译。

---

## 12. 交付物清单

- [ ] GitHub 仓库：代码、configs、README、requirements
- [ ] `runs/`：每次实验的 config、log、checkpoint、BLEU
- [ ] 数据统计：`data_stats.json` + 长度分布图
- [ ] 模型结构图：LSTM+Attention、Transformer
- [ ] 训练曲线：loss、BLEU
- [ ] 消融表、对比表
- [ ] 注意力热图、错误分析
- [ ] 项目实践报告（按模板）
- [ ] 代码截图（5 段核心代码）
- [ ] 可选：答辩 PPT / 演示脚本

---

## 13. 立即执行的下一步

1. 本地建 `transformer-nmt/` 仓库，把当前 Notebook 能跑的代码搬进 `models/`、`data/`、`train.py`；
2. 本地做 `py_compile` + 随机张量测试；
3. 推到 GitHub；
4. Kaggle CPU 挂载数据，跑一个 batch；
5. Kaggle GPU 跑 M2 的 1 epoch；
6. 确认 OOM、速度、BLEU 正常后，再跑 E1/E2/E3 和 A1/A2。

---

## 14. 当前进度

- [x] 报告要求拆解
- [x] 数据集与路径确认
- [x] 5 组实验矩阵
- [x] 本地代码骨架：`transformer-nmt/`
- [x] 数据准备与 SentencePiece 训练脚本
- [x] Transformer 与 LSTM + Attention
- [x] train / evaluate / analyze / attention 脚本
- [x] 本地合成数据 smoke test
- [ ] Kaggle 上训练 tokenizer
- [ ] Kaggle 上跑 5 组训练
- [ ] 生成 BLEU/图表/错误分析
- [ ] 按模板写报告

代码位置：`transformer-nmt/`；运行说明见 `transformer-nmt/README.md`。

---

## 附：核心命令速查

### 本地检查

```bash
python -m py_compile train.py models/transformer.py data/dataset.py
python -c "from models.transformer import Transformer; print('ok')"
```

### Kaggle 拉代码

```python
!git clone https://github.com/你的用户名/transformer-nmt.git /kaggle/working/transformer-nmt
%cd /kaggle/working/transformer-nmt
```

### Kaggle 确认数据

```python
!find /kaggle/input -name "train.zh" -o -name "train.en"
```

### Kaggle 训练

```python
!python train.py \
  --config configs/transformer.yaml \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --sp-model /kaggle/input/datasets/hermionegrangerislet/translationdataset/spm_zh_en.model \
  --epochs 1 \
  --batch-size 32 \
  --run-name smoke_transformer
```

### 评测

```python
!python evaluate.py \
  --checkpoint runs/transformer_improved/best.pt \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --sp-model /kaggle/input/datasets/hermionegrangerislet/translationdataset/spm_zh_en.model \
  --split test \
  --beam 4
```
