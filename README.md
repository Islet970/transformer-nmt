# Transformer 中英神经机器翻译

基于 IWSLT 2017 TED Talks（zh→en）的 Transformer 机器翻译项目，包含：

- LSTM + Bahdanau Attention 基线；
- 基础 Transformer；
- 改进 Transformer（weight tying + label smoothing + beam search）；
- 2 组消融：去掉位置编码、单头注意力；
- BLEU / chrF、loss 曲线、长度分组评测、错误样例。

> 数据只使用中文→英文（zh→en）。训练默认在本地 RTX 4060 Laptop（8GB）上完成，也可以在 Kaggle GPU 上跑；两条路共用同一套代码。

## 0. 快速开始（本地 Windows）

```powershell
# 0) 依赖：PyTorch(CUDA) 已装好；Windows 上 sentencepiece 0.2.x 缺字符映射，固定 0.1.99
python -m pip install -r requirements.txt
python -m pip install sentencepiece==0.1.99

# 1) 数据：下载 WIT3 的 IWSLT17 zh-en 原始压缩包并转成 6 个文件（约 27 MB）
#    输出 data/iwslt17_zh_en/{train,valid,test}.{zh,en}
#    数量校验：231,164 / 879 / 8,549
python scripts/download_iwslt17.py
#    国内网络慢时：python scripts/download_iwslt17.py --mirror https://hf-mirror.com

# 2) 分词器：32k 共享 BPE（只在中英 train 上训练）
python -m data.prepare --data-dir data/iwslt17_zh_en --out-dir prepared --vocab-size 32000

# 3) 20 步 smoke test：确认数据/mask/loss/显存正常
powershell -File scripts/train_local.ps1 -RunName smoke -Epochs 1 -MaxSteps 20

# 4) 正式训练一个模型（其余配置同理）
powershell -File scripts/train_local.ps1 -Config configs/transformer_improved.yaml

# 5) 测试集评测（greedy / beam=4）
python evaluate.py `
  --checkpoint runs/transformer_improved/best.pt `
  --data-dir data/iwslt17_zh_en `
  --sp-model prepared/spm_zh_en.model `
  --split test --beam 4

# 6) 曲线与结果表
python analyze.py --runs-dir runs --out-dir figures
```

5 组实验依次跑（已完成的会跳过，`-Force` 重跑）：

```powershell
powershell -File scripts/run_all_local.ps1
```

数据、分词器、checkpoint 的落点：

```text
data/iwslt17_zh_en/           # 6 个文本文件（git 忽略）
data/_raw/zh-en.zip           # 原始压缩包缓存，可删
prepared/spm_zh_en.model      # 分词器
prepared/spm_zh_en.vocab
prepared/data_stats.json      # 数据统计，写报告用
runs/<run-name>/              # 每次实验的输出
```


## 1. 目录

```text
transformer-nmt/
├── configs/                   # 5 组实验配置
├── data/
│   ├── prepare.py             # 数据统计 + 训练 SentencePiece
│   ├── tokenizer.py           # SentencePiece 封装
│   └── dataset.py             # Dataset / collate
├── models/
│   ├── transformer.py         # Transformer
│   ├── lstm.py                # LSTM + Bahdanau attention
│   └── factory.py             # 根据 config 建模型
├── train.py                   # 训练、验证、checkpoint、resume
├── evaluate.py                # greedy / beam，BLEU/chrF
├── analyze.py                 # 曲线、结果表、错误样例
├── plot_attention.py          # 注意力热图（Transformer）
├── scripts/
│   ├── download_iwslt17.py    # 下载 WIT3 原始数据并转成 6 个文件（本地/ Kaggle 通用）
│   ├── prepare.sh             # 训练 tokenizer（bash）
│   ├── train_local.ps1        # 本地训练一个模型（Windows）
│   ├── run_all_local.ps1      # 本地依次跑 5 组（Windows）
│   ├── train_one.sh           # 只训练一个模型
│   ├── run_pair.sh            # 两张 T4 并行训练两个模型
│   ├── save_model.sh          # 导出一个模型的 zip
│   ├── run_all.sh             # 5 组训练
│   ├── evaluate_all.sh        # 5 组测试
│   └── kaggle_all.sh          # 一键跑完 5 组 + 测试 + 图表
├── tests/test_smoke.py        # 合成数据 smoke test
└── requirements.txt
```

## 2. 环境

```bash
pip install -r requirements.txt
```

Windows 上 `sentencepiece` 0.2.x 的 wheel 缺 `nmt_nfkc` 字符映射，训练分词器会报
`NOT_FOUND: No precompiled charsmap is found`，固定用 0.1.99：

```powershell
python -m pip install sentencepiece==0.1.99
```

Kaggle 已预装 PyTorch，通常只需：

```bash
pip install -q sentencepiece sacrebleu pyyaml
```

## 3. 本地代码检查

不依赖数据，只验证代码：

```bash
python -m py_compile config.py utils.py train.py evaluate.py analyze.py \
  data/*.py models/*.py

python -m unittest tests.test_smoke -v
```

预期：

```text
Ran 4 tests ... OK
```

## 4. Kaggle 流程

### 4.1 拉代码

```python
!git clone https://github.com/你的用户名/transformer-nmt.git /kaggle/working/transformer-nmt
%cd /kaggle/working/transformer-nmt
```

### 4.2 确认数据

本地训练的数据在 `data/iwslt17_zh_en`（见第 0 节），不需要这一步；下面只用于 Kaggle。

已确认的 Kaggle 挂载路径：

```text
/kaggle/input/datasets/hermionegrangerislet/translationdataset/data
```

检查：

```bash
ls -lh /kaggle/input/datasets/hermionegrangerislet/translationdataset/data
```

应有六个文件：

```text
train.zh train.en valid.zh valid.en test.zh test.en
```

### 4.3 准备 tokenizer

```bash
bash scripts/prepare.sh
```

默认输出：

```text
/kaggle/working/prepared/spm_zh_en.model
/kaggle/working/prepared/spm_zh_en.vocab
/kaggle/working/prepared/data_stats.json
```

如果 tokenizer 已存在，脚本会跳过，不会覆盖；需要强制重训时：

```bash
FORCE=1 bash scripts/prepare.sh
```

也可以手动：

```bash
python -m data.prepare \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --out-dir /kaggle/working/prepared \
  --vocab-size 32000
```

> 如果换 session 后 `/kaggle/working/prepared` 不在了，把 prepared 目录发布成 Kaggle Dataset，或重新运行一次 `prepare.sh`。

### 4.4 单步 smoke test

先跑 20 步，确认数据、mask、loss 正常：

```bash
python train.py \
  --config configs/transformer_improved.yaml \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --sp-model /kaggle/working/prepared/spm_zh_en.model \
  --epochs 1 \
  --batch-size 32 \
  --max-steps 20 \
  --limit-val 128 \
  --val-max-batches 4 \
  --run-name smoke
```

看日志里 loss 是否在下降、有没有 OOM。

### 4.5 跑 5 组实验

也可以一条命令跑完 5 组训练 + 测试 + 图表：

```bash
bash scripts/kaggle_all.sh
```

如果只想训练：

```bash
bash scripts/run_all.sh
```

默认 10 epochs，依次跑：

| 配置 | 实验 |
|---|---|
| `configs/lstm.yaml` | E1 LSTM + Attention |
| `configs/transformer.yaml` | E2 基础 Transformer |
| `configs/transformer_improved.yaml` | E3 改进 Transformer |
| `configs/ablation_no_pe.yaml` | A1 去掉位置编码 |
| `configs/ablation_1head.yaml` | A2 单头注意力 |

配置里默认 `patience: 0`，固定跑满 10 轮；如果需要早停，把对应 YAML 的 `patience` 改成 3–5。

已完成（存在 `runs/<name>/summary.json`）的实验默认会跳过，避免重复训练；要强制重跑：

```bash
SKIP_EXISTING=0 bash scripts/run_all.sh
```

每组输出：

```text
runs/<name>/
├── config.json
├── train.log
├── loss.csv
├── last.pt
├── best.pt
├── val_last.json
└── summary.json
```

如果一次 session 跑不完，单组续训：

```bash
python train.py \
  --config configs/transformer_improved.yaml \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --sp-model /kaggle/working/prepared/spm_zh_en.model \
  --output-dir runs \
  --run-name transformer_improved \
  --resume
```

### 4.6 测试

```bash
bash scripts/evaluate_all.sh
```

会为每组跑：

- greedy：完整 test（8,549 句）；
- beam=4 + length penalty=0.6：test 前 500 句（beam 逐句解码较慢）。

输出：

```text
runs/<name>/test_beam1.json
runs/<name>/test_beam1_predictions.txt
runs/<name>/test_beam4.json
runs/<name>/test_beam4_predictions.txt
```

### 4.7 生成图表

```bash
python analyze.py --runs-dir runs --out-dir figures
```

输出：

```text
figures/loss_curves.png
figures/val_bleu.png
figures/length_bleu.png
figures/results_table.csv
```

错误样例：

```bash
python analyze.py \
  --runs-dir runs \
  --out-dir figures \
  --predictions runs/transformer_improved/test_beam4_predictions.txt
```

注意力热图（只支持 Transformer checkpoint）：

```bash
python plot_attention.py \
  --checkpoint runs/transformer_improved/best.pt \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --sp-model /kaggle/working/prepared/spm_zh_en.model \
  --split test \
  --index 0 \
  --out figures/attention.png
```

## 4.8 一次只训练一个模型（推荐）

不要一次 Commit 跑 5 组。每次只跑一个模型，跑完立刻导出：

```bash
# 1. 训练一个模型
bash scripts/train_one.sh transformer_improved

# 2. 导出成可下载的 zip
bash scripts/save_model.sh transformer_improved
```

输出：

```text
exports/transformer_improved/
├── model.pt                  # 轻量权重，可直接用于 evaluate.py
├── config.json
├── train.log
├── loss.csv
├── summary.json
├── val_last.json
├── spm_zh_en.model
└── spm_zh_en.vocab
exports/transformer_improved.zip
```

在 Kaggle 左侧文件浏览器下载：

```text
/kaggle/working/transformer-nmt/exports/transformer_improved.zip
```

如果模型训练中断，已有 `last.pt`，可以续训：

```bash
python train.py \
  --config configs/transformer_improved.yaml \
  --data-dir /kaggle/input/datasets/hermionegrangerislet/translationdataset/data \
  --sp-model /kaggle/working/prepared/spm_zh_en.model \
  --output-dir runs \
  --run-name transformer_improved \
  --resume
```

## 4.9 两张 T4 并行训练两个模型

```bash
# 两个都没跑过
bash scripts/run_pair.sh transformer_improved lstm

# 如果 transformer_improved 已经跑了一部分，续训它
RESUME_A=1 bash scripts/run_pair.sh transformer_improved lstm
```

- GPU0 跑第一个，GPU1 跑第二个；
- 两边各自写 `runs/<name>/train.log`；
- 控制台日志在 `runs/<name>.console.log`；
- 每个模型一跑完就立即调用 `save_model.sh`，生成 `exports/<name>.zip`，不等另一个模型。

## 5. 关键配置

### 基础 Transformer

```yaml
d_model: 256
nlayers: 4
nheads: 8
d_ff: 1024
dropout: 0.1
label_smoothing: 0.0
tie_weights: false
batch_size: 32
max_len: 128
epochs: 10
warmup: 4000
```

### 改进 Transformer

```yaml
tie_weights: true
label_smoothing: 0.1
# 测试时 beam=4, length_penalty=0.6
```

### LSTM 基线

```yaml
model: lstm
d_model: 256
nlayers: 1
lr: 0.001
warmup: 0
```

## 6. 评测口径

- 方向：zh→en；
- BLEU：`sacreBLEU`，`tokenize=13a`；
- chrF：`sacreBLEU.corpus_chrf`；
- 验证集选模型，测试集只跑一次；
- 5 组实验使用同一数据、同一 tokenizer、同一 `max_len`、同一 batch、同一 seed。

## 7. 报告材料

对应 `translation/plan.md` 的章节：

| 报告章节 | 材料 |
|---|---|
| 2 数据集 | `data_stats.json`、长度分布 |
| 3.1 基础网络 | `models/transformer.py`、`models/lstm.py` |
| 3.2 改进 | weight tying、label smoothing、beam search |
| 4.2 训练 | `runs/*/train.log`、`loss.csv` |
| 4.3 测试 | `runs/*/test_beam*.json` |
| 4.4 可视化 | `figures/*.png` |
| 4.5 消融/对比 | `figures/results_table.csv` |

## 8. 注意事项

1. 不要用测试集调参；只看验证集。
2. 每次实验只改一个变量。
3. `best.pt` 可能不存在（如果 BLEU 一直为 0）；评测脚本会自动回退到 `last.pt`。
4. Kaggle 单次 session 约 12 小时；`last.pt` + `--resume` 支持续训。
5. 不要把 `data/`、`runs/`、`*.pt` 提交到 Git。
6. 代码截图只截核心片段：Dataset、MHA、Encoder/Decoder、训练循环、beam search。
