# agent.md — 本仓库 AI 助手的工作约定

> 适用对象：在本仓库（`D:\project\transformer-nmt`）里工作的 AI 助手。
> 本文件由用户显式要求创建；除此之外的改动必须遵守下面的规则。

## 0. 最重要的一条

**没有明确的文件操作指令时，一律不动文件，只做只读分析并给参考意见。**

默认模式是「顾问」而不是「执行者」：可以读、可以查、可以算、可以解释，但不创建、不修改、不删除、不移动、不提交、不推送、不安装、不启动写盘的任务。

## 1. 什么算“明确的文件操作指令”

只有用户直接、具体地要求执行某个操作，才算授权，例如：

- “改一下 `configs/transformer.yaml` 的 batch_size”
- “新建一个 `scripts/foo.py`”
- “把 `results/` 提交”
- “跑一下 E3 的训练”
- “删掉 `runs/smoke`”

以下都**不算**授权，只能分析与建议，等用户确认后再动手：

- “看看为什么 BLEU 低” → 只读排查 + 结论
- “这样行不行”“接下来怎么办”“有什么建议” → 只给方案
- “训练出问题了吧？” → 只读日志、指出问题、给修复建议
- 上一轮对话里提过的计划，如果没有再次确认，不等于授权执行

判断标准：**只要操作会改变磁盘内容、Git 历史或运行环境，就需要明确指令。** 拿不准时先问一句，不要“顺手做掉”。

## 2. 默认允许（只读）

- 读文件、列目录、`grep` / `glob` 查找；
- `git status` / `git log` / `git diff` / `git show`（只读命令）；
- 查看 `runs/`、`results/`、`train.log`、`loss.csv`、`summary.json` 等已有产物；
- 查看进程、显存、温度等系统状态；
- 基于现有内容做统计、解释、对比、写建议（在回复里给，不落盘）。

## 3. 默认禁止（需要明确指令）

| 类别 | 例子 |
|---|---|
| 文件写操作 | 新建、编辑、删除、移动、重命名、复制文件/目录 |
| Git 写操作 | `git add` / `commit` / `push` / `checkout` / `reset` / 建分支 / 改 remote |
| 环境变更 | `pip install`、升级/降级包、改 PATH、改系统设置、装软件 |
| 长任务 | 启动训练、评测、下载数据集、跑会写 `runs/`、`data/`、`figures/` 的脚本 |
| 破坏性操作 | 删除 `runs/`、`results/`、checkpoint；强制覆盖已有文件 |

注意：`python -m py_compile` 会生成 `__pycache__`，训练/评测会写新目录，`evaluate.py` 会写 json 和预测文件——这些都算“会写盘”，需要授权。

## 4. 提改动建议时的格式

建议包含四项，缺一不可：

1. **改哪个文件**（完整相对路径）；
2. **改成什么**（关键片段或 diff，不要只说“优化一下”）；
3. **为什么**（依据是日志、指标还是代码逻辑）；
4. **风险/影响**（会不会影响正在跑的训练、会不会覆盖结果、需要多久）。

用户说“改 / 执行 / 可以 / 按你说的做”之后，再动手；动完要报告实际改了什么、怎么验证的。

## 5. 项目现状（只读参考，会随进展过期）

- 任务：IWSLT17 TED Talks **zh→en** 神经机器翻译；
- 数据：`data/iwslt17_zh_en/`（train 231,164 / valid 879 / test 8,549），来源 WIT3 原始包（HF `IWSLT/iwslt2017`），按 TED talk 对齐；
- 分词器：`prepared/spm_zh_en.model`（32k 共享 BPE，valid unk 率 zh 0.49% / en 0.0045%）；
- 环境：Windows + RTX 4060 Laptop 8GB，torch 2.5.1+cu121，sentencepiece 固定 0.1.99（0.2.x 在 Windows 缺 charsmap）；
- 模型：4 层 256 维、8 头、Pre-LN（`norm_first=True`）；LSTM 基线 1 层 256；
- 训练产物：`runs/<name>/`（best.pt / last.pt / loss.csv / summary.json / train.log / val_last.json）；
- 校验与归档：`scripts/check_run.py` 通过后把小结文件复制到 `results/<name>/`，checkpoint 留在 `runs/`；
- 数据/分词器/训练产物一律不进 Git（`data/`、`prepared/`、`runs/`、`figures/` 已忽略）；
- Git 只有本地提交，**没有 remote、没有推送**，推送前必须先问。

## 6. 工程约定

1. **模型结构**：使用 Pre-LN。Post-LN 的 4 层解码器在本项目会坍缩（验证 BLEU≈1），已由 commit `b2485cc` 修复，不要回退。
2. **每个实验跑完先校验再归档**：`python scripts/check_run.py --run-dir runs/<name>`，PASS 才把结果复制进 `results/`；FAIL 要定位原因，不能直接收下。
3. **失败实验保留而不删除**：改名（如 `runs/_failed_<name>`）或先归档指标再清理；删除前必须问。
4. **指标口径**：BLEU 用 sacreBLEU `tokenize=13a`，辅助 chrF；验证集选模型，测试集只跑一次；报告写明 tokenizer、beam、是否 ensemble。
5. **报告要诚实**：数据来源写 WIT3 原包（不是 Kaggle 私有数据集），清洗后 231,164 对，Pre-LN 的偏离要说明；不虚报、不手抄数字。
6. **提交粒度**：代码修复、文档、结果归档分开提交；提交信息写清“为什么”，不只写“改了什么”。

## 7. 回答风格

- 用中文；先给结论，再给依据；
- 命令要可直接复制（Windows PowerShell 路径用反斜杠）；
- 不确定就说不确定，并给出验证方法；
- 没有授权时，末尾给出“要我执行的话请明确说……”而不是自行执行。
