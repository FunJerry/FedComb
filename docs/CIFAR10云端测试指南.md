# CIFAR-10 三种方法云端真实训练

目标：FedComb、FedComb-Random、FedComb-Greedy。代码与本地 60000 张 CIFAR-10 PNG 上传到用户确认的 AutoDL 实例，不是 Google Cloud。

## 1. 方法口径

三组共用 `configs/cifar10_cloud.json`，只改变分配策略：

| 方法 | 分配策略 | 蒸馏 | 聚合 |
|---|---|---|---|
| FedComb | dynamic | adaptive | mean |
| FedComb-Random | 随机长度及合法起点，超预算剔除，不重抽 | adaptive | mean |
| FedComb-Greedy | 随机起点 0～15，允许 1～2 片，选择该起点下最大可行参数包 | adaptive | mean |

共同使用 ResNet-50、Dirichlet=0.5、100 客户端、每轮 10 位、每位 500 张、batch=100、本地 5 epoch、温度 2。dynamic/random 最少 3 片；用户确认 Greedy 使用原作者不同的长度约束。三组数据集划分保持一致，Random 不自动变成 IPC。

当前资源和评分的修正、Greedy 新定义以 [算法意图对齐修改说明](算法意图对齐修改说明.md) 为准；不是逐数值复刻旧日志。旧的 cifar10_paper_baselines_s42 结果必须保留，使用新输出目录 cifar10_intent_v2_s42。虚拟资源预算和估算耗时仍是模拟值，实测 GPU 峰值及本机实际时间单独记录。

旧 `results/cifar10_s42` 和 `results/cifar10_paper_baselines_s42` 均属于此前实现，不能当作本次新流程结果。重新实验必须换目录，如 `cifar10_intent_v2_s42`；不要在旧结果上使用 `--resume`。

## 2. 服务器位置与安装

进入上传目录：

```bash
cd /root/autodl-tmp/fedcomb_cifar10
bash scripts/setup_cloud.sh
```

使用数据盘，避免把数据和结果放到小系统盘。安装脚本复用镜像中的 CUDA PyTorch，不用 macOS 的 requirements-lock.txt 覆盖服务器 GPU 版本。每次实验 manifest 记录云端实际版本。

若缺少某个依赖，安装对应依赖即可；不要随意替换已能识别 GPU 的 torch/torchvision 配对。

## 3. 先检查，不启动正式训练

```bash
.venv/bin/python scripts/verify_upload.py
.venv/bin/python scripts/check_environment.py --require-cuda
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/train_cifar10.py --method all --profile paper --output preview --dry-run
```

检查 CIFAR-10 训练/测试数应为 50000/10000，CUDA=True。最后一个命令只显示配置，不训练。

## 4. 三种运行规模

- `smoke`：真实 CIFAR-10 和真实宽度 ResNet-50，但仅 4 客户端、每轮 2 位、每位 16 张、本地 1 epoch、2 轮；只用于云端机制验证，不能作为最终对比精度。
- `quick`：保留论文规模的客户端、数据和本地训练设置，仅运行 3 轮。用于检查正式运行是否正常。
- `paper`：完整配置，各 1000 轮。单个种子不是多种子统计结论。

先运行少轮检查：

```bash
.venv/bin/python scripts/train_cifar10.py --method all --profile quick --output intent_v2_quick
```

如果希望覆盖其它轮数，可加 `--rounds 10`，实际配置会记录到结果。

## 5. 分别启动三种方法

```bash
bash scripts/run_fedcomb.sh --profile paper --seed 42 --output intent_v2_fedcomb_s42 --background
bash scripts/run_random.sh --profile paper --seed 42 --output intent_v2_random_s42 --background
bash scripts/run_greedy.sh --profile paper --seed 42 --output intent_v2_greedy_s42 --background
```

**单张 GPU 不要同时启动上述三个任务。** 每个方法执行结束后再启动下一个，否则会争抢 GPU，实测时间也不再公平。

推荐一个任务按顺序跑完三种：

```bash
.venv/bin/python scripts/train_cifar10.py --method all --profile paper --seed 42 --output cifar10_intent_v2_s42 --background
```

后台进程在浏览器断开后继续运行，但实例停机/释放会停止进程。脚本不会调整付费配置或自动关机。完整训练会持续使用已租赁 GPU，需自行关注费用和磁盘容量。

## 6. 查看进度与结果

```bash
tail -n 30 results/cifar10_intent_v2_s42/console.log
tail -f results/cifar10_intent_v2_s42/console.log
.venv/bin/python scripts/show_results.py results/cifar10_intent_v2_s42
```

查看图表（需要 matplotlib，通常镜像已安装）：

```bash
.venv/bin/python scripts/show_results.py results/cifar10_intent_v2_s42 --plot
```

输出 `comparison.png`：精度—轮次、精度—实测时间、测试 CE。Jupyter 文件浏览器中双击可看，右键下载。没有完整 summary 的方法标记“未完成”，不会伪装成最终结果。

每个方法的目录：

```text
results/cifar10_intent_v2_s42/
├── console.log、process.json、job_status.json
├── fedcomb/seed-42/
├── fedcomb-random/seed-42/
└── fedcomb-greedy/seed-42/
```

每个 seed 目录包含 metrics.csv、rounds.jsonl、config.json、manifest.json、summary.json、best.pt、周期 checkpoint。精度在原始文件中是 0～1，查看脚本显示为百分比；最高精度与最后精度分开报告。

## 7. 中断恢复

```bash
.venv/bin/python scripts/train_cifar10.py --method all --profile paper --seed 42 --output cifar10_intent_v2_s42 --resume --background
```

必须使用原始配置、相同源码。已完成的方法跳过；未完成的方法从最近完整检查点恢复。保存全局权重、成熟度、资源状态和三条随机流；当前不再需要跨轮历史损失。旧策略检查点不能用本版续训。

检查点之后已经写出的日志会先备份为 `.before_resume_*`，随后从检查点重放，避免重复统计。默认每 50 轮保存，最多可能重跑不足 50 轮；最后一轮也保存。最佳权重单独保留，不每次改进都永久新增大文件。

旧版没有 round_state 的检查点不能精确恢复。CPU 同环境的不中断/恢复权重一致已测试；跨 PyTorch、硬件或源码变化不承诺位级一致。后台恢复前确认旧进程已退出，不能让两个进程写同一个结果目录。

## 8. 数据与上传安全

上传包只包含源码、配置、脚本、测试、文档及 CIFAR-10；不包含 .git、.idea、.venv、SSH 信息、CIFAR-100/SVHN、旧 results。逐文件 SHA256 清单验证传输。

不能在 smoke/quick 的准确率基础上宣称三种方法优劣。正式比较至少看完整轮次、训练是否有成功客户端、同一目标精度的时间/通信量、多个种子及波动。
