# FedComb 实验工程

CIFAR-10 三方法云端训练入口与结果查看：见 [云端测试指南](docs/CIFAR10云端测试指南.md)。

当前资源、评分和 Greedy 已按用户确认对齐原算法意图并修正缺陷，详见 [本次修改说明](docs/算法意图对齐修改说明.md)。新实验须使用新目录，不能续训旧结果。

这版工程围绕“怎样运行和扩展实验”组织，而不是围绕旧脚本中的变量逐个拆文件。

**阅读入口只有两条：**

- 看实验流程：`experiment.py` → `federation/round.py` → `federation/client.py` / `server.py`。
- 改实验条件：`configs/` → `experiments/` → `strategies/`。

所有变体共用同一个联邦循环。新增消融不复制训练脚本，不通过改全局变量影响下一组实验。原作者项目和上一版工程均保留不动。

## 1. 先看这条固定主线

```text
实验配置
  → 初始化数据、模型、资源
  → 每轮抽取客户端
  → 为客户端分配连续区间和额外学生
  → 客户端本地任务训练 / 学生蒸馏
  → 服务器按组件聚合
  → 更新成熟度和虚拟队列
  → 测试完整模型
  → 记录结果，下一轮
```

主线参考原作者实验顺序。模型底层沿用已经验证的 ResNet-50 / 辅助 Bottleneck 构造；训练细节不照搬旧代码缺陷。详细区别见 [实验口径与边界](docs/实验口径与边界.md)。

## 2. 目录按研究工作的职责划分

```text
Fedcomb工程/
├── configs/                      单次实验配置，按数据/模型/训练/策略分组
│   ├── paper.json                论文规模 CIFAR-10 默认配置
│   ├── smoke.json                不下载数据的快速机制检查
│   └── fixed_path.json           固定 [3,6] 路径、足够容量的蒸馏对照
├── experiments/                  变体 × 参数网格 × 种子的实验表
│   ├── ablations.json            主消融：蒸馏、调度、评分项、静态资源
│   ├── ipc_comparison.json       同一首批梯度口径的 IPC 对照
│   ├── fixed_path_distillation.json
│   ├── time_weight_sweep.json    V 参数讨论
│   └── smoke_matrix.json         小模型实验矩阵验证
├── src/fedcomb/
│   ├── cli.py                    单实验 / 批量实验入口
│   ├── config.py                 分组配置及非法组合检查
│   ├── experiment.py             准备一次实验、执行轮次、保存结果
│   ├── suite.py                  展开实验表、跨种子汇总
│   ├── federation/               固定实验流程，不放变化策略
│   │   ├── round.py              一轮联邦流程，按步骤 1～6 排列
│   │   ├── client.py             本地准备、一个 batch、一次客户端训练
│   │   └── server.py             全局聚合、进度更新、完整模型测试
│   ├── strategies/               主要实验扩展位置
│   │   ├── allocation.py         dynamic / random / greedy / random_feasible / full / fixed
│   │   ├── distillation.py       adaptive / head / tail / none 与损失
│   │   └── aggregation.py        mean / gradient_mean / ipc
│   ├── models/network.py         E/B/H/T/D、本地模型独立副本
│   ├── data/datasets.py          数据、无重复划分、可重放增强和批次
│   ├── simulation/               资源模拟与每个组合的成本统计
│   ├── tracking/                 成熟度、队列、结果与检查点
│   └── core/                     共享类型与种子工具
├── tests/                        关键机制测试，无真实数据下载
├── docs/                         主线阅读、扩展实验、口径、验证记录
├── pyproject.toml                安装后提供 fedcomb 命令
├── requirements-lock.txt         本机验证环境依赖版本
├── results/                      各实验独立结果，Git 忽略
├── data/                         下载数据，Git 忽略
└── .git/                         本地版本历史；不连接远程仓库
```

## 3. 在本机怎么运行

在 `/Users/zwp0728/Project/Fedcomb工程` 中运行。独立环境 `.venv` 与代码、数据、结果均在同一个工程下；移动工程后需重新执行 `.venv/bin/python -m pip install -e . --no-deps --no-build-isolation` 修复安装路径。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/fedcomb run configs/smoke.json --output my_smoke
```

先展开实验表，确认到底改了哪些条件：

```bash
.venv/bin/fedcomb suite experiments/ablations.json --list
```

再跑快速矩阵：

```bash
.venv/bin/fedcomb suite experiments/smoke_matrix.json --output my_smoke_matrix
```

真实数据默认不自动下载。在配置中明确设置 `data.download=true` 后，数据存入工程 `data/`。

```bash
.venv/bin/fedcomb run configs/paper.json --output paper_trial
```

不要第一次就运行 1000 轮：先在新的配置文件把 `federation.rounds` 改小，核对设备、容量和日志。输出目录非空时拒绝覆盖。

### 迁移到另一台机器

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

Python 最低 3.10，本机验证使用 3.13。无需依赖上一个工程的环境或路径。

## 4. 做消融时通常只改 JSON

| 研究问题 | 配置变化 | 不会发生的隐式变化 |
|---|---|---|
| 无额外蒸馏 | `distillation.mode: none` | 不删除任务路径边界 H/T；额外学生预算确实移除 |
| 仅头 / 仅尾 | `distillation.mode: head / tail` | 不复制客户端训练代码 |
| 固定 λ | `weight_mode: fixed`、`fixed_weight: 1` | 不再依赖成熟度差距 |
| 无成熟度评分项 | `allocation.maturity_weight: 0` | 成熟度仍记录，学生选择/λ 仍独立存在 |
| 无覆盖评分项 | `allocation.coverage_weight: 0` | 覆盖诊断日志仍保留 |
| 静态资源 | `environment.change_probability: 0` | 不改变数据划分或参与者种子 |
| 参数讨论 | `grid` 中加入字段及取值 | 不手动生成多份训练脚本 |

特别注意：“去掉成熟度评分项”不等于“去掉所有成熟度机制”。消融命名要与实际覆盖字段一致。

固定路径蒸馏对照用足够容量排除资源拒绝干扰；动态无蒸馏对照允许重新分配，观察的是端到端收益。两者回答不同问题。

## 5. 每个实验保存什么

`results/<实验表>/<变体>/seed-42/` 中保存：

- `config.json`：完整展开的配置，没有隐藏默认值。
- `manifest.json`：依赖、源码哈希、初始模型哈希、数据划分哈希、初始资源。
- `partition.json`、`label_counts.json`：具体划分和类别数量。
- `component_costs.json`：真实组件参数字节和近似 FLOPs。
- `rounds.jsonl`：逐轮、逐客户端计划，分项评分、损失、λ、队列、预算、报告与拒绝原因。
- `metrics.csv`：轮次精度、成功/跳过数、估算时间、通信量、实测时间。
- `checkpoint_*.pt`：全局权重和学习状态。
- `summary.json`：最后精度和最佳精度分开保存。

实验表根目录保存 `resolved_cases.json` 和 `suite_summary.json`，按同一变体跨种子汇总均值与样本标准差；单种子标准差为 null，不伪造误差棒。

## 6. 建议阅读顺序

1. [主线阅读指南](docs/主线阅读指南.md)：只沿训练流程看代码。
2. [如何扩展实验](docs/如何扩展实验.md)：添加消融、网格或算法策略。
3. [实验口径与边界](docs/实验口径与边界.md)：避免不公平对照及过度复现声明。
4. [验证记录](docs/验证记录.md)：哪些真正执行过，哪些尚未验证。

当前仍是单机模拟，不是跨设备网络系统；合成数据精度没有研究意义。真实数据集论文规模实验、CUDA/MPS、真实峰值内存和分布式基线尚未验证。
