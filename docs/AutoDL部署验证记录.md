# AutoDL 部署验证记录

验证日期：2026-10-14。目标为用户确认的实例 `790b4aac6e-fc16455a`。

## 保存位置

- 本地工程：`/Users/zwp0728/Documents/Fedcomb工程`
- 云端工程：`/root/autodl-tmp/fedcomb_cifar10`
- 云端短程验证结果：`results/cloud_gpu_verified`
- 本地上传包：`deploy/fedcomb_cifar10_bundle.tar.gz`
- 上传包 SHA256：`8cbaaf1ab49da3d200c3703fe5e008ffd0cdc5e6395faf8f26325daba8ec5ff5`

上传包在服务器校验一致；逐文件校验通过 60057 个文件，其中 CIFAR-10 PNG 60000 张。训练集 50000，测试集 10000。

## 实际服务器环境

Python 3.12.3，PyTorch 2.12.1+cu130，torchvision 0.27.1+cu130，CUDA 13.0。GPU 报告为 NVIDIA GeForce RTX 4090，可用总显存约 47.37 GiB。没有替换服务器 CUDA PyTorch。

服务器最初缺少 SciPy，预设阿里云镜像未能找到版本。最终使用以下命令成功安装 SciPy 1.18.1；matplotlib 3.11.0 已存在：

```bash
.venv/bin/python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'scipy>=1.11,<2'
```

依赖只装到工程虚拟环境，没有修改系统驱动或服务器付费配置。

## 已验证的内容

本地和云端的 15 项自动测试通过。包括候选区间、真实反向传播、蒸馏梯度隔离、数据加载、三方法配置一致性、检查点恢复重放。

云端执行 `--method all --profile smoke --output cloud_gpu_verified --background`。三个方法均完成 2/2 轮，真实 CIFAR-10、width=64 ResNet-50、CUDA 训练，完整 10000 张测试集评估。

| 方法 | 成功客户端更新数 | 任务优化步数 | 蒸馏优化步数 | 客户端分配器峰值显存 MiB |
|---|---:|---:|---:|---:|
| FedComb | 4 | 4 | 2 | 42.3 |
| FedComb-Random | 4 | 4 | 2 | 251.3 |
| FedComb-Greedy | 3 | 3 | 1 | 383.6 |

Greedy 的 4 次抽取中有一次没有成功训练；资源预算接纳机制允许跳过客户端，不能把“抽中”计作“成功更新”。峰值统计仅为客户端阶段 CUDA 分配器指标，不是整机显存占用。

结果汇总脚本、`comparison.png` 图表已验证可生成。这里的少轮精度不作为方法优劣或论文精度结果。

## 尚未执行

没有启动各 1000 轮的正式实验，也没有完成多个随机种子的对比。推荐先运行 `quick`（论文规模配置，仅 3 轮），确认后启动 `paper`。完整命令见 [CIFAR10 云端测试指南](CIFAR10云端测试指南.md)。

一份模型库快照约 261.1 MiB。三种方法各 20 个周期检查点约 15.3 GiB，另外还有最佳权重、短程验证结果和数据；正式测试前建议保留至少 25 GiB 空间。脚本不会自动删除检查点或关停实例。

实际部署截图保存在本地 `deploy/cloud_tests.png` 和 `deploy/cloud_training_verified.png`。
