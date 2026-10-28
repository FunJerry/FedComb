"""按组件合并；动态状态平均与随机 IPC 是两条明确的更新路径。"""
import torch


def average_state(module, entries, old_weight):
    """仅对持有该组件的客户端按样本量归一化；未收到组件保留旧状态。

    浮点缓冲区也平均；BN 整数计数取最大，不进行没有语义的浮点插值。
    """
    if not entries:
        return
    total = sum(samples for _, samples in entries)
    state = module.state_dict()
    for name, previous in state.items():
        tensors = [(update[name], samples / total) for update, samples in entries]
        if previous.is_floating_point():
            averaged = sum(tensor * weight for tensor, weight in tensors)
            previous.copy_(old_weight * previous + (1 - old_weight) * averaged)
        else:
            previous.copy_(torch.stack([tensor for tensor, _ in tensors]).amax(dim=0))


def aggregate_ipc_shard(module, key, shard_index, results, learning_rate, old_weight, inverse_probability=True):
    """Eq.28–30：参数 W ← W − μ Σ_n q_n I_n,c/p_n,c g_n,c。

    绝不把 q/p 再归一化为和 1，否则会破坏逆概率校正。
    主分片 BN 缓冲区没有参数梯度，单独走状态平均；不是论文的无偏梯度结论。
    """
    total_samples = sum(result.samples for result in results)
    included = [result for result in results if key in result.first_gradients]
    buffers = {name: value.clone() for name, value in module.named_buffers()}
    for name, parameter in module.named_parameters():
        update = torch.zeros_like(parameter)
        for result in included:
            probability = result.probabilities[shard_index] if inverse_probability else 1.0
            if probability <= 0:
                raise ValueError('收到更新却记录了零入选概率')
            gradient = result.first_gradients[key].get(name)
            if gradient is not None:
                update.add_(gradient, alpha=result.samples / total_samples / probability)
        parameter.add_(update, alpha=-learning_rate)
    # 只汇总 buffers，不用模型平均覆盖上面的 IPC 参数更新。
    entries = [(result.states[key], result.samples) for result in included]
    if entries:
        total = sum(samples for _, samples in entries)
        for name, value in module.named_buffers():
            tensors = [entry[name] for entry, _ in entries]
            if value.is_floating_point():
                averaged = sum(entry[name] * samples / total for entry, samples in entries)
                value.copy_(old_weight * buffers[name] + (1 - old_weight) * averaged)
            else:
                value.copy_(torch.stack(tensors).amax(dim=0))


@torch.no_grad()
def aggregate(bank, results, config):
    """模型库位于 CPU，客户端回传 CPU 状态；空轮不产生任何更新。"""
    if not results:
        return
    for key, module in bank.components().items():
        if config.method in ('ipc', 'gradient_mean') and key.startswith('block.'):
            aggregate_ipc_shard(module, key, int(key.split('.')[1]), results,
                                 config.ipc_learning_rate, config.old_weight,
                                 inverse_probability=config.method == 'ipc')
        else:
            entries = [(result.states[key], result.samples) for result in results if key in result.states]
            average_state(module, entries, config.old_weight)
