"""只模拟资源，绝不声称限制了真实 CPU、内存或网络。"""
from dataclasses import dataclass, astuple
from ..core.types import Resources


@dataclass
class ResourceState:
    actual: Resources  # 客户端当前值，服务器不直接读取。
    reported: Resources  # 服务器最后收到的值。


def limits(config):
    """配置 MB/GFLOP/s/Mb/s → 字节、FLOP/s、字节/s。"""
    return [(config.memory_min_mb * 1e6, config.memory_max_mb * 1e6),
            (config.compute_min_gflops * 1e9, config.compute_max_gflops * 1e9),
            (config.bandwidth_min_mbps * 1e6 / 8, config.bandwidth_max_mbps * 1e6 / 8)]


def create_resources(config, clients, rng):
    return [ResourceState(resource, resource) for resource in
            (Resources(*(rng.uniform(lo, hi) for lo, hi in limits(config)))
             for _ in range(clients))]


def fluctuate(state, config, rng):
    """原作者变化分布：20% 联合变化，内存/算力/带宽默认 ±20/30/25%。

    仅对本轮选中设备调用，在分配前执行；不照搬原来分配后容量变化却继续训练。
    初始上下界只用于初始化，不裁剪累积变化；所以后续资源可超出初始范围。
    """
    if config.change_probability == 0 or rng.random() >= config.change_probability:
        return False
    amplitudes = (config.memory_fluctuation, config.compute_fluctuation,
                  config.bandwidth_fluctuation)
    state.actual = Resources(*(value * rng.uniform(1 - amplitude, 1 + amplitude)
                              for value, amplitude in zip(astuple(state.actual), amplitudes)))
    return True


def report_if_triggered(state, threshold):
    """VI-A：相对最后一次上报的任意一维变化超过阈值，更新全部资源。"""
    changed = any(abs(current - last) / last >= threshold
                  for current, last in zip(astuple(state.actual), astuple(state.reported)))
    if changed:
        state.reported = state.actual
    return changed
