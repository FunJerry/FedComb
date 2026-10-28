"""共享的小型数据结构；索引均从 0 开始，单位均用 SI。"""
from dataclasses import dataclass


def component_id(kind: str, index: int, shards: int = 16) -> int:
    """B:0..15，H:16..31，T:32..47；公共 E/D 不纳入成熟度。"""
    if kind not in ('block', 'head', 'tail') or not 0 <= index < shards:
        raise ValueError('组件类型或索引越界')
    return {'block': 0, 'head': shards, 'tail': 2 * shards}[kind] + index


@dataclass(frozen=True)
class Resources:
    memory_bytes: float
    compute_flops_per_second: float
    bandwidth_bytes_per_second: float


@dataclass(frozen=True)
class Plan:
    """一次分配。student=None 表示无额外蒸馏，不计学生预算。"""
    start: int
    end: int
    student_kind: str | None = None
    student_index: int | None = None

    def __post_init__(self):
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end < 16:
            raise ValueError('主分片区间必须是 0～15 内的整数闭区间')
        if self.student_kind is None:
            if self.student_index is not None:
                raise ValueError('无学生时不能设置学生索引')
        elif (self.student_kind not in ('head', 'tail') or type(self.student_index) is not int
              or not self.start <= self.student_index <= self.end):
            raise ValueError('学生类型必须为 head/tail，位置必须在本次区间内')

    def task_keys(self, shards: int = 16) -> list[str]:
        """本地任务模型实际持有的组件名，不含额外学生。"""
        keys = ['stem']
        if self.start > 0:
            keys.append(f'head.{self.start - 1}')
        keys += [f'block.{i}' for i in range(self.start, self.end + 1)]
        if self.end < shards - 1:
            keys.append(f'tail.{self.end + 1}')
        return keys + ['classifier']

    def student_key(self) -> str | None:
        return None if self.student_kind is None else f'{self.student_kind}.{self.student_index}'

    def keys(self, shards: int = 16) -> list[str]:
        keys = self.task_keys(shards)
        return keys + ([self.student_key()] if self.student_key() else [])


@dataclass(frozen=True)
class Cost:
    weight_bytes: int
    forward_flops_per_sample: float
    comm_seconds: float  # 单向模型包，不含极小的控制信息。
    compute_seconds: float
    extra_upload_bytes: int = 0  # IPC 首批主梯度；普通模式为 0。
    extra_upload_seconds: float = 0.0

    @property
    def communication_bytes(self) -> int:
        return 2 * self.weight_bytes + self.extra_upload_bytes

    @property
    def communication_seconds(self) -> float:
        return 2 * self.comm_seconds + self.extra_upload_seconds

    @property
    def round_seconds(self) -> float:
        return self.communication_seconds + self.compute_seconds
