"""蒸馏实验唯一变化点：选谁、权重多少、怎么计算损失。

none 不带额外学生；head/tail 只在对应辅助库内选最低成熟项。
adaptive 在两库中选择。边界辅助始终属于任务模型，不受 none 开关删除。
"""
from torch.nn import functional as F


def select_student(start, end, maturity, config):
    if config.mode == 'none':
        return None, None
    if config.mode in ('head', 'tail'):
        offset = 16 if config.mode == 'head' else 32
        p = min(range(start, end + 1), key=lambda i: maturity[offset + i])
        return config.mode, p
    p = min(range(start, end + 1), key=lambda i: min(maturity[16 + i], maturity[32 + i]))
    kind = 'head' if maturity[16 + p] < maturity[32 + p] else 'tail'
    return kind, p


def student_weight(progress, plan, config):
    if plan.student_key() is None:
        return 0.0
    if config.weight_mode == 'fixed':
        return config.fixed_weight
    return progress.distillation_weight(plan)


def compute_student_loss(student, classifier, plan, stem_features, target, logits, config):
    """所有教师张量 detach；冻结 D 参数，但保留对学生特征的导数。"""
    if plan.student_kind == 'head':
        return F.mse_loss(student(stem_features.detach()), target.detach())
    scores = classifier(student(target.detach()))
    tau = config.temperature
    return F.kl_div(F.log_softmax(scores / tau, dim=1),
                    F.softmax(logits.detach() / tau, dim=1), reduction='batchmean') * tau**2
