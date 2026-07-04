import torch
import torch.nn.functional as F


def get_unlearning_loss(student_pred, teacher_pred):
    """Match student forget-prompt scores to M_{t-1} neutral-target scores."""
    return F.mse_loss(student_pred.float(), teacher_pred.detach().float(), reduction="mean")


def get_preservation_loss(student_pred, teacher_pred):
    """Distill M_{t-1} on local retain prompts to preserve nearby concepts."""
    return F.mse_loss(student_pred.float(), teacher_pred.detach().float(), reduction="mean")


def get_regularization_loss(student_model, teacher_model):
    """Parameter-space L2 regularization against previous checkpoint M_{t-1}."""
    l2_loss = 0.0
    student_params = dict(student_model.named_parameters())
    teacher_params = dict(teacher_model.named_parameters())
    for name, param in student_params.items():
        if param.requires_grad:
            teacher_param = teacher_params[name].detach()
            if teacher_param.device != param.device:
                teacher_param = teacher_param.to(device=param.device)
            param_for_loss = param
            if teacher_param.dtype != param.dtype:
                param_for_loss = param.to(dtype=teacher_param.dtype)
            l2_loss += F.mse_loss(param_for_loss, teacher_param, reduction="sum")
    return l2_loss
