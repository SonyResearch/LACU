import torch
import torch.nn.functional as F
def get_unlearning_loss(
    student_pred,
    teacher_pred,
    loss_type="l2"
):
    """
    Calculates the unlearning loss (Inverse Generative Distillation).
    This loss pushes the student model's prediction for a "forget" prompt
    to match a teacher model's prediction for a neutral prompt.
    Args:
        student_pred (torch.Tensor): Noise prediction from the student model for the forget prompt.
        teacher_pred (torch.Tensor): Noise prediction from the teacher model for the neutral prompt.
        loss_type (str): The type of loss to use ('l2').
    Returns:
        torch.Tensor: The calculated unlearning loss.
    """
    teacher_pred = teacher_pred.detach()  # Don't track gradients for the teacher
    if loss_type == "l2":
        return F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")
    else:
        raise ValueError(f"Unknown unlearning loss type: {loss_type}")

def get_preservation_loss(
    student_pred,
    teacher_pred,
    original_noise,
    mode="distillation"
):
    """
    Calculates the preservation loss using the Golden Teacher.
    Args:
        student_pred (torch.Tensor): Noise prediction from the student model for a retain prompt.
        teacher_pred (torch.Tensor): Noise prediction from the Golden Teacher for the same retain prompt.
        original_noise (torch.Tensor): The ground truth noise added to the latents.
        mode (str): The preservation strategy ('distillation' or 'replay').
    Returns:
        torch.Tensor: The calculated preservation loss.
    """
    if mode == "distillation":
        # Generative Distillation: Match the Golden Teacher's entire denoising process.
        teacher_pred = teacher_pred.detach()
        return F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")
    elif mode == "replay":
        # Standard Generative Replay: Train the student on the teacher's generated output.
        return F.mse_loss(student_pred.float(), original_noise.float(), reduction="mean")
    else:
        raise ValueError(f"Unknown preservation mode: {mode}")

def get_regularization_loss(student_model, teacher_model):
    """
    Calculates a parameter-space regularization loss.
    Args:
        student_model (torch.nn.Module): The student model (being trained).
        teacher_model (torch.nn.Module): The teacher model from the previous step (frozen).
    Returns:
        torch.Tensor: The calculated regularization loss.
    """
    l2_loss = 0.0
    student_params = dict(student_model.named_parameters())
    teacher_params = dict(teacher_model.named_parameters())
    for name, param in student_params.items():
        if param.requires_grad:
            l2_loss += F.mse_loss(param, teacher_params[name].detach(), reduction="sum")
    return l2_loss