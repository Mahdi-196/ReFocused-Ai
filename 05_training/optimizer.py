"""
Optimizer and learning rate scheduler configurations
"""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    training_config
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and scheduler with proper weight decay settings"""
    
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ["bias", "norm", "LayerNorm", "layernorm", "ln"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": training_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    # Log parameter counts
    num_decay_params = sum(p.numel() for p in decay_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {len(decay_params)} weight decay params ({num_decay_params:,} total)")
    print(f"Optimizer: {len(no_decay_params)} no weight decay params ({num_no_decay_params:,} total)")
    
    # Create optimizer
    if training_config.optimizer == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_eps,
            fused=True  # Use fused AdamW if available (faster)
        )
    else:
        raise ValueError(f"Unknown optimizer: {training_config.optimizer}")
    
    # Create scheduler
    if training_config.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_config.warmup_steps,
            num_training_steps=training_config.max_steps,
            min_lr_ratio=training_config.min_lr_ratio
        )
    elif training_config.lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_config.warmup_steps,
            num_training_steps=training_config.max_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {training_config.lr_scheduler}")
    
    return optimizer, scheduler


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between initial lr and min_lr_ratio * initial lr
    """
    
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """
    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


# SAM (Sharpness-Aware Minimization) optimizer wrapper for improved generalization
class SAM(torch.optim.Optimizer):
    """
    SAM optimizer that helps find flatter minima for better generalization
    Based on: https://arxiv.org/abs/2010.01412
    """
    
    def __init__(self, base_optimizer, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and then `second_step`.")
    
    def _grad_norm(self):
        # avoid memory leaks
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def create_sam_optimizer(model, training_config, use_sam=True, rho=0.05):
    """Create SAM-wrapped optimizer if requested"""
    base_optimizer, scheduler = create_optimizer_and_scheduler(model, training_config)
    
    if use_sam:
        optimizer = SAM(base_optimizer, rho=rho, adaptive=True)
    else:
        optimizer = base_optimizer
    
    return optimizer, scheduler 