#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) utilities for parameter-efficient fine-tuning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    r: int = 8  # Rank
    lora_alpha: int = 16  # LoRA scaling parameter
    target_modules: List[str] = None  # Modules to apply LoRA to
    lora_dropout: float = 0.1  # Dropout probability
    bias: str = "none"  # Bias configuration: "none", "all", or "lora_only"
    task_type: str = "CAUSAL_LM"  # Task type
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRALayer(nn.Module):
    """LoRA layer implementation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.in_features = in_features
        self.out_features = out_features
        
        # Create LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        # Initialize A with random values
        nn.init.kaiming_uniform_(self.lora_A, a=torch.sqrt(torch.tensor(5.0)))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer"""
        # Apply dropout to input
        x_dropout = self.lora_dropout(x)
        
        # Compute LoRA output: (x @ A^T @ B^T) * scaling
        lora_output = x_dropout @ self.lora_A.T @ self.lora_B.T
        
        return lora_output * self.scaling


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original layer and LoRA"""
        # Original layer output
        original_output = self.original_layer(x)
        
        # Add LoRA adaptation
        lora_output = self.lora(x)
        
        return original_output + lora_output


def apply_lora_to_model(
    model: nn.Module,
    lora_config: LoRAConfig,
    verbose: bool = True
) -> nn.Module:
    """Apply LoRA to specified modules in the model"""
    
    lora_modules = []
    total_params = 0
    trainable_params = 0
    
    # Find and replace target modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should have LoRA
            should_apply_lora = any(
                target in name for target in lora_config.target_modules
            )
            
            if should_apply_lora:
                # Get parent module and attribute name
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                # Replace with LoRA layer
                lora_layer = LinearWithLoRA(
                    original_layer=module,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                )
                
                setattr(parent, child_name, lora_layer)
                lora_modules.append(name)
                
                # Count parameters
                for param in module.parameters():
                    total_params += param.numel()
                for param in lora_layer.lora.parameters():
                    trainable_params += param.numel()
    
    # Handle bias
    if lora_config.bias != "none":
        for name, param in model.named_parameters():
            if "bias" in name:
                if lora_config.bias == "all":
                    param.requires_grad = True
                    trainable_params += param.numel()
                elif lora_config.bias == "lora_only":
                    # Only unfreeze bias in LoRA modules
                    if any(lora_name in name for lora_name in lora_modules):
                        param.requires_grad = True
                        trainable_params += param.numel()
    
    # Count total parameters
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    
    if verbose:
        logger.info(f"🔧 Applied LoRA to {len(lora_modules)} modules")
        logger.info(f"📊 LoRA configuration:")
        logger.info(f"   Rank (r): {lora_config.r}")
        logger.info(f"   Alpha: {lora_config.lora_alpha}")
        logger.info(f"   Target modules: {lora_config.target_modules}")
        logger.info(f"   Dropout: {lora_config.lora_dropout}")
        logger.info(f"   Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights back into the original model"""
    
    merged_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # Merge weights
            with torch.no_grad():
                # Compute merged weight: W + BA * scaling
                lora_weight = module.lora.lora_B @ module.lora.lora_A
                merged_weight = module.original_layer.weight + lora_weight * module.lora.scaling
                
                # Create new linear layer with merged weights
                merged_layer = nn.Linear(
                    module.original_layer.in_features,
                    module.original_layer.out_features,
                    bias=module.original_layer.bias is not None
                )
                
                merged_layer.weight.data = merged_weight
                if module.original_layer.bias is not None:
                    merged_layer.bias.data = module.original_layer.bias.data
            
            # Replace module
            setattr(parent, child_name, merged_layer)
            merged_count += 1
    
    logger.info(f"✅ Merged {merged_count} LoRA modules back into the model")
    return model


def save_lora_weights(
    model: nn.Module,
    save_path: str,
    config: Optional[LoRAConfig] = None
):
    """Save only the LoRA weights"""
    
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B
    
    # Save weights and config
    save_dict = {
        "lora_weights": lora_state_dict,
        "config": config.__dict__ if config else None
    }
    
    torch.save(save_dict, save_path)
    logger.info(f"💾 Saved LoRA weights to {save_path}")


def load_lora_weights(
    model: nn.Module,
    load_path: str,
    strict: bool = True
) -> nn.Module:
    """Load LoRA weights into a model"""
    
    checkpoint = torch.load(load_path, map_location="cpu")
    lora_weights = checkpoint["lora_weights"]
    
    # Load weights
    loaded_count = 0
    for name, param in lora_weights.items():
        module_name = name.rsplit('.', 2)[0]
        param_name = name.split('.')[-1]
        
        # Find module
        try:
            module = model
            for part in module_name.split('.'):
                module = getattr(module, part)
            
            if isinstance(module, LinearWithLoRA):
                if param_name == "lora_A":
                    module.lora.lora_A.data = param
                elif param_name == "lora_B":
                    module.lora.lora_B.data = param
                loaded_count += 1
        except AttributeError:
            if strict:
                raise RuntimeError(f"Module {module_name} not found in model")
            else:
                logger.warning(f"Skipping {name} - module not found")
    
    logger.info(f"✅ Loaded {loaded_count} LoRA parameters from {load_path}")
    return model


def calculate_lora_params(
    model: nn.Module,
    lora_config: LoRAConfig
) -> Dict[str, int]:
    """Calculate the number of parameters that would be added by LoRA"""
    
    lora_params = 0
    affected_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_apply_lora = any(
                target in name for target in lora_config.target_modules
            )
            
            if should_apply_lora:
                # Calculate LoRA parameters for this module
                in_features = module.in_features
                out_features = module.out_features
                
                # A: r x in_features, B: out_features x r
                module_lora_params = lora_config.r * (in_features + out_features)
                lora_params += module_lora_params
                
                affected_modules.append({
                    "name": name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "lora_params": module_lora_params
                })
    
    # Calculate total model parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "total_params": total_params,
        "lora_params": lora_params,
        "percentage": (lora_params / total_params) * 100,
        "affected_modules": affected_modules
    }


if __name__ == "__main__":
    # Example usage
    print("🔧 LoRA Utilities Example")
    
    # Create example config
    config = LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1
    )
    
    print(f"\n📊 LoRA Configuration:")
    print(f"   Rank: {config.r}")
    print(f"   Alpha: {config.lora_alpha}")
    print(f"   Target modules: {config.target_modules}")
    print(f"   Dropout: {config.lora_dropout}") 