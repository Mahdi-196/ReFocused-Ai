"""
Configuration package for ReFocused-AI training
"""

from .model_config import get_model_config
from .training_config import get_training_config, TrainingConfig

__all__ = [
    'get_model_config',
    'get_training_config', 
    'TrainingConfig',
] 