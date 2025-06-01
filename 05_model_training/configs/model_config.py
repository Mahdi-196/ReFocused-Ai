"""
Model configuration for ReFocused-AI 1.2B parameter model
"""

from transformers import GPTNeoXConfig

def get_model_config():
    """
    Returns configuration for a 1.2B parameter model optimized for H100
    """
    config = GPTNeoXConfig(
        # Model architecture - targeting ~1.2B parameters
        hidden_size=2048,           # Hidden dimension
        num_hidden_layers=24,       # Number of transformer blocks
        num_attention_heads=16,     # Number of attention heads
        intermediate_size=8192,     # FFN intermediate size (4x hidden)
        
        # Vocab and sequence settings
        vocab_size=50257,           # Standard GPT-2 vocab size
        max_position_embeddings=2048,  # Max sequence length
        
        # Attention configuration
        rotary_pct=0.25,           # Rotary position embeddings
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        
        # Activation and normalization
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        use_parallel_residual=True,  # Parallel attention and FFN
        
        # Initialization
        initializer_range=0.02,
        
        # Training stability
        use_cache=False,  # Disable KV cache during training
        tie_word_embeddings=False,  # Don't tie embeddings for better capacity
    )
    
    # Calculate approximate parameter count
    param_count = calculate_params(config)
    print(f"Model configuration created with ~{param_count/1e9:.2f}B parameters")
    
    return config


def calculate_params(config):
    """Calculate approximate parameter count"""
    params = 0
    
    # Embeddings
    params += config.vocab_size * config.hidden_size  # Token embeddings
    params += config.max_position_embeddings * config.hidden_size  # Position embeddings
    
    # Transformer layers
    per_layer = 0
    # Attention
    per_layer += 4 * config.hidden_size * config.hidden_size  # Q, K, V, O projections
    # FFN
    per_layer += 2 * config.hidden_size * config.intermediate_size  # Up and down projections
    # Layer norms
    per_layer += 4 * config.hidden_size  # 2 layer norms per block
    
    params += per_layer * config.num_hidden_layers
    
    # Output layer
    params += config.vocab_size * config.hidden_size
    
    return params 