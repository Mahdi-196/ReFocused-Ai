"""
Test script to verify FSDP configuration
"""

import torch
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

def main():
    print("Testing FSDP configuration...")
    
    # Create small model config for testing
    model_config = GPTNeoXConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        vocab_size=50257,
        max_position_embeddings=512,
    )
    
    # Create auto wrap policy for GPTNeoXLayer
    auto_wrap_policy = lambda module, *args, **kwargs: transformer_auto_wrap_policy(
        module,
        {GPTNeoXLayer},
        *args,
        **kwargs
    )
    
    # Initialize FSDP plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",
        cpu_offload=False,
        backward_prefetch="BACKWARD_PRE",
        forward_prefetch=True,
        use_orig_params=True,
        sync_module_states=True,
        activation_checkpointing=True,
        auto_wrap_policy=auto_wrap_policy,
        state_dict_type="FULL_STATE_DICT",
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
    
    # Initialize accelerator with FSDP
    accelerator = Accelerator(
        mixed_precision="bf16",
        fsdp_plugin=fsdp_plugin,
    )
    
    # Print accelerator info
    print(f"Accelerator state: {accelerator.state}")
    print(f"Device: {accelerator.device}")
    print(f"Distributed type: {accelerator.distributed_type}")
    
    # Initialize model
    print("Initializing test model...")
    model = GPTNeoXForCausalLM(model_config)
    
    # Prepare model with FSDP
    model = accelerator.prepare(model)
    
    print(f"Model successfully prepared with FSDP")
    print(f"Model type: {type(model)}")
    
    print("FSDP configuration test successful!")


if __name__ == "__main__":
    main() 