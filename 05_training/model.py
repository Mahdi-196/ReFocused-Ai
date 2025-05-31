"""
Efficient 1B parameter GPT model with HybridNorm and modern optimizations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from typing import Optional, Tuple
from flash_attn import flash_attn_func
from model_config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Precompute sin/cos for efficiency
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with grouped-query optimization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        
        # Grouped query attention for efficiency
        self.n_rep = self.n_head // self.n_kv_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # QKV normalization for HybridNorm
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.use_flash = config.use_flash_attention
        
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply QK normalization (part of HybridNorm)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(x, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat KV heads for grouped query attention
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        if self.use_flash and flash_attn_func is not None:
            # Use Flash Attention for efficiency
            output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            # Standard attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            output = att @ v
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        
        return output


class SwiGLU(nn.Module):
    """SwiGLU activation function for improved efficiency"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Transformer block with HybridNorm"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        
        # HybridNorm: No norm before attention (QKV norm inside attention)
        # Post-norm for FFN
        self.post_attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.post_ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None):
        # Attention with residual (no pre-norm due to HybridNorm)
        h = x + self.dropout(self.attention(x, attention_mask))
        h = self.post_attention_norm(h)
        
        # FFN with residual and post-norm
        out = h + self.dropout(self.feed_forward(h))
        out = self.post_ffn_norm(out)
        
        return out


class GPTModel(nn.Module):
    """Efficient 1B parameter GPT model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.init_std/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.size()
        
        # Token embeddings
        h = self.token_embedding(input_ids)
        
        # Create causal mask if needed
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).view(1, 1, T, T)
            attention_mask = attention_mask.repeat(B, 1, 1, 1)
        
        # Forward through transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, attention_mask)
            else:
                h = block(h, attention_mask)
        
        # Final layer norm
        h = self.ln_f(h)
        
        # Get logits
        logits = self.lm_head(h)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"loss": loss, "logits": logits}
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """Simple generation method"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids)
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == 2:  # EOS token
                    break
        
        return input_ids 