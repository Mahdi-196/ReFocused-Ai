#!/usr/bin/env python3
"""
Verify tokenized output format and content.
"""

import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

def verify_output():
    """Verify the tokenized output file."""
    
    # Load test output
    test_files = list(Path("data_tokenized_test").glob("*.npz"))
    if not test_files:
        print("❌ No test output files found")
        return False
    
    test_file = test_files[0]
    print(f"📋 Verifying: {test_file}")
    
    # Load data
    data = np.load(test_file)
    
    print(f"📊 Data shape: {data['input_ids'].shape}")
    print(f"📊 Files in archive: {list(data.files)}")
    
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    sequence_lengths = data['sequence_lengths']
    
    print(f"✅ Input IDs shape: {input_ids.shape}")
    print(f"✅ Attention mask shape: {attention_mask.shape}")
    print(f"✅ Sequence lengths shape: {sequence_lengths.shape}")
    
    # Check a sample sequence
    sample_seq = input_ids[0]
    sample_mask = attention_mask[0]
    sample_length = sequence_lengths[0]
    
    print(f"\n🔍 Sample sequence:")
    print(f"  Length: {sample_length}")
    print(f"  First 20 tokens: {sample_seq[:20]}")
    print(f"  Last 20 tokens: {sample_seq[-20:]}")
    print(f"  Attention mask (first 20): {sample_mask[:20]}")
    print(f"  Attention mask (last 20): {sample_mask[-20:]}")
    
    # Check for special tokens
    unique_tokens = np.unique(sample_seq[:sample_length])
    print(f"\n🎯 Special tokens found in sample:")
    special_token_map = {0: "START", 1: "END", 2: "PAD", 3: "UNK", 5: "REDDIT", 6: "HF"}
    for token_id in unique_tokens[:10]:  # Show first 10
        if token_id in special_token_map:
            print(f"  {token_id}: {special_token_map[token_id]}")
    
    # Load tokenizer and decode a sample
    try:
        tokenizer = Tokenizer.from_file("models/tokenizer/transformers_tokenizer/tokenizer.json")
        
        # Decode the first 100 tokens
        sample_tokens = sample_seq[:min(100, sample_length)]
        decoded_text = tokenizer.decode(sample_tokens)
        print(f"\n📝 Decoded sample (first 100 tokens):")
        print(f"  {decoded_text}")
        
    except Exception as e:
        print(f"⚠️  Could not decode sample: {e}")
    
    return True

if __name__ == "__main__":
    verify_output() 