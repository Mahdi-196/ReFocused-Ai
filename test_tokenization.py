#!/usr/bin/env python3
"""
Test script for data tokenization.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer

def test_tokenizer():
    """Test the tokenizer with sample data."""
    
    # Try to find tokenizer
    tokenizer_paths = [
        "models/tokenizer/tokenizer/tokenizer.json",
        "models/tokenizer/transformers_tokenizer/tokenizer.json",
        "tokenizer_750M/tokenizer.json"
    ]
    
    tokenizer_path = None
    for path in tokenizer_paths:
        if Path(path).exists():
            tokenizer_path = path
            break
    
    if not tokenizer_path:
        print("Error: Could not find tokenizer.json file")
        print("Looking for tokenizer files...")
        for pattern in ["**/tokenizer.json", "**/vocab.json"]:
            files = list(Path(".").glob(pattern))
            if files:
                print(f"Found: {files}")
        return False
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    # Test special tokens
    special_tokens = ["<|startoftext|>", "<|endoftext|>", "<|pad|>", "<|unk|>", "<|reddit|>", "<|hf|>"]
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"{token}: {token_id}")
    
    # Test encoding
    sample_text = "<|startoftext|><|reddit|>This is a test post from Reddit about machine learning.<|endoftext|>"
    encoding = tokenizer.encode(sample_text)
    print(f"\nSample text: {sample_text}")
    print(f"Encoded length: {len(encoding.ids)}")
    print(f"First 10 tokens: {encoding.ids[:10]}")
    print(f"Decoded: {tokenizer.decode(encoding.ids)}")
    
    return True

if __name__ == "__main__":
    success = test_tokenizer()
    if success:
        print("\n✅ Tokenizer test passed!")
    else:
        print("\n❌ Tokenizer test failed!") 