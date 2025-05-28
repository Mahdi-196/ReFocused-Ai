#!/usr/bin/env python3
"""
Tokenizer Training Setup and Test Script
=======================================

Validates environment setup and provides quick tokenizer training examples.
Perfect for getting started with tokenizer training.

Requirements:
- tokenizers
- transformers  
- tqdm

Author: AI Assistant
Version: 1.0
"""

import sys
import os
from pathlib import Path
from typing import List, Dict

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'tokenizers': 'tokenizers',
        'transformers': 'transformers', 
        'tqdm': 'tqdm',
        'torch': 'torch',
        'numpy': 'numpy',
        'json': 'json (built-in)'
    }
    
    missing_packages = []
    
    for package, display_name in required_packages.items():
        try:
            if package == 'json':
                import json
            elif package == 'tokenizers':
                from tokenizers import ByteLevelBPETokenizer
            elif package == 'transformers':
                from transformers import GPT2TokenizerFast
            elif package == 'tqdm':
                import tqdm
            elif package == 'torch':
                import torch
            elif package == 'numpy':
                import numpy
            
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name}")
            if package != 'json':
                missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_data_availability() -> Dict[str, bool]:
    """Check if training data is available."""
    print("\n📁 Checking data availability...")
    
    data_paths = {
        'data/cleaned/': 'Cleaned data directory',
        'data/processed/': 'Processed data directory', 
        'data/unified_raw/': 'Unified raw data directory'
    }
    
    availability = {}
    
    for path, description in data_paths.items():
        path_obj = Path(path)
        exists = path_obj.exists()
        availability[path] = exists
        
        if exists:
            files = list(path_obj.glob("*.jsonl"))
            file_count = len(files)
            total_size = sum(f.stat().st_size for f in files) / (1024*1024)  # MB
            print(f"✓ {description}: {file_count} files, {total_size:.1f}MB")
        else:
            print(f"✗ {description}: Not found")
    
    return availability

def create_sample_data() -> Path:
    """Create sample training data if none exists."""
    print("\n📝 Creating sample training data...")
    
    sample_texts = [
        "I will build better habits every day.",
        "The quick brown fox jumps over the lazy dog.",
        "Tokenization is fundamental to natural language processing and machine learning.",
        "Artificial intelligence and machine learning are transforming technology.",
        "Python programming language is widely used for data science and AI development.",
        "Deep learning models require large amounts of training data to perform well.",
        "Natural language understanding involves complex linguistic processing.",
        "Transformer architectures have revolutionized the field of natural language processing.",
        "Text preprocessing is an essential step in building language models.",
        "Tokenizers convert raw text into numerical representations for neural networks.",
        "Byte-pair encoding is a popular subword tokenization algorithm.",
        "WordPiece tokenization is used in BERT and similar transformer models.",
        "Unigram language models provide probabilistic approaches to tokenization.",
        "Custom tokenizers can be optimized for specific domains and languages.",
        "The vocabulary size affects both model performance and computational requirements.",
        "Special tokens mark boundaries and provide structural information to models.",
        "Character-level tokenization handles out-of-vocabulary words naturally.",
        "Subword tokenization balances vocabulary size with linguistic meaning.",
        "Training data quality significantly impacts tokenizer effectiveness.",
        "Compression ratio measures how efficiently tokenizers encode text.",
    ]
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample JSONL file
    sample_file = data_dir / "sample_training_data.jsonl"
    
    import json
    with open(sample_file, 'w', encoding='utf-8') as f:
        # Repeat texts to create more training data
        for _ in range(50):  # Create 1000 examples
            for text in sample_texts:
                record = {"text": text}
                f.write(json.dumps(record) + '\n')
    
    file_size = sample_file.stat().st_size / 1024  # KB
    print(f"✓ Created sample data: {sample_file} ({file_size:.1f}KB)")
    
    return sample_file

def test_basic_tokenizer_training(data_file: Path) -> bool:
    """Test basic tokenizer training functionality."""
    print(f"\n🧪 Testing basic tokenizer training with {data_file}...")
    
    try:
        from tokenizers import ByteLevelBPETokenizer
        from transformers import GPT2TokenizerFast
        import json
        
        # Extract text from JSONL
        texts = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    texts.append(data.get('text', ''))
        
        print(f"✓ Loaded {len(texts)} training texts")
        
        # Create temporary text file for training
        temp_file = Path("temp_test_training.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Initialize and train tokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        print("🔧 Training tokenizer...")
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        
        tokenizer.train(
            files=[str(temp_file)],
            vocab_size=2000,  # Small vocab for testing
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True
        )
        
        print("✓ Tokenizer training completed")
        
        # Test tokenizer
        test_text = "I will build better habits."
        encoding = tokenizer.encode(test_text)
        tokens = encoding.tokens
        token_ids = encoding.ids
        
        print(f"✓ Test tokenization:")
        print(f"  Text: '{test_text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        
        # Save tokenizer for testing
        output_dir = Path("test_tokenizer_output")
        output_dir.mkdir(exist_ok=True)
        tokenizer.save(str(output_dir / "tokenizer.json"))
        tokenizer.save_model(str(output_dir))
        
        print(f"✓ Tokenizer saved to {output_dir}")
        
        # Test loading with transformers
        fast_tokenizer = GPT2TokenizerFast.from_pretrained(str(output_dir))
        
        # Test round-trip
        encoded = fast_tokenizer.encode(test_text)
        decoded = fast_tokenizer.decode(encoded)
        
        print(f"✓ Round-trip test:")
        print(f"  Original: '{test_text}'")
        print(f"  Decoded: '{decoded}'")
        print(f"  Vocab size: {fast_tokenizer.vocab_size}")
        
        # Cleanup
        temp_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during tokenizer training test: {e}")
        return False

def run_performance_test():
    """Run performance test on different text lengths."""
    print("\n⚡ Running performance tests...")
    
    try:
        from tokenizers import ByteLevelBPETokenizer
        import time
        
        # Create a simple tokenizer for testing
        tokenizer = ByteLevelBPETokenizer()
        
        # Test texts of different lengths
        test_cases = {
            "Short": "Hello world!",
            "Medium": "The quick brown fox jumps over the lazy dog. " * 10,
            "Long": "Natural language processing is a fascinating field. " * 100
        }
        
        print("📊 Performance Results:")
        for name, text in test_cases.items():
            # Time tokenization
            start_time = time.time()
            for _ in range(100):  # Multiple runs for averaging
                # Basic encode without training (character level)
                result = list(text.encode('utf-8'))
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100 * 1000  # milliseconds
            
            print(f"  {name} ({len(text)} chars): {avg_time:.2f}ms avg")
        
        print("✓ Performance test completed")
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")

def show_quick_examples():
    """Show quick examples of different tokenizer usage."""
    print("\n📚 Quick Examples:")
    
    examples = [
        {
            "title": "🚀 Quick Test Training",
            "command": "python test_tokenizer_small.py",
            "description": "Train a small tokenizer for testing (8K vocab, 2 files)"
        },
        {
            "title": "🎯 Full Production Training", 
            "command": "python train_tokenizer.py",
            "description": "Train production tokenizer (50K vocab, all data)"
        },
        {
            "title": "🔍 Compare Tokenizers",
            "command": "python tokenizer_comparison.py", 
            "description": "Compare different tokenizer types and performance"
        },
        {
            "title": "📊 Custom Domain Training",
            "command": "python train_tokenizer.py --vocab-size 30000 --domain medical",
            "description": "Train domain-specific tokenizer with custom settings"
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}")
        print(f"  Command: {example['command']}")
        print(f"  Purpose: {example['description']}")

def display_next_steps(data_available: bool):
    """Display recommended next steps based on setup status."""
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    print("=" * 50)
    
    if data_available:
        steps = [
            "1. 🧪 Test basic setup: python test_tokenizer_small.py",
            "2. 🔍 Compare tokenizers: python tokenizer_comparison.py", 
            "3. 🚀 Train production tokenizer: python train_tokenizer.py",
            "4. 📊 Analyze results and iterate on configuration",
            "5. 🔗 Integrate with your model training pipeline"
        ]
    else:
        steps = [
            "1. 📁 Prepare training data (JSONL format required)",
            "2. 🧪 Test with sample data: python setup_and_test.py",
            "3. 📂 Run data collection: python ../01_data_collection/multi_source_collector_infinite.py",
            "4. 🧹 Run data processing: python ../02_data_processing/data_cleaner.py",
            "5. 🚀 Then train tokenizer: python train_tokenizer.py"
        ]
    
    for step in steps:
        print(f"  {step}")

def main():
    """Main setup and test function."""
    print("🚀 TOKENIZER TRAINING SETUP & TEST")
    print("=" * 50)
    print("Validating environment and testing tokenizer training functionality.")
    
    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Please install missing dependencies before proceeding.")
        sys.exit(1)
    
    # Check data availability
    data_status = check_data_availability()
    data_available = any(data_status.values())
    
    # Create sample data if none exists
    if not data_available:
        print("\n⚠️  No training data found. Creating sample data for testing...")
        sample_file = create_sample_data()
        test_basic_tokenizer_training(sample_file)
    else:
        # Find first available data file
        for path, available in data_status.items():
            if available:
                path_obj = Path(path)
                jsonl_files = list(path_obj.glob("*.jsonl"))
                if jsonl_files:
                    test_basic_tokenizer_training(jsonl_files[0])
                    break
    
    # Run performance test
    run_performance_test()
    
    # Show examples
    show_quick_examples()
    
    # Display next steps
    display_next_steps(data_available)
    
    print("\n" + "=" * 50)
    print("✅ SETUP AND TEST COMPLETE!")
    print("\n💡 TIP: Start with 'python test_tokenizer_small.py' for quick validation.")
    print("📖 See TOKENIZER_GUIDE.md for comprehensive documentation.")

if __name__ == "__main__":
    main() 