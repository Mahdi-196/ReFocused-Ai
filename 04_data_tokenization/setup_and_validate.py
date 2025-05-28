#!/usr/bin/env python3
"""
Data Tokenization Setup and Validation Script
==============================================

Validates environment setup and provides quick tokenization testing.
Perfect for getting started with data tokenization.

Requirements:
- tokenizers
- transformers
- numpy
- psutil

Author: AI Assistant  
Version: 1.0
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import tempfile

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'tokenizers': 'tokenizers',
        'transformers': 'transformers',
        'numpy': 'numpy', 
        'psutil': 'psutil',
        'json': 'json (built-in)',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    for package, display_name in required_packages.items():
        try:
            if package == 'json':
                import json
            elif package == 'tokenizers':
                from tokenizers import Tokenizer
            elif package == 'transformers':
                from transformers import GPT2TokenizerFast
            elif package == 'numpy':
                import numpy
            elif package == 'psutil':
                import psutil
            elif package == 'tqdm':
                import tqdm
            
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

def check_tokenizer_availability() -> Dict[str, Optional[Path]]:
    """Check for available tokenizers."""
    print("\n🔍 Checking tokenizer availability...")
    
    tokenizer_paths = {
        'production': 'models/tokenizer/transformers_tokenizer/tokenizer.json',
        'basic': 'models/tokenizer/tokenizer.json',
        'test': '../03_tokenizer_training/test_tokenizer_output/tokenizer.json',
        'legacy': 'tokenizer_750M/tokenizer.json'
    }
    
    available_tokenizers = {}
    
    for name, path in tokenizer_paths.items():
        path_obj = Path(path)
        if path_obj.exists():
            available_tokenizers[name] = path_obj
            print(f"✓ {name.title()} tokenizer: {path}")
        else:
            available_tokenizers[name] = None
            print(f"✗ {name.title()} tokenizer: {path}")
    
    if not any(available_tokenizers.values()):
        print("\n❌ No tokenizers found!")
        print("💡 Please train a tokenizer first:")
        print("   cd ../03_tokenizer_training")
        print("   python train_tokenizer.py")
        return available_tokenizers
    
    print(f"\n✅ Found {sum(1 for v in available_tokenizers.values() if v)} tokenizer(s)")
    return available_tokenizers

def check_data_availability() -> Dict[str, Dict]:
    """Check if training data is available."""
    print("\n📁 Checking data availability...")
    
    data_directories = {
        'cleaned': 'data/cleaned',
        'processed': 'data/processed', 
        'unified_raw': 'data/unified_raw'
    }
    
    availability = {}
    
    for name, path in data_directories.items():
        path_obj = Path(path)
        exists = path_obj.exists()
        
        data_info = {
            'exists': exists,
            'path': path_obj,
            'files': [],
            'total_size_mb': 0,
            'file_count': 0
        }
        
        if exists:
            jsonl_files = list(path_obj.glob("*.jsonl"))
            data_info['files'] = jsonl_files
            data_info['file_count'] = len(jsonl_files)
            
            if jsonl_files:
                total_size = sum(f.stat().st_size for f in jsonl_files)
                data_info['total_size_mb'] = total_size / (1024*1024)
                print(f"✓ {name.title()} data: {len(jsonl_files)} files, {data_info['total_size_mb']:.1f}MB")
            else:
                print(f"⚠️  {name.title()} directory exists but no .jsonl files found")
        else:
            print(f"✗ {name.title()} data: Directory not found")
        
        availability[name] = data_info
    
    return availability

def check_output_directories() -> Dict[str, Dict]:
    """Check output directory status."""
    print("\n📂 Checking output directories...")
    
    output_dirs = {
        'production': 'data_tokenized_production',
        'test': 'data_tokenized_test',
        'basic': 'data_tokenized'
    }
    
    status = {}
    
    for name, path in output_dirs.items():
        path_obj = Path(path)
        exists = path_obj.exists()
        
        dir_info = {
            'exists': exists,
            'path': path_obj,
            'file_count': 0,
            'total_size_mb': 0,
            'has_data': False
        }
        
        if exists:
            npz_files = list(path_obj.glob("*.npz"))
            dir_info['file_count'] = len(npz_files)
            dir_info['has_data'] = len(npz_files) > 0
            
            if npz_files:
                total_size = sum(f.stat().st_size for f in npz_files)
                dir_info['total_size_mb'] = total_size / (1024*1024)
                print(f"✓ {name.title()} output: {len(npz_files)} files, {dir_info['total_size_mb']:.1f}MB")
            else:
                print(f"📁 {name.title()} output: Directory exists, no files yet")
        else:
            print(f"📁 {name.title()} output: Will be created when needed")
        
        status[name] = dir_info
    
    return status

def create_sample_data() -> Path:
    """Create sample data for testing."""
    print("\n📝 Creating sample data for testing...")
    
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
    
    # Create sample directory
    sample_dir = Path("temp_sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    sample_file = sample_dir / "sample_tokenization_data.jsonl"
    
    import json
    with open(sample_file, 'w', encoding='utf-8') as f:
        # Create varied samples
        for i in range(200):  # Create 200 examples
            text_idx = i % len(sample_texts)
            text = sample_texts[text_idx]
            
            # Add variation
            if i % 5 == 0:
                text = f"Sample {i}: {text}"
            elif i % 5 == 1:
                text = f"{text} This is additional content for testing purposes."
            
            record = {
                "text": text,
                "title": f"Test Title {i}",
                "subreddit": "test_data",
                "id": f"test_{i}"
            }
            f.write(json.dumps(record) + '\n')
    
    file_size = sample_file.stat().st_size / 1024  # KB
    print(f"✓ Created sample data: {sample_file} ({file_size:.1f}KB)")
    
    return sample_file

def test_tokenization_functionality(tokenizer_path: Path, data_file: Path) -> bool:
    """Test basic tokenization functionality."""
    print(f"\n🧪 Testing tokenization functionality...")
    
    try:
        from tokenizers import Tokenizer
        import numpy as np
        
        # Load tokenizer
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        print(f"✓ Tokenizer loaded - vocab size: {tokenizer.get_vocab_size()}")
        
        # Test special tokens
        special_tokens = {
            "<|startoftext|>": tokenizer.token_to_id("<|startoftext|>"),
            "<|endoftext|>": tokenizer.token_to_id("<|endoftext|>"),
            "<|pad|>": tokenizer.token_to_id("<|pad|>")
        }
        
        print("🔍 Special token status:")
        for token, token_id in special_tokens.items():
            if token_id is not None:
                print(f"   ✓ {token}: ID {token_id}")
            else:
                print(f"   ⚠️  {token}: Not found (will use fallback)")
        
        # Load and process sample data
        print("📊 Processing sample data...")
        sample_texts = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 50:  # Process first 50 samples
                    break
                try:
                    data = json.loads(line)
                    # Combine title and text
                    text_parts = []
                    if data.get('title'):
                        text_parts.append(data['title'])
                    if data.get('text'):
                        text_parts.append(data['text'])
                    combined_text = '\n'.join(text_parts)
                    sample_texts.append(combined_text)
                except:
                    continue
        
        if not sample_texts:
            print("❌ No valid texts found in sample data")
            return False
        
        print(f"✓ Loaded {len(sample_texts)} sample texts")
        
        # Test tokenization
        print("🔧 Testing tokenization process...")
        start_time = time.time()
        
        total_tokens = 0
        total_sequences = 0
        test_results = []
        
        for i, text in enumerate(sample_texts[:10]):  # Test first 10
            try:
                # Tokenize
                encoding = tokenizer.encode(text)
                tokens = encoding.tokens
                token_ids = encoding.ids
                
                total_tokens += len(token_ids)
                
                # Simulate sequence splitting (1024 max length)
                if len(token_ids) <= 1024:
                    sequences = 1
                else:
                    sequences = 1 + ((len(token_ids) - 1024) // 512)
                
                total_sequences += sequences
                
                test_results.append({
                    'text_length': len(text),
                    'token_count': len(token_ids),
                    'sequence_count': sequences,
                    'first_tokens': tokens[:5] if len(tokens) > 5 else tokens
                })
                
            except Exception as e:
                print(f"⚠️  Error tokenizing text {i}: {e}")
                continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not test_results:
            print("❌ No successful tokenizations")
            return False
        
        # Display results
        print("✅ Tokenization test successful!")
        print(f"📊 Test Results:")
        print(f"   Processed texts: {len(test_results)}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total sequences: {total_sequences}")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Tokens per second: {total_tokens/processing_time:.0f}")
        
        # Show sample tokenization
        if test_results:
            sample = test_results[0]
            print(f"\n🔍 Sample tokenization:")
            print(f"   Text length: {sample['text_length']} chars")
            print(f"   Token count: {sample['token_count']}")
            print(f"   Sequences: {sample['sequence_count']}")
            print(f"   First tokens: {sample['first_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")
        return False

def check_system_resources():
    """Check system resources and provide recommendations."""
    print("\n💻 System Resource Analysis:")
    
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        memory_percent = memory.percent
        
        # Disk info
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        
        print(f"🖥️  CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")
        print(f"💾 Memory: {memory_gb:.1f}GB total, {available_gb:.1f}GB available ({memory_percent:.1f}% used)")
        print(f"💿 Disk: {free_gb:.1f}GB free space")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if memory_gb < 4:
            print("   ⚠️  Low memory system - Use single-threaded scripts")
            print("   ✓ Recommended: run_full_tokenization.py")
        elif memory_gb < 8:
            print("   📊 Medium memory system - Limited parallel processing")
            print("   ✓ Recommended: tokenize_data.py with 2-4 workers")
        else:
            print("   🚀 High memory system - Full parallel processing available")
            print("   ✓ Recommended: tokenize_data.py with 4-8 workers")
        
        if available_gb < 2:
            print("   ❌ Very low available memory - Close applications before processing")
        
        if free_gb < 5:
            print("   ⚠️  Low disk space - Monitor output size carefully")
        
        if cpu_count >= 4:
            print("   🔥 Multiple CPU cores available - Parallel processing beneficial")
        
    except ImportError:
        print("   ⚠️  psutil not available - install for detailed system analysis")

def show_next_steps(tokenizer_available: bool, data_available: bool):
    """Show recommended next steps based on current status."""
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    print("=" * 50)
    
    if not tokenizer_available:
        print("1. 🔧 Train a tokenizer first:")
        print("   cd ../03_tokenizer_training")
        print("   python train_tokenizer.py")
        print("")
        print("2. 🔙 Return here and run setup again:")
        print("   cd ../04_data_tokenization")
        print("   python setup_and_validate.py")
        
    elif not data_available:
        print("1. 📁 Prepare training data:")
        print("   - Ensure data is in JSONL format")
        print("   - Place files in data/cleaned/ directory")
        print("")
        print("2. 🔄 Or run data collection/processing:")
        print("   cd ../01_data_collection")
        print("   python multi_source_collector_infinite.py")
        print("   cd ../02_data_processing") 
        print("   python data_cleaner.py")
        
    else:
        print("✅ Environment is ready! Recommended workflow:")
        print("")
        print("1. 🧪 Quick test (recommended first step):")
        print("   python test_cloud_tokenization_10min.py")
        print("")
        print("2. 🔍 Compare approaches:")
        print("   python tokenization_comparison.py")
        print("")
        print("3. 🚀 Run production tokenization:")
        print("   python run_full_tokenization.py")
        print("")
        print("4. 📊 If interrupted, resume with:")
        print("   python resume_tokenization.py")

def main():
    """Main setup and validation function."""
    print("🚀 DATA TOKENIZATION SETUP & VALIDATION")
    print("=" * 60)
    print("Validating environment and testing tokenization functionality.")
    
    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Please install missing dependencies before proceeding.")
        sys.exit(1)
    
    # Check tokenizers
    tokenizers = check_tokenizer_availability()
    tokenizer_available = any(tokenizers.values())
    available_tokenizer = None
    
    if tokenizer_available:
        # Use first available tokenizer
        for name, path in tokenizers.items():
            if path:
                available_tokenizer = path
                break
    
    # Check data
    data_status = check_data_availability()
    data_available = any(info['file_count'] > 0 for info in data_status.values())
    
    # Check output directories
    output_status = check_output_directories()
    
    # Check system resources
    check_system_resources()
    
    # Test functionality if possible
    if tokenizer_available:
        if data_available:
            # Use existing data
            for name, info in data_status.items():
                if info['file_count'] > 0:
                    test_file = info['files'][0]  # Use first file
                    break
        else:
            # Create sample data
            test_file = create_sample_data()
        
        # Run tokenization test
        test_success = test_tokenization_functionality(available_tokenizer, test_file)
        
        # Cleanup sample data if created
        if not data_available and test_file.parent.name == "temp_sample_data":
            import shutil
            shutil.rmtree(test_file.parent)
            print("🧹 Cleaned up sample data")
        
        if test_success:
            print("\n✅ Tokenization functionality validated!")
        else:
            print("\n⚠️  Tokenization test had issues - check configuration")
    
    # Show next steps
    show_next_steps(tokenizer_available, data_available)
    
    print("\n" + "=" * 60)
    print("✅ SETUP AND VALIDATION COMPLETE!")
    print("\n📖 For detailed documentation, see DATA_TOKENIZATION_GUIDE.md")
    print("🔧 For script comparison, run: python tokenization_comparison.py")

if __name__ == "__main__":
    main() 