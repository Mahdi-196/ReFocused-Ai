#!/usr/bin/env python3
"""
Tokenization Scripts Comparison and Performance Tool
====================================================

Compare different tokenization approaches with practical examples.
Helps choose the right script for your dataset size and requirements.

Requirements:
- tokenizers
- transformers
- numpy
- psutil
- tabulate

Author: AI Assistant
Version: 1.0
"""

import os
import json
import time
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile

try:
    import numpy as np
    from tokenizers import Tokenizer
    from tabulate import tabulate
    print("✓ All required packages are available")
except ImportError as e:
    print(f"✗ Missing required package: {e}")
    print("Please install missing packages:")
    print("pip install numpy tokenizers tabulate psutil")
    exit(1)


class TokenizationComparison:
    """Compare different tokenization scripts and approaches."""
    
    def __init__(self):
        self.scripts = {
            "test_cloud_tokenization_10min": {
                "name": "Quick Test (10min)",
                "purpose": "Fast validation and testing",
                "best_for": "Testing setup and performance",
                "file": "test_cloud_tokenization_10min.py",
                "expected_rate": "~3,200 seq/sec",
                "memory_usage": "Low",
                "stability": "High"
            },
            "tokenize_data": {
                "name": "Multi-threaded Basic",
                "purpose": "Standard parallel processing",
                "best_for": "Development and medium datasets",
                "file": "tokenize_data.py",
                "expected_rate": "~2,800 seq/sec",
                "memory_usage": "Medium",
                "stability": "Medium"
            },
            "run_full_tokenization": {
                "name": "Production Single-threaded",
                "purpose": "Reliable production processing",
                "best_for": "Large datasets and production",
                "file": "run_full_tokenization.py",
                "expected_rate": "~3,000 seq/sec",
                "memory_usage": "Low",
                "stability": "Very High"
            },
            "resume_tokenization": {
                "name": "Resume Interrupted",
                "purpose": "Continue from interruption",
                "best_for": "Recovery from failures",
                "file": "resume_tokenization.py",
                "expected_rate": "~3,000 seq/sec",
                "memory_usage": "Low",
                "stability": "High"
            },
            "fix_tokenization_single_threaded": {
                "name": "Debug Single-threaded",
                "purpose": "Debugging and error diagnosis",
                "best_for": "Troubleshooting issues",
                "file": "fix_tokenization_single_threaded.py",
                "expected_rate": "~2,500 seq/sec",
                "memory_usage": "Low",
                "stability": "Very High"
            }
        }
        
        self.dataset_sizes = {
            "tiny": {"size": "< 100MB", "files": "1-5", "sequences": "< 50K"},
            "small": {"size": "100MB - 1GB", "files": "5-20", "sequences": "50K - 500K"},
            "medium": {"size": "1GB - 10GB", "files": "20-100", "sequences": "500K - 5M"},
            "large": {"size": "10GB - 100GB", "files": "100-500", "sequences": "5M - 50M"},
            "xl": {"size": "> 100GB", "files": "> 500", "sequences": "> 50M"}
        }
        
        self.performance_data = {}
        self.system_info = self.get_system_info()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
        }
    
    def create_sample_data(self, size_category: str) -> Path:
        """Create sample data for testing."""
        print(f"📝 Creating sample data for {size_category} dataset...")
        
        # Define data parameters based on size
        params = {
            "tiny": {"files": 2, "entries_per_file": 100},
            "small": {"files": 5, "entries_per_file": 500},
            "medium": {"files": 10, "entries_per_file": 1000},
            "large": {"files": 20, "entries_per_file": 2000},
            "xl": {"files": 50, "entries_per_file": 5000}
        }
        
        config = params.get(size_category, params["small"])
        
        # Sample texts of varying lengths
        sample_texts = [
            "This is a short example for tokenization testing.",
            "Natural language processing involves converting text into numerical representations that machine learning models can understand and process effectively.",
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing text processing systems.",
            "Artificial intelligence and machine learning are revolutionizing how we process and understand human language. Tokenization is a fundamental step in this process that converts raw text into meaningful units.",
            "In computational linguistics, tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing such as parsing or text mining.",
            "Deep learning models, particularly transformer architectures, have shown remarkable success in natural language understanding tasks. These models require carefully preprocessed and tokenized input data to achieve optimal performance.",
            "Large language models are trained on massive datasets containing billions of tokens. The quality of tokenization directly impacts the model's ability to learn linguistic patterns and generate coherent text.",
            "Byte-pair encoding, WordPiece, and SentencePiece are popular tokenization algorithms used in modern language models. Each has its own advantages and is suited to different types of text and languages."
        ]
        
        # Create temporary directory
        temp_dir = Path("temp_sample_data")
        temp_dir.mkdir(exist_ok=True)
        
        created_files = []
        total_entries = 0
        
        for i in range(config["files"]):
            file_path = temp_dir / f"sample_data_{size_category}_{i+1:03d}.jsonl"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for j in range(config["entries_per_file"]):
                    # Vary text length and content
                    text_idx = (i + j) % len(sample_texts)
                    text = sample_texts[text_idx]
                    
                    # Add some variation
                    if j % 3 == 0:
                        text = f"Entry {j}: {text}"
                    elif j % 3 == 1:
                        text = f"{text} Additional context for entry {j}."
                    
                    entry = {
                        "text": text,
                        "title": f"Sample Title {j}",
                        "subreddit": "sample_data",
                        "entry_id": f"{i}_{j}"
                    }
                    
                    f.write(json.dumps(entry) + '\n')
                    total_entries += 1
            
            created_files.append(file_path)
        
        total_size_mb = sum(f.stat().st_size for f in created_files) / (1024**2)
        
        print(f"✓ Created {len(created_files)} files with {total_entries} entries ({total_size_mb:.1f}MB)")
        return temp_dir
    
    def estimate_performance(self, data_dir: Path, sample_size: int = 100) -> Dict[str, Any]:
        """Estimate tokenization performance on sample data."""
        print(f"⚡ Estimating performance with {sample_size} samples...")
        
        # Find tokenizer
        tokenizer_paths = [
            "models/tokenizer/transformers_tokenizer/tokenizer.json",
            "models/tokenizer/tokenizer.json",
            "../03_tokenizer_training/test_tokenizer_output/tokenizer.json",
            "tokenizer_750M/tokenizer.json"
        ]
        
        tokenizer = None
        for path in tokenizer_paths:
            if Path(path).exists():
                try:
                    tokenizer = Tokenizer.from_file(path)
                    print(f"✓ Using tokenizer from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        
        if tokenizer is None:
            print("❌ No tokenizer found. Please train a tokenizer first.")
            return {"error": "No tokenizer available"}
        
        # Sample data for performance testing
        data_files = list(data_dir.glob("*.jsonl"))
        if not data_files:
            return {"error": "No data files found"}
        
        # Read sample texts
        sample_texts = []
        for file_path in data_files[:3]:  # Use first 3 files
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_size // len(data_files[:3]):
                        break
                    try:
                        data = json.loads(line)
                        text = data.get('text', '') + ' ' + data.get('title', '')
                        sample_texts.append(text.strip())
                    except:
                        continue
        
        if not sample_texts:
            return {"error": "No valid texts found"}
        
        # Performance test
        start_time = time.time()
        total_tokens = 0
        total_sequences = 0
        
        for text in sample_texts[:sample_size]:
            try:
                encoding = tokenizer.encode(text)
                tokens = len(encoding.ids)
                total_tokens += tokens
                
                # Simulate sequence splitting (1024 max length, 512 stride)
                if tokens <= 1024:
                    total_sequences += 1
                else:
                    sequences = 1 + ((tokens - 1024) // 512)
                    total_sequences += sequences
                    
            except Exception:
                continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate metrics
        texts_per_second = len(sample_texts) / processing_time if processing_time > 0 else 0
        sequences_per_second = total_sequences / processing_time if processing_time > 0 else 0
        tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0
        
        return {
            "processing_time": processing_time,
            "total_texts": len(sample_texts),
            "total_sequences": total_sequences,
            "total_tokens": total_tokens,
            "texts_per_second": texts_per_second,
            "sequences_per_second": sequences_per_second,
            "tokens_per_second": tokens_per_second,
            "avg_tokens_per_text": total_tokens / len(sample_texts) if sample_texts else 0,
            "avg_sequences_per_text": total_sequences / len(sample_texts) if sample_texts else 0
        }
    
    def show_script_comparison(self):
        """Display comparison table of all tokenization scripts."""
        print("\n🔧 TOKENIZATION SCRIPTS COMPARISON")
        print("=" * 80)
        
        table_data = []
        for script_id, info in self.scripts.items():
            table_data.append([
                info["name"],
                info["purpose"],
                info["best_for"],
                info["expected_rate"],
                info["memory_usage"],
                info["stability"]
            ])
        
        headers = ["Script", "Purpose", "Best For", "Speed", "Memory", "Stability"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def show_dataset_recommendations(self):
        """Show recommendations based on dataset size."""
        print("\n📊 DATASET SIZE RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = {
            "tiny": "test_cloud_tokenization_10min → Quick validation",
            "small": "test_cloud_tokenization_10min → Then tokenize_data",
            "medium": "tokenize_data → Multi-threaded processing",
            "large": "run_full_tokenization → Production single-threaded",
            "xl": "run_full_tokenization → With resume_tokenization for recovery"
        }
        
        table_data = []
        for size, info in self.dataset_sizes.items():
            recommendation = recommendations[size]
            table_data.append([
                size.upper(),
                info["size"],
                info["files"],
                info["sequences"],
                recommendation
            ])
        
        headers = ["Category", "Size", "Files", "Sequences", "Recommended Script"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def show_system_analysis(self):
        """Analyze current system capabilities."""
        print("\n💻 SYSTEM ANALYSIS")
        print("=" * 80)
        
        info = self.system_info
        
        # System specs
        print(f"🖥️  System Specifications:")
        print(f"   CPU Cores: {info['cpu_count']}")
        print(f"   Total Memory: {info['memory_gb']:.1f} GB")
        print(f"   Available Memory: {info['available_memory_gb']:.1f} GB")
        print(f"   Free Disk Space: {info['disk_free_gb']:.1f} GB")
        
        # Current usage
        print(f"\n📈 Current Usage:")
        print(f"   CPU Usage: {info['cpu_percent']:.1f}%")
        print(f"   Memory Usage: {info['memory_percent']:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if info['memory_gb'] < 4:
            print("   ⚠️  Low memory - Use single-threaded scripts")
            print("   ✓ Recommended: run_full_tokenization.py")
        elif info['memory_gb'] < 8:
            print("   📊 Medium memory - Limited parallel processing")
            print("   ✓ Recommended: tokenize_data.py with 2-4 workers")
        else:
            print("   🚀 High memory - Full parallel processing available")
            print("   ✓ Recommended: tokenize_data.py with 4-8 workers")
        
        if info['cpu_count'] < 4:
            print("   ⚠️  Limited CPU cores - Single-threaded recommended")
        else:
            print("   🔥 Multiple cores available - Parallel processing beneficial")
        
        if info['available_memory_gb'] < 2:
            print("   ❌ Very low available memory - Free up space before processing")
        
        if info['disk_free_gb'] < 10:
            print("   ⚠️  Low disk space - Monitor output size carefully")
    
    def run_performance_test(self, size_category: str = "small"):
        """Run a performance test with sample data."""
        print(f"\n⚡ PERFORMANCE TEST - {size_category.upper()} DATASET")
        print("=" * 80)
        
        # Create sample data
        data_dir = self.create_sample_data(size_category)
        
        try:
            # Run performance estimation
            perf_results = self.estimate_performance(data_dir)
            
            if "error" in perf_results:
                print(f"❌ Performance test failed: {perf_results['error']}")
                return
            
            # Display results
            print(f"📊 Performance Results:")
            print(f"   Processing Time: {perf_results['processing_time']:.2f} seconds")
            print(f"   Total Texts: {perf_results['total_texts']:,}")
            print(f"   Total Sequences: {perf_results['total_sequences']:,}")
            print(f"   Total Tokens: {perf_results['total_tokens']:,}")
            print(f"   ")
            print(f"   📈 Processing Rates:")
            print(f"   Texts/second: {perf_results['texts_per_second']:.1f}")
            print(f"   Sequences/second: {perf_results['sequences_per_second']:.1f}")
            print(f"   Tokens/second: {perf_results['tokens_per_second']:.1f}")
            print(f"   ")
            print(f"   📊 Averages:")
            print(f"   Tokens per text: {perf_results['avg_tokens_per_text']:.1f}")
            print(f"   Sequences per text: {perf_results['avg_sequences_per_text']:.1f}")
            
            # Extrapolate to full dataset sizes
            print(f"\n🔮 Extrapolated Full Dataset Performance:")
            
            for size, info in self.dataset_sizes.items():
                if size == "tiny":
                    continue
                    
                # Estimate number of sequences for this size
                size_multipliers = {
                    "small": 10, "medium": 100, "large": 1000, "xl": 10000
                }
                
                multiplier = size_multipliers.get(size, 1)
                estimated_sequences = perf_results['total_sequences'] * multiplier
                estimated_time = estimated_sequences / perf_results['sequences_per_second']
                
                hours = estimated_time // 3600
                minutes = (estimated_time % 3600) // 60
                
                time_str = f"{int(hours)}h {int(minutes)}m" if hours > 0 else f"{int(minutes)}m"
                
                print(f"   {size.upper()}: ~{estimated_sequences:,} sequences → ~{time_str}")
        
        finally:
            # Cleanup sample data
            import shutil
            if data_dir.exists():
                shutil.rmtree(data_dir)
                print(f"\n🧹 Cleaned up sample data")
    
    def show_usage_examples(self):
        """Show practical usage examples for each script."""
        print("\n📚 USAGE EXAMPLES")
        print("=" * 80)
        
        examples = {
            "Quick Testing": {
                "command": "python test_cloud_tokenization_10min.py",
                "description": "10-minute validation test",
                "when": "First time setup, testing environment",
                "output": "Quick validation results and performance metrics"
            },
            "Development": {
                "command": "python tokenize_data.py --workers 4",
                "description": "Multi-threaded processing",
                "when": "Medium datasets, development phase",
                "output": "Parallel processing with progress tracking"
            },
            "Production": {
                "command": "python run_full_tokenization.py",
                "description": "Reliable single-threaded processing",
                "when": "Large datasets, production deployment",
                "output": "Stable processing with detailed logging"
            },
            "Resume Job": {
                "command": "python resume_tokenization.py",
                "description": "Continue interrupted tokenization",
                "when": "Recovery after interruption or failure",
                "output": "Picks up where it left off automatically"
            },
            "Debug Issues": {
                "command": "python fix_tokenization_single_threaded.py",
                "description": "Single-threaded debugging",
                "when": "Troubleshooting errors or issues",
                "output": "Detailed error reporting and step-by-step processing"
            }
        }
        
        for scenario, info in examples.items():
            print(f"\n🎯 {scenario}:")
            print(f"   Command: {info['command']}")
            print(f"   Purpose: {info['description']}")
            print(f"   When: {info['when']}")
            print(f"   Output: {info['output']}")
    
    def show_troubleshooting_guide(self):
        """Show quick troubleshooting guide."""
        print("\n🛠️  QUICK TROUBLESHOOTING")
        print("=" * 80)
        
        issues = {
            "Out of Memory": {
                "symptoms": "MemoryError, system freezing",
                "solutions": [
                    "Use run_full_tokenization.py (memory optimized)",
                    "Reduce batch_size in script configuration",
                    "Close other applications",
                    "Use smaller max_length (512 instead of 1024)"
                ]
            },
            "Slow Processing": {
                "symptoms": "Very low sequences/second rate",
                "solutions": [
                    "Check system resources (CPU, memory, disk)",
                    "Use tokenize_data.py with more workers",
                    "Ensure SSD storage for faster I/O",
                    "Close unnecessary applications"
                ]
            },
            "Tokenizer Not Found": {
                "symptoms": "FileNotFoundError for tokenizer.json",
                "solutions": [
                    "Train tokenizer first: cd ../03_tokenizer_training && python train_tokenizer.py",
                    "Check tokenizer path in script configuration",
                    "Verify tokenizer files exist in models/tokenizer/"
                ]
            },
            "Corrupted Output": {
                "symptoms": "Unable to load .npz files, shape mismatches",
                "solutions": [
                    "Use resume_tokenization.py to reprocess",
                    "Delete corrupted files and re-run",
                    "Check available disk space",
                    "Verify input data integrity"
                ]
            }
        }
        
        for issue, info in issues.items():
            print(f"\n❌ {issue}:")
            print(f"   Symptoms: {info['symptoms']}")
            print(f"   Solutions:")
            for solution in info['solutions']:
                print(f"     • {solution}")
    
    def run_comprehensive_analysis(self):
        """Run the complete comparison and analysis."""
        print("🎯 COMPREHENSIVE TOKENIZATION ANALYSIS")
        print("=" * 80)
        print("Analyzing tokenization scripts, performance, and providing recommendations.")
        
        # Show script comparison
        self.show_script_comparison()
        
        # Show dataset recommendations
        self.show_dataset_recommendations()
        
        # Analyze system
        self.show_system_analysis()
        
        # Show usage examples
        self.show_usage_examples()
        
        # Run performance test
        self.run_performance_test("small")
        
        # Show troubleshooting
        self.show_troubleshooting_guide()
        
        print("\n" + "=" * 80)
        print("🎉 ANALYSIS COMPLETE!")
        print("\n💡 QUICK START RECOMMENDATIONS:")
        
        # Quick recommendations based on system
        info = self.system_info
        if info['memory_gb'] < 4:
            print("   1. 🧪 Test: python test_cloud_tokenization_10min.py")
            print("   2. 🚀 Production: python run_full_tokenization.py")
        else:
            print("   1. 🧪 Test: python test_cloud_tokenization_10min.py")
            print("   2. 🔧 Development: python tokenize_data.py")
            print("   3. 🚀 Production: python run_full_tokenization.py")
        
        print("\n📖 See DATA_TOKENIZATION_GUIDE.md for detailed documentation.")


def main():
    """Main function to run tokenization comparison."""
    comparison = TokenizationComparison()
    comparison.run_comprehensive_analysis()


if __name__ == "__main__":
    main() 