#!/usr/bin/env python3
"""
Massive Dataset Setup for ReFocused AI
Handles multiple large datasets: 13GB Reddit + 41GB OpenWebText + more
Optimized for up to 150GB total processing capacity
"""

import gzip
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import subprocess
from datetime import datetime
from loguru import logger
import psutil
from datasets import load_dataset
import dask.dataframe as dd
from tqdm import tqdm

class MassiveDatasetManager:
    """Manages multiple large datasets for cleaning - Reddit + HuggingFace + more"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.reddit_dir = self.base_dir / "reddit_large"
        self.huggingface_dir = self.base_dir / "huggingface_cache"
        self.unified_dir = self.base_dir / "unified_raw"
        self.cleaned_dir = self.base_dir / "cleaned"
        self.processed_dir = self.base_dir / "processed"
        self.logs_dir = Path("logs")
        
        # Create all directories
        for directory in [self.reddit_dir, self.huggingface_dir, self.unified_dir, 
                         self.cleaned_dir, self.processed_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # System specs for optimization
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_cores = psutil.cpu_count()
        self.available_disk_gb = psutil.disk_usage('.').free / (1024**3)
        
        logger.info(f"💻 System: {self.total_memory_gb:.1f}GB RAM, {self.cpu_cores} cores, {self.available_disk_gb:.1f}GB free")
    
    def analyze_system_capacity(self) -> Dict:
        """Analyze system capacity for massive dataset processing"""
        logger.info("🔍 Analyzing system capacity for 150GB processing...")
        
        # Memory-based recommendations
        if self.total_memory_gb >= 32:
            recommended_batch = 10000
            recommended_workers = min(self.cpu_cores, 12)
            chunk_size_mb = 500
        elif self.total_memory_gb >= 16:
            recommended_batch = 7500
            recommended_workers = min(self.cpu_cores, 8)
            chunk_size_mb = 250
        elif self.total_memory_gb >= 8:
            recommended_batch = 5000
            recommended_workers = min(self.cpu_cores, 6)
            chunk_size_mb = 100
        else:
            recommended_batch = 2000
            recommended_workers = min(self.cpu_cores, 4)
            chunk_size_mb = 50
        
        # Disk space check
        disk_warning = self.available_disk_gb < 200  # Need ~200GB for 150GB processing
        
        capacity = {
            'total_memory_gb': self.total_memory_gb,
            'cpu_cores': self.cpu_cores,
            'available_disk_gb': self.available_disk_gb,
            'recommended_batch_size': recommended_batch,
            'recommended_workers': recommended_workers,
            'chunk_size_mb': chunk_size_mb,
            'disk_warning': disk_warning,
            'estimated_max_dataset_gb': min(150, self.available_disk_gb * 0.7),
            'parallel_source_processing': self.cpu_cores >= 6
        }
        
        if disk_warning:
            logger.warning(f"⚠️ Only {self.available_disk_gb:.1f}GB disk space - recommend 200GB+ for 150GB processing")
        
        return capacity
    
    def setup_huggingface_dataset(self, dataset_name: str = "Skylion007/openwebtext", 
                                  streaming: bool = True, sample_size: Optional[int] = None) -> Dict:
        """Setup and analyze Hugging Face dataset"""
        logger.info(f"🤗 Setting up Hugging Face dataset: {dataset_name}")
        
        try:
            # Load dataset info first
            if streaming:
                logger.info("📡 Using streaming mode (no local download)")
                dataset = load_dataset(dataset_name, streaming=True, split="train")
            else:
                logger.info("💾 Downloading dataset locally (41GB)")
                dataset = load_dataset(dataset_name, split="train")
            
            # Analyze sample
            sample_count = 0
            total_chars = 0
            avg_text_length = 0
            
            logger.info("🔍 Analyzing dataset sample...")
            for i, example in enumerate(dataset):
                if sample_size and i >= sample_size:
                    break
                if i >= 10000:  # Sample first 10K for analysis
                    break
                    
                text = example.get("text", "")
                total_chars += len(text)
                sample_count += 1
                
                if i % 1000 == 0:
                    logger.info(f"📊 Analyzed {i:,} examples...")
            
            if sample_count > 0:
                avg_text_length = total_chars / sample_count
                estimated_total_records = 8000000 if not sample_size else sample_size  # ~8M records in OpenWebText
                estimated_total_gb = (estimated_total_records * avg_text_length) / (1024**3)
            else:
                estimated_total_records = 8000000
                estimated_total_gb = 41  # Known size
            
            analysis = {
                'dataset_name': dataset_name,
                'streaming': streaming,
                'sample_size': sample_size or estimated_total_records,
                'avg_text_length': avg_text_length,
                'estimated_records': estimated_total_records,
                'estimated_size_gb': estimated_total_gb,
                'source_type': 'huggingface',
                'format': 'text'
            }
            
            logger.success(f"✅ HuggingFace analysis: ~{estimated_total_records:,} records, ~{estimated_total_gb:.1f}GB")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error setting up HuggingFace dataset: {e}")
            return {}
    
    def analyze_reddit_dataset(self, file_path: Path) -> Dict:
        """Analyze Reddit dataset (from existing setup)"""
        logger.info(f"📱 Analyzing Reddit dataset: {file_path}")
        
        try:
            file_size = file_path.stat().st_size / (1024**3)  # GB
            
            # Quick sample analysis
            sample_lines = []
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if i >= 100:  # Sample more lines for better estimate
                            break
                        sample_lines.append(line.strip())
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if i >= 100:
                            break
                        sample_lines.append(line.strip())
            
            # Analyze format and estimate records
            json_lines = 0
            total_chars = 0
            for line in sample_lines:
                if line:
                    try:
                        data = json.loads(line)
                        json_lines += 1
                        total_chars += len(str(data))
                    except json.JSONDecodeError:
                        continue
            
            if json_lines > 0:
                avg_line_size = total_chars / json_lines
                estimated_records = int((file_size * 1024**3) / avg_line_size)
            else:
                estimated_records = int(file_size * 200000)  # Fallback estimate
            
            analysis = {
                'file_path': str(file_path),
                'size_gb': file_size,
                'format': 'JSONL' if json_lines > len(sample_lines) * 0.8 else 'Unknown',
                'estimated_records': estimated_records,
                'compressed': file_path.suffix == '.gz',
                'source_type': 'reddit',
                'avg_line_size': avg_line_size if json_lines > 0 else 0
            }
            
            logger.success(f"✅ Reddit analysis: ~{estimated_records:,} records, {file_size:.2f}GB")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Reddit dataset: {e}")
            return {}
    
    def create_unified_processing_plan(self, sources: List[Dict]) -> Dict:
        """Create unified processing plan for all data sources"""
        logger.info("📋 Creating unified processing plan...")
        
        total_records = sum(source.get('estimated_records', 0) for source in sources)
        total_size_gb = sum(source.get('estimated_size_gb', 0) or source.get('size_gb', 0) for source in sources)
        
        # Get system capacity
        capacity = self.analyze_system_capacity()
        
        # Optimize for total scale
        if total_records > 10000000:  # 10M+ records
            batch_size = capacity['recommended_batch_size']
            workers = capacity['recommended_workers']
            chunk_size_mb = capacity['chunk_size_mb']
            process_in_stages = True
        else:
            batch_size = min(capacity['recommended_batch_size'], 5000)
            workers = min(capacity['recommended_workers'], 6)
            chunk_size_mb = min(capacity['chunk_size_mb'], 200)
            process_in_stages = False
        
        # Time estimates (conservative)
        posts_per_hour = 2000 * workers  # Conservative estimate
        estimated_hours = total_records / posts_per_hour
        
        plan = {
            'sources': sources,
            'total_estimated_records': total_records,
            'total_estimated_size_gb': total_size_gb,
            'processing_config': {
                'batch_size': batch_size,
                'workers': workers,
                'chunk_size_mb': chunk_size_mb,
                'process_in_stages': process_in_stages,
                'memory_efficient': True
            },
            'time_estimates': {
                'estimated_hours': estimated_hours,
                'estimated_days': estimated_hours / 24,
                'posts_per_hour': posts_per_hour
            },
            'system_requirements': {
                'recommended_ram_gb': 16 if total_size_gb > 50 else 8,
                'recommended_disk_gb': total_size_gb * 3,  # 3x for processing overhead
                'recommended_cpu_cores': max(4, workers)
            }
        }
        
        logger.info(f"📊 Plan: {total_records:,} records, {total_size_gb:.1f}GB, ~{estimated_hours:.1f}h processing")
        return plan
    
    def extract_reddit_data(self, file_path: Path, chunk_size_mb: int = 100) -> List[Path]:
        """Extract Reddit data into chunks (from existing logic)"""
        logger.info(f"📦 Extracting Reddit data: {file_path}")
        
        output_files = []
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    self._extract_reddit_chunks(f, output_files, chunk_size_bytes, "reddit")
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    self._extract_reddit_chunks(f, output_files, chunk_size_bytes, "reddit")
            
            logger.success(f"✅ Reddit extracted: {len(output_files)} chunks")
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Reddit extraction failed: {e}")
            return []
    
    def extract_huggingface_data(self, dataset_name: str, chunk_size_mb: int = 100, 
                                max_samples: Optional[int] = None) -> List[Path]:
        """Extract HuggingFace data into chunks"""
        logger.info(f"🤗 Extracting HuggingFace data: {dataset_name}")
        
        output_files = []
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
        try:
            # Use streaming to avoid downloading entire dataset
            dataset = load_dataset(dataset_name, streaming=True, split="train")
            
            chunk_num = 1
            current_chunk_size = 0
            current_chunk_path = None
            current_chunk_file = None
            processed_count = 0
            
            logger.info("🔄 Processing HuggingFace dataset...")
            
            for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
                if max_samples and i >= max_samples:
                    break
                
                # Start new chunk if needed
                if current_chunk_file is None or current_chunk_size >= chunk_size_bytes:
                    if current_chunk_file:
                        current_chunk_file.close()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_chunk_path = self.unified_dir / f"huggingface_chunk_{chunk_num:03d}_{timestamp}.txt"
                    current_chunk_file = open(current_chunk_path, 'w', encoding='utf-8')
                    output_files.append(current_chunk_path)
                    current_chunk_size = 0
                    chunk_num += 1
                    
                    if chunk_num % 10 == 1:  # Log every 10 chunks
                        logger.info(f"📝 Started HF chunk {chunk_num-1}: {current_chunk_path.name}")
                
                # Convert to unified format and write
                text = example.get("text", "")
                if text.strip():  # Only process non-empty texts
                    unified_record = {
                        "id": f"hf_{i}",
                        "text": text,
                        "source": "huggingface_openwebtext",
                        "source_dataset": dataset_name,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    line = json.dumps(unified_record, ensure_ascii=False) + '\n'
                    current_chunk_file.write(line)
                    current_chunk_size += len(line.encode('utf-8'))
                    processed_count += 1
                
                # Progress update
                if (i + 1) % 50000 == 0:
                    logger.info(f"📊 HF processed {i+1:,} examples, {len(output_files)} chunks created")
            
            if current_chunk_file:
                current_chunk_file.close()
            
            logger.success(f"✅ HuggingFace extracted: {processed_count:,} records in {len(output_files)} chunks")
            return output_files
            
        except Exception as e:
            logger.error(f"❌ HuggingFace extraction failed: {e}")
            return []
    
    def _extract_reddit_chunks(self, file_obj, output_files: List[Path], 
                              chunk_size_bytes: int, source_prefix: str):
        """Helper to extract Reddit data in chunks with unified format"""
        chunk_num = 1
        current_chunk_size = 0
        current_chunk_path = None
        current_chunk_file = None
        
        try:
            for line_num, line in enumerate(file_obj, 1):
                # Start new chunk if needed
                if current_chunk_file is None or current_chunk_size >= chunk_size_bytes:
                    if current_chunk_file:
                        current_chunk_file.close()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_chunk_path = self.unified_dir / f"{source_prefix}_chunk_{chunk_num:03d}_{timestamp}.txt"
                    current_chunk_file = open(current_chunk_path, 'w', encoding='utf-8')
                    output_files.append(current_chunk_path)
                    current_chunk_size = 0
                    chunk_num += 1
                    
                    if chunk_num % 20 == 1:  # Log every 20 chunks
                        logger.info(f"📝 Started {source_prefix} chunk {chunk_num-1}")
                
                # Parse and convert to unified format
                line = line.strip()
                if line:
                    try:
                        reddit_data = json.loads(line)
                        
                        # Convert to unified format
                        unified_record = {
                            "id": reddit_data.get("id", f"reddit_{line_num}"),
                            "text": f"{reddit_data.get('title', '')} {reddit_data.get('selftext', '')}".strip(),
                            "source": "reddit",
                            "subreddit": reddit_data.get("subreddit", ""),
                            "score": reddit_data.get("score", 0),
                            "created_utc": reddit_data.get("created_utc", 0),
                            "processed_at": datetime.now().isoformat()
                        }
                        
                        if unified_record["text"]:  # Only write if there's text content
                            unified_line = json.dumps(unified_record, ensure_ascii=False) + '\n'
                            current_chunk_file.write(unified_line)
                            current_chunk_size += len(unified_line.encode('utf-8'))
                        
                    except json.JSONDecodeError:
                        continue
                
                # Progress update
                if line_num % 100000 == 0:
                    logger.info(f"📊 {source_prefix} processed {line_num:,} lines, {len(output_files)} chunks")
        
        finally:
            if current_chunk_file:
                current_chunk_file.close()
    
    def create_massive_processing_script(self, plan: Dict) -> Path:
        """Create optimized processing script for massive dataset"""
        script_path = Path("process_massive_dataset.py")
        
        config = plan['processing_config']
        
        script_content = f'''#!/usr/bin/env python3
"""
Massive Dataset Processing Script
Optimized for up to 150GB Reddit + HuggingFace datasets
"""

import sys
import time
import psutil
from pathlib import Path
from loguru import logger
from data_cleaner import RedditDataCleaner
from memory_profiler import profile

@profile
def main():
    """Process massive dataset with comprehensive monitoring"""
    logger.info("🚀 Starting MASSIVE dataset processing...")
    logger.info("📊 Target: Up to 150GB mixed data sources")
    
    # System monitoring setup
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.info(f"💾 Initial memory: {{initial_memory:.1f}} MB")
    logger.info(f"🔧 Config: {{config['workers']}} workers, {{config['batch_size']:,}} batch size")
    
    # Initialize cleaner with massive dataset settings
    cleaner = RedditDataCleaner(
        data_dir="data",
        output_dir="data/cleaned",
        n_workers={config['workers']}
    )
    
    # Override default settings for massive processing
    cleaner.stats['start_time'] = time.time()
    
    start_time = time.time()
    peak_memory = initial_memory
    
    try:
        # Process with optimized batch size
        logger.info("🔄 Starting cleaning pipeline...")
        cleaned_posts = cleaner.run_cleaning_pipeline(batch_size={config['batch_size']})
        
        # Final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
        
        # Comprehensive reporting
        total_processed = cleaner.stats.get('total_processed', 0)
        retention_rate = (len(cleaned_posts) / max(total_processed, 1)) * 100
        
        logger.success("🎉 MASSIVE PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {{processing_time/3600:.2f}} hours")
        logger.info(f"💾 Peak memory: {{peak_memory:.1f}} MB")
        logger.info(f"📊 Total processed: {{total_processed:,}} posts")
        logger.info(f"📊 Cleaned posts: {{len(cleaned_posts):,}}")
        logger.info(f"📈 Retention rate: {{retention_rate:.1f}}%")
        logger.info(f"⚡ Processing rate: {{len(cleaned_posts)/(processing_time/3600):.0f}} posts/hour")
        
        # Size calculations
        output_path = Path("data/cleaned/cleaned_reddit_data.jsonl")
        if output_path.exists():
            output_size_gb = output_path.stat().st_size / (1024**3)
            logger.info(f"💾 Output size: {{output_size_gb:.2f}} GB")
        
        logger.info("🎯 Ready for training data processing!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Processing failed: {{e}}")
        # Memory leak detection
        current_memory = process.memory_info().rss / 1024 / 1024
        if current_memory > initial_memory * 3:
            logger.warning(f"⚠️  Potential memory leak: {{current_memory:.1f}}MB vs {{initial_memory:.1f}}MB initial")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_path.write_text(script_content)
        logger.success(f"✅ Created massive processing script: {script_path}")
        return script_path
    
    def generate_comprehensive_report(self, plan: Dict) -> str:
        """Generate comprehensive setup report"""
        
        sources = plan['sources']
        config = plan['processing_config']
        estimates = plan['time_estimates']
        requirements = plan['system_requirements']
        
        report = f"""
🗂️  MASSIVE DATASET SETUP REPORT (150GB CAPACITY)
{'='*70}

📊 DATA SOURCES ANALYSIS:
"""
        
        for i, source in enumerate(sources, 1):
            source_type = source.get('source_type', 'unknown')
            records = source.get('estimated_records', 0)
            size = source.get('estimated_size_gb', 0) or source.get('size_gb', 0)
            
            report += f"""
   Source {i}: {source_type.upper()}
   └── Records: ~{records:,}
   └── Size: {size:.1f} GB
   └── Format: {source.get('format', 'Unknown')}"""
        
        report += f"""

🔧 OPTIMIZATION CONFIGURATION:
   Batch Size: {config['batch_size']:,} posts
   Workers: {config['workers']} parallel processes
   Chunk Size: {config['chunk_size_mb']} MB
   Memory Efficient: {config['memory_efficient']}
   Staged Processing: {config['process_in_stages']}

📁 DIRECTORY STRUCTURE:
   ✅ data/unified_raw/       - Extracted & unified chunks
   ✅ data/cleaned/           - Cleaned output  
   ✅ data/processed/         - Training splits
   ✅ data/huggingface_cache/ - HF dataset cache
   ✅ logs/                   - Processing logs

⏱️  PROCESSING ESTIMATES:
   Total Records: ~{plan['total_estimated_records']:,}
   Total Size: {plan['total_estimated_size_gb']:.1f} GB
   Estimated Time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)
   Processing Rate: {estimates['posts_per_hour']:,} posts/hour
   Expected Output: ~{int(plan['total_estimated_records'] * 0.65):,} clean posts

💻 SYSTEM REQUIREMENTS:
   Recommended RAM: {requirements['recommended_ram_gb']} GB
   Recommended Disk: {requirements['recommended_disk_gb']:.0f} GB free
   Recommended CPU: {requirements['recommended_cpu_cores']}+ cores
   Current System: {self.total_memory_gb:.1f}GB RAM, {self.cpu_cores} cores

🚀 PROCESSING WORKFLOW:
   1. Extract Reddit data: python setup_massive_dataset.py --reddit YOUR_FILE
   2. Extract HuggingFace: python setup_massive_dataset.py --huggingface
   3. Process all data: python process_massive_dataset.py
   4. Monitor progress: tail -f logs/massive_processing.log

⚡ PERFORMANCE OPTIMIZATIONS:
   • Unified data format for efficient processing
   • Memory-mapped file processing for large datasets
   • Parallel extraction and cleaning
   • Progressive batch size adjustment
   • Comprehensive memory monitoring

📈 EXPECTED OUTCOMES:
   • High-quality cleaned dataset
   • 60-80% retention rate
   • Training-ready format
   • Comprehensive quality reports
   • Memory and performance metrics
"""
        
        return report

def main():
    """Main setup function for massive datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup massive multi-source dataset for cleaning")
    parser.add_argument('--reddit', type=str, help='Path to Reddit dataset file')
    parser.add_argument('--huggingface', action='store_true', help='Setup HuggingFace OpenWebText dataset')
    parser.add_argument('--hf-dataset', type=str, default='Skylion007/openwebtext', help='HuggingFace dataset name')
    parser.add_argument('--hf-samples', type=int, help='Limit HuggingFace samples (for testing)')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, don\'t extract')
    parser.add_argument('--extract', action='store_true', help='Extract and prepare all datasets')
    
    args = parser.parse_args()
    
    manager = MassiveDatasetManager()
    sources = []
    
    # Analyze Reddit dataset if provided
    if args.reddit:
        reddit_file = Path(args.reddit)
        if reddit_file.exists():
            reddit_analysis = manager.analyze_reddit_dataset(reddit_file)
            if reddit_analysis:
                sources.append(reddit_analysis)
        else:
            logger.error(f"❌ Reddit file not found: {reddit_file}")
            return
    
    # Analyze HuggingFace dataset if requested
    if args.huggingface:
        hf_analysis = manager.setup_huggingface_dataset(
            dataset_name=args.hf_dataset,
            streaming=True,
            sample_size=args.hf_samples
        )
        if hf_analysis:
            sources.append(hf_analysis)
    
    if not sources:
        logger.error("❌ No data sources specified. Use --reddit or --huggingface")
        logger.info("💡 Usage examples:")
        logger.info("  python setup_massive_dataset.py --reddit data.gz --huggingface --analyze-only")
        logger.info("  python setup_massive_dataset.py --reddit data.gz --huggingface --extract")
        return
    
    # Create processing plan
    plan = manager.create_unified_processing_plan(sources)
    
    # Extract datasets if requested
    if args.extract and not args.analyze_only:
        logger.info("🔄 Starting data extraction...")
        
        for source in sources:
            if source['source_type'] == 'reddit':
                manager.extract_reddit_data(Path(source['file_path']))
            elif source['source_type'] == 'huggingface':
                manager.extract_huggingface_data(
                    source['dataset_name'], 
                    max_samples=source.get('sample_size')
                )
    
    # Create processing script
    manager.create_massive_processing_script(plan)
    
    # Generate and display report
    report = manager.generate_comprehensive_report(plan)
    logger.info(report)
    
    # Save setup information
    setup_info = {
        'sources': sources,
        'plan': plan,
        'setup_date': datetime.now().isoformat(),
        'system_specs': {
            'memory_gb': manager.total_memory_gb,
            'cpu_cores': manager.cpu_cores,
            'disk_gb': manager.available_disk_gb
        }
    }
    
    with open('data/massive_dataset_setup.json', 'w') as f:
        json.dump(setup_info, f, indent=2)
    
    logger.success("✅ Massive dataset setup complete!")
    
    if args.extract and not args.analyze_only:
        logger.info("📋 Next: Run 'python process_massive_dataset.py' to start cleaning")
    else:
        logger.info("📋 Next: Add --extract flag to extract data, then run cleaning")

if __name__ == "__main__":
    main() 