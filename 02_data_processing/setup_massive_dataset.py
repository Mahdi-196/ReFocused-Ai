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
                dataset = load_dataset(dataset_name, streaming=True, split="train", trust_remote_code=True)
            else:
                logger.info("💾 Downloading dataset locally (41GB)")
                dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
            
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
    
    def analyze_reddit_directory(self, directory_path: Path) -> Dict:
        """Analyze entire directory of Reddit .zst files"""
        logger.info(f"📱 Analyzing Reddit directory: {directory_path}")
        
        if not directory_path.exists():
            logger.error(f"❌ Directory not found: {directory_path}")
            return {}
        
        zst_files = list(directory_path.glob("*.zst"))
        if not zst_files:
            logger.error(f"❌ No .zst files found in {directory_path}")
            return {}
        
        total_size = sum(f.stat().st_size for f in zst_files)
        total_size_gb = total_size / (1024**3)
        
        # Analyze subreddits
        subreddits = {}
        for file in zst_files:
            name_parts = file.stem.split('_')
            if len(name_parts) >= 2:
                subreddit = name_parts[0]
                file_type = name_parts[1] if len(name_parts) > 1 else 'unknown'
                
                if subreddit not in subreddits:
                    subreddits[subreddit] = {'files': 0, 'size_mb': 0}
                
                subreddits[subreddit]['files'] += 1
                subreddits[subreddit]['size_mb'] += file.stat().st_size / (1024**2)
        
        # Estimate records (typical Reddit data has ~1000-5000 records per MB compressed)
        estimated_records = int(total_size / 500)  # Conservative estimate
        
        analysis = {
            'directory_path': str(directory_path),
            'total_files': len(zst_files),
            'size_gb': total_size_gb,
            'estimated_records': estimated_records,
            'unique_subreddits': len(subreddits),
            'subreddits_sample': dict(list(subreddits.items())[:10]),  # Top 10 for display
            'source_type': 'reddit_directory',
            'format': 'ZST_compressed'
        }
        
        logger.success(f"✅ Reddit directory analysis:")
        logger.info(f"   📁 Files: {len(zst_files)} .zst files")
        logger.info(f"   📊 Size: {total_size_gb:.1f}GB")
        logger.info(f"   🏷️  Subreddits: {len(subreddits)} unique")
        logger.info(f"   📈 Est. records: {estimated_records:,}")
        
        return analysis

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
                                max_samples: Optional[int] = 1000000) -> List[Path]:
        """Extract HuggingFace data into chunks"""
        logger.info(f"🤗 Extracting HuggingFace data: {dataset_name}")
        
        output_files = []
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
        try:
            # Use streaming to avoid downloading entire dataset
            dataset = load_dataset(dataset_name, streaming=True, split="train", trust_remote_code=True)
            
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
    
    def extract_reddit_directory(self, directory_path: Path):
        """Extract all Reddit .zst files from a directory"""
        logger.info(f"📁 Extracting Reddit directory: {directory_path}")
        
        if not directory_path.exists():
            logger.error(f"❌ Directory not found: {directory_path}")
            return
        
        zst_files = list(directory_path.glob("*.zst"))
        if not zst_files:
            logger.error(f"❌ No .zst files found in {directory_path}")
            return
        
        # Create output directory
        output_dir = Path(self.base_dir) / "unified_raw"
        output_dir.mkdir(exist_ok=True)
        
        total_files = len(zst_files)
        logger.info(f"📊 Found {total_files} .zst files to process")
        
        processed = 0
        for zst_file in zst_files:
            try:
                logger.info(f"📄 Processing {zst_file.name} ({processed+1}/{total_files})")
                self._extract_single_zst_file(zst_file, output_dir)
                processed += 1
                logger.success(f"✅ Completed {zst_file.name}")
            except Exception as e:
                logger.error(f"❌ Failed to process {zst_file.name}: {e}")
                continue
        
        logger.success(f"🎉 Extracted {processed}/{total_files} Reddit files to {output_dir}")
    
    def _extract_single_zst_file(self, zst_file: Path, output_dir: Path):
        """Extract a single .zst file to JSONL format"""
        import zstandard as zstd
        
        # Create output filename
        base_name = zst_file.stem  # removes .zst extension
        output_file = output_dir / f"{base_name}.jsonl"
        
        posts_written = 0
        
        with open(zst_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    text_stream = reader.read().decode('utf-8', errors='ignore')
                    
                    for line in text_stream.split('\n'):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                # Normalize the data format
                                normalized = self._normalize_reddit_post(data, base_name)
                                if normalized:
                                    out_f.write(json.dumps(normalized) + '\n')
                                    posts_written += 1
                            except json.JSONDecodeError:
                                continue
        
        logger.info(f"   💾 Wrote {posts_written:,} posts to {output_file.name}")
    
    def _normalize_reddit_post(self, data: Dict, filename: str) -> Optional[Dict]:
        """Normalize Reddit post data to unified format"""
        try:
            # Extract subreddit from filename
            if '_comments' in filename:
                subreddit = filename.replace('_comments', '')
                post_type = 'comment'
            elif '_submissions' in filename:
                subreddit = filename.replace('_submissions', '')
                post_type = 'submission'
            else:
                subreddit = filename
                post_type = 'unknown'
            
            # Create normalized format
            normalized = {
                'id': data.get('id', ''),
                'subreddit': subreddit,
                'type': post_type,
                'author': data.get('author', '[deleted]'),
                'created_utc': data.get('created_utc', 0),
                'score': data.get('score', 0),
                'text': '',
                'title': data.get('title', ''),
                'url': data.get('url', ''),
                'num_comments': data.get('num_comments', 0)
            }
            
            # Handle text content
            if post_type == 'comment':
                normalized['text'] = data.get('body', '')
            else:
                normalized['text'] = data.get('selftext', '')
            
            # Skip empty or deleted content
            if not normalized['text'] or normalized['text'] in ['[deleted]', '[removed]']:
                return None
            
            # Skip very short content
            if len(normalized['text']) < 10:
                return None
            
            return normalized
            
        except Exception as e:
            logger.debug(f"Error normalizing post: {e}")
            return None
    
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
   2. Extract HuggingFace: python setup_massive_dataset.py --reddit-dir subreddits24 --huggingface --extract
   3. Process: python process_massive_dataset.py
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

    def show_unified_analysis(self, analysis: Dict):
        """Display unified analysis of all data sources"""
        logger.info("📋 UNIFIED DATASET ANALYSIS")
        logger.info("=" * 50)
        
        total_size_gb = 0
        total_records = 0
        sources = []
        
        if 'reddit' in analysis and analysis['reddit']:
            reddit = analysis['reddit']
            total_size_gb += reddit['size_gb']
            total_records += reddit['estimated_records']
            sources.append(f"Reddit: {reddit['total_files']} files, {reddit['size_gb']:.1f}GB")
            
            logger.info(f"📱 REDDIT DATA:")
            logger.info(f"   Files: {reddit['total_files']} .zst files")
            logger.info(f"   Size: {reddit['size_gb']:.1f}GB")
            logger.info(f"   Subreddits: {reddit['unique_subreddits']} unique")
            logger.info(f"   Est. Records: {reddit['estimated_records']:,}")
        
        if 'huggingface' in analysis and analysis['huggingface']:
            hf = analysis['huggingface']
            total_size_gb += hf.get('estimated_size_gb', 0)
            total_records += hf.get('estimated_records', 0)
            sources.append(f"HuggingFace: {hf.get('estimated_records', 0):,} records")
            
            logger.info(f"🤗 HUGGINGFACE DATA:")
            logger.info(f"   Dataset: {hf.get('dataset_name', 'Unknown')}")
            logger.info(f"   Est. Size: {hf.get('estimated_size_gb', 0):.1f}GB")
            logger.info(f"   Est. Records: {hf.get('estimated_records', 0):,}")
        
        # Processing estimates
        logger.info(f"\n⚡ PROCESSING ESTIMATES:")
        logger.info(f"   Total Input: {total_size_gb:.1f}GB")
        logger.info(f"   Est. Records: {total_records:,}")
        
        # Time estimates based on system specs
        extraction_time = max(1, total_size_gb * 0.5)  # ~30 minutes per GB
        processing_time = max(2, total_records / 50000)  # ~50k records per hour
        
        logger.info(f"   Extraction Time: {extraction_time:.1f} hours")
        logger.info(f"   Processing Time: {processing_time:.1f} hours")
        logger.info(f"   Total Time: {extraction_time + processing_time:.1f} hours")
        
        # Output estimates
        clean_size_gb = total_size_gb * 0.3  # ~30% of input size after cleaning
        logger.info(f"   Est. Clean Output: {clean_size_gb:.1f}GB")
        
        logger.info(f"\n🚀 NEXT STEPS:")
        if 'reddit' in analysis:
            logger.info(f"   1. Extract: python setup_massive_dataset.py --reddit-dir subreddits24 --extract")
        if 'huggingface' in analysis:
            logger.info(f"   2. Include HF: python setup_massive_dataset.py --reddit-dir subreddits24 --huggingface --extract")
        logger.info(f"   3. Process: python process_massive_dataset.py")
        logger.info(f"   4. Monitor: python monitor_reddit_pipeline.py")

def main():
    """Main setup function for massive datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup massive multi-source dataset for cleaning")
    parser.add_argument('--reddit', type=str, help='Path to Reddit dataset file')
    parser.add_argument('--reddit-dir', type=str, help='Path to directory containing Reddit .zst files')
    parser.add_argument('--huggingface', action='store_true', help='Include HuggingFace OpenWebText dataset')
    parser.add_argument('--extract', action='store_true', help='Extract and unify datasets')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze datasets without processing')

    args = parser.parse_args()
    
    # Initialize manager
    manager = MassiveDatasetManager()
    
    reddit_path = None
    if args.reddit:
        reddit_path = Path(args.reddit)
    elif args.reddit_dir:
        reddit_path = Path(args.reddit_dir)
    
    if args.analyze_only:
        logger.info("🔍 ANALYSIS MODE - No data will be extracted")
        analysis = {}
        
        if reddit_path:
            if reddit_path.is_dir():
                analysis['reddit'] = manager.analyze_reddit_directory(reddit_path)
            else:
                analysis['reddit'] = manager.analyze_reddit_dataset(reddit_path)
        
        if args.huggingface:
            analysis['huggingface'] = manager.setup_huggingface_dataset()
        
        manager.show_unified_analysis(analysis)
        return
    
    if args.extract:
        if reddit_path:
            if reddit_path.is_dir():
                logger.info(f"📁 Processing Reddit directory: {reddit_path}")
                manager.extract_reddit_directory(reddit_path)
            else:
                logger.info(f"📄 Processing Reddit file: {reddit_path}")
                manager.extract_reddit_dataset(reddit_path)
        
        if args.huggingface:
            manager.extract_huggingface_data("Skylion007/openwebtext")

if __name__ == "__main__":
    main() 