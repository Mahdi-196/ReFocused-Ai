import gzip
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
from datetime import datetime
from loguru import logger

class LargeDatasetManager:
    """Manages large compressed Reddit datasets for cleaning"""
    
    def __init__(self, input_path: str = None, extract_dir: str = "data/reddit_large"):
        self.input_path = Path(input_path) if input_path else None
        self.extract_dir = Path(extract_dir)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Create other necessary directories
        (Path("data/cleaned")).mkdir(parents=True, exist_ok=True)
        (Path("data/processed")).mkdir(parents=True, exist_ok=True)
        (Path("logs")).mkdir(parents=True, exist_ok=True)
    
    def analyze_compressed_file(self, file_path: Path) -> Dict:
        """Analyze compressed file to understand its structure"""
        logger.info(f"🔍 Analyzing compressed file: {file_path}")
        
        try:
            file_size = file_path.stat().st_size / (1024**3)  # GB
            logger.info(f"📏 File size: {file_size:.2f} GB")
            
            # Try to peek at the first few lines
            sample_lines = []
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if i >= 10:  # Sample first 10 lines
                            break
                        sample_lines.append(line.strip())
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if i >= 10:
                            break
                        sample_lines.append(line.strip())
            
            # Analyze format
            json_lines = 0
            total_sample = len(sample_lines)
            
            for line in sample_lines:
                if line:
                    try:
                        json.loads(line)
                        json_lines += 1
                    except json.JSONDecodeError:
                        continue
            
            format_type = "JSONL" if json_lines > total_sample * 0.8 else "Unknown"
            
            analysis = {
                'file_path': str(file_path),
                'size_gb': file_size,
                'format': format_type,
                'sample_lines': len(sample_lines),
                'valid_json_lines': json_lines,
                'estimated_records': int((json_lines / max(total_sample, 1)) * file_size * 1000000),  # Rough estimate
                'compressed': file_path.suffix == '.gz'
            }
            
            logger.success(f"✅ Analysis complete: {format_type} format, ~{analysis['estimated_records']:,} estimated records")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing file: {e}")
            return {}
    
    def extract_compressed_data(self, source_file: Path, chunk_size_mb: int = 100) -> List[Path]:
        """Extract compressed data into manageable chunks"""
        logger.info(f"📦 Extracting compressed data from {source_file}")
        
        output_files = []
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
        try:
            if source_file.suffix == '.gz':
                with gzip.open(source_file, 'rt', encoding='utf-8', errors='ignore') as f:
                    self._extract_in_chunks(f, output_files, chunk_size_bytes)
            else:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    self._extract_in_chunks(f, output_files, chunk_size_bytes)
            
            logger.success(f"✅ Extracted to {len(output_files)} chunk files")
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return []
    
    def _extract_in_chunks(self, file_obj, output_files: List[Path], chunk_size_bytes: int):
        """Helper to extract file in chunks"""
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
                    current_chunk_path = self.extract_dir / f"chunk_{chunk_num:03d}_{timestamp}.txt"
                    current_chunk_file = open(current_chunk_path, 'w', encoding='utf-8')
                    output_files.append(current_chunk_path)
                    current_chunk_size = 0
                    chunk_num += 1
                    
                    logger.info(f"📝 Started chunk {chunk_num-1}: {current_chunk_path.name}")
                
                # Write line to current chunk
                current_chunk_file.write(line)
                current_chunk_size += len(line.encode('utf-8'))
                
                # Progress update
                if line_num % 100000 == 0:
                    logger.info(f"📊 Processed {line_num:,} lines, created {len(output_files)} chunks")
        
        finally:
            if current_chunk_file:
                current_chunk_file.close()
    
    def setup_cleaning_config(self, estimated_records: int) -> Dict:
        """Configure cleaning settings for large dataset"""
        
        # Adjust batch sizes based on dataset size
        if estimated_records > 1000000:  # 1M+ records
            batch_size = 5000
            n_workers = max(2, min(8, estimated_records // 100000))
        elif estimated_records > 500000:  # 500K+ records
            batch_size = 3000
            n_workers = 4
        else:
            batch_size = 2000
            n_workers = 2
        
        config = {
            'batch_size': batch_size,
            'n_workers': n_workers,
            'estimated_records': estimated_records,
            'memory_efficient': True,
            'progress_interval': 1000,
            'save_intermediate': True
        }
        
        logger.info(f"🔧 Configured for {estimated_records:,} records: batch_size={batch_size}, workers={n_workers}")
        return config
    
    def update_data_cleaner_for_large_dataset(self, config: Dict):
        """Update data_cleaner.py for optimal large dataset processing"""
        logger.info("🔧 Optimizing data_cleaner.py for large dataset...")
        
        # Read current cleaner
        cleaner_path = Path("data_cleaner.py")
        if not cleaner_path.exists():
            logger.error("❌ data_cleaner.py not found")
            return
        
        content = cleaner_path.read_text()
        
        # Update default parameters for large dataset
        updates = [
            ('batch_size: int = 1000', f'batch_size: int = {config["batch_size"]}'),
            ('n_workers: int = None', f'n_workers: int = {config["n_workers"]}'),
        ]
        
        for old, new in updates:
            if old in content:
                content = content.replace(old, new)
                logger.info(f"✅ Updated: {old} → {new}")
        
        # Add large dataset directories
        if 'reddit_large' not in content:
            # Find the data_dirs definition and add reddit_large
            data_dirs_start = content.find('data_dirs = [')
            if data_dirs_start != -1:
                insert_pos = content.find(']', data_dirs_start)
                if insert_pos != -1:
                    new_dir = ',\n            self.data_dir / "reddit_large"'
                    content = content[:insert_pos] + new_dir + content[insert_pos:]
                    logger.info("✅ Added reddit_large directory to data sources")
        
        # Write updated cleaner
        cleaner_path.write_text(content)
        logger.success("✅ data_cleaner.py optimized for large dataset")
    
    def create_processing_script(self) -> Path:
        """Create optimized processing script for the large dataset"""
        script_path = Path("process_large_dataset.py")
        
        script_content = '''#!/usr/bin/env python3
"""
Large Dataset Processing Script
Optimized for 13GB+ Reddit datasets
"""

import sys
from pathlib import Path
from loguru import logger
from data_cleaner import RedditDataCleaner
import time

def main():
    """Process large dataset with monitoring"""
    logger.info("🚀 Starting large dataset processing...")
    
    # Initialize with optimized settings
    cleaner = RedditDataCleaner(
        data_dir="data",
        output_dir="data/cleaned",
        n_workers=4  # Adjust based on your CPU
    )
    
    # Monitor memory usage
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.info(f"💾 Initial memory usage: {initial_memory:.1f} MB")
    
    # Run cleaning with progress monitoring
    start_time = time.time()
    
    try:
        cleaned_posts = cleaner.run_cleaning_pipeline(batch_size=5000)
        
        # Final statistics
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024
        processing_time = end_time - start_time
        
        logger.success(f"✅ Processing complete!")
        logger.info(f"⏱️  Total time: {processing_time/60:.1f} minutes")
        logger.info(f"💾 Peak memory: {final_memory:.1f} MB")
        logger.info(f"📊 Cleaned posts: {len(cleaned_posts):,}")
        logger.info(f"⚡ Processing rate: {len(cleaned_posts)/(processing_time/3600):.0f} posts/hour")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_path.write_text(script_content)
        logger.success(f"✅ Created optimized processing script: {script_path}")
        return script_path
    
    def generate_setup_report(self, analysis: Dict, config: Dict) -> str:
        """Generate setup report for the large dataset"""
        
        if not analysis:
            return "❌ No analysis data available"
        
        report = f"""
🗂️  LARGE DATASET SETUP REPORT
{'='*50}

📊 DATASET ANALYSIS:
   File: {analysis.get('file_path', 'Unknown')}
   Size: {analysis.get('size_gb', 0):.2f} GB
   Format: {analysis.get('format', 'Unknown')}
   Estimated Records: {analysis.get('estimated_records', 0):,}
   Compressed: {analysis.get('compressed', False)}

🔧 OPTIMIZATION CONFIG:
   Batch Size: {config.get('batch_size', 2000):,} posts
   Workers: {config.get('n_workers', 2)} parallel processes
   Memory Efficient: {config.get('memory_efficient', True)}

📁 DIRECTORY STRUCTURE:
   ✅ data/reddit_large/     - Extracted chunks
   ✅ data/cleaned/          - Cleaned output
   ✅ data/processed/        - Training splits

🚀 NEXT STEPS:
   1. Place your 13GB file in data/reddit_large/
   2. Run: python setup_large_dataset.py --extract YOUR_FILE
   3. Run: python process_large_dataset.py
   4. Monitor: python cleanup_summary.py

⚡ ESTIMATED PROCESSING:
   Time: {analysis.get('estimated_records', 0) // 10000:.0f}-{analysis.get('estimated_records', 0) // 5000:.0f} hours
   Output: {int(analysis.get('estimated_records', 0) * 0.65):,} clean posts (65% retention)
   Size: {analysis.get('size_gb', 0) * 0.3:.1f}-{analysis.get('size_gb', 0) * 0.5:.1f} GB cleaned
"""
        return report

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup large Reddit dataset for cleaning")
    parser.add_argument('--file', type=str, help='Path to compressed Reddit data file')
    parser.add_argument('--extract', action='store_true', help='Extract compressed file into chunks')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze file, don\'t extract')
    
    args = parser.parse_args()
    
    manager = LargeDatasetManager()
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return
        
        # Analyze the file
        analysis = manager.analyze_compressed_file(file_path)
        if not analysis:
            return
        
        # Create configuration
        config = manager.setup_cleaning_config(analysis.get('estimated_records', 0))
        
        # Update cleaner for large dataset
        manager.update_data_cleaner_for_large_dataset(config)
        
        # Extract if requested
        if args.extract and not args.analyze_only:
            output_files = manager.extract_compressed_data(file_path)
            logger.info(f"📝 Extracted {len(output_files)} chunk files")
        
        # Create processing script
        manager.create_processing_script()
        
        # Generate report
        report = manager.generate_setup_report(analysis, config)
        logger.info(report)
        
        # Save setup info
        setup_info = {
            'analysis': analysis,
            'config': config,
            'setup_date': datetime.now().isoformat()
        }
        
        import json
        with open('data/large_dataset_setup.json', 'w') as f:
            json.dump(setup_info, f, indent=2)
        
        logger.success("✅ Large dataset setup complete!")
        logger.info("📋 Next: Run 'python process_large_dataset.py' to start cleaning")
    
    else:
        logger.info("💡 Usage: python setup_large_dataset.py --file YOUR_13GB_FILE.gz --extract")
        logger.info("📁 Make sure to place your file in an accessible location first")

if __name__ == "__main__":
    main() 