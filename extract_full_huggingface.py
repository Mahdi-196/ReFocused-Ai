#!/usr/bin/env python3
"""
Extract Full HuggingFace Dataset
Remove the 1M sample limit and process all ~8M records
"""

import json
import time
import psutil
from pathlib import Path
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

class FullHuggingFaceExtractor:
    """Extract the complete HuggingFace dataset without sample limits"""
    
    def __init__(self):
        self.base_dir = Path("data")
        self.unified_dir = self.base_dir / "unified_raw"
        self.logs_dir = Path("logs")
        
        # Create directories
        self.unified_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / f"full_huggingface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="100 MB")
        
        # Stats tracking
        self.stats = {
            'chunks_created': 0,
            'records_processed': 0,
            'start_time': time.time()
        }
    
    def extract_full_openwebtext(self, chunk_size_mb: int = 100):
        """Extract the complete OpenWebText dataset without limits"""
        logger.info("🤗 Starting FULL HuggingFace OpenWebText extraction...")
        logger.info("⚠️  This will process ~8M records (~36GB)")
        
        dataset_name = "Skylion007/openwebtext"
        output_files = []
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
        try:
            # Load dataset in streaming mode (no download limit)
            logger.info("📡 Loading HuggingFace dataset in streaming mode...")
            dataset = load_dataset(dataset_name, streaming=True, split="train", trust_remote_code=True)
            
            chunk_num = 1
            current_chunk_size = 0
            current_chunk_path = None
            current_chunk_file = None
            
            logger.info("🔄 Processing ALL HuggingFace examples (no limit)...")
            
            # Process ALL examples (no max_samples limit)
            for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
                
                # Start new chunk if needed
                if current_chunk_file is None or current_chunk_size >= chunk_size_bytes:
                    if current_chunk_file:
                        current_chunk_file.close()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_chunk_path = self.unified_dir / f"huggingface_full_chunk_{chunk_num:03d}_{timestamp}.txt"
                    current_chunk_file = open(current_chunk_path, 'w', encoding='utf-8')
                    output_files.append(current_chunk_path)
                    current_chunk_size = 0
                    chunk_num += 1
                    
                    if chunk_num % 20 == 1:  # Log every 20 chunks
                        logger.info(f"📝 Started full HF chunk {chunk_num-1}: {current_chunk_path.name}")
                
                # Convert to unified format and write
                text = example.get("text", "")
                if text.strip():  # Only process non-empty texts
                    unified_record = {
                        "id": f"hf_full_{i}",
                        "text": text,
                        "source": "huggingface_openwebtext_full",
                        "source_dataset": dataset_name,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    line = json.dumps(unified_record, ensure_ascii=False) + '\n'
                    current_chunk_file.write(line)
                    current_chunk_size += len(line.encode('utf-8'))
                    self.stats['records_processed'] += 1
                
                # Progress update every 50k
                if (i + 1) % 50000 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    rate = self.stats['records_processed'] / (elapsed / 3600)
                    
                    logger.info(f"📊 Processed {i+1:,} examples")
                    logger.info(f"   💾 Created {len(output_files)} chunks ({current_chunk_size/(1024*1024):.1f}MB current)")
                    logger.info(f"   ⚡ Rate: {rate:.0f} records/hour")
                    logger.info(f"   ⏱️  Elapsed: {elapsed/3600:.2f} hours")
                
                # Save progress every 500k records
                if (i + 1) % 500000 == 0:
                    logger.success(f"🎯 Milestone: {(i+1)/1000000:.1f}M records processed!")
            
            if current_chunk_file:
                current_chunk_file.close()
            
            self.stats['chunks_created'] = len(output_files)
            self.generate_extraction_report()
            
            return output_files
            
        except Exception as e:
            logger.error(f"❌ Full HuggingFace extraction failed: {e}")
            if current_chunk_file:
                current_chunk_file.close()
            return []
    
    def generate_extraction_report(self):
        """Generate extraction completion report"""
        elapsed = time.time() - self.stats['start_time']
        
        # Calculate output size
        hf_files = list(self.unified_dir.glob("huggingface_full_chunk_*.txt"))
        total_size = sum(f.stat().st_size for f in hf_files)
        total_size_gb = total_size / (1024**3)
        
        logger.success("\n🎉 FULL HUGGINGFACE EXTRACTION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {elapsed/3600:.2f} hours")
        logger.info(f"📊 Records extracted: {self.stats['records_processed']:,}")
        logger.info(f"📁 Chunks created: {self.stats['chunks_created']}")
        logger.info(f"💾 Total size: {total_size_gb:.2f}GB")
        logger.info(f"⚡ Extraction rate: {self.stats['records_processed']/(elapsed/3600):.0f} records/hour")
        
        logger.info(f"\n📁 Next Steps:")
        logger.info(f"   1. Run: python process_huggingface_only.py")
        logger.info(f"   2. This will clean and format the new data")
        logger.info(f"   3. Final dataset will be ~60GB+ total")

def main():
    """Main extraction function"""
    extractor = FullHuggingFaceExtractor()
    
    # System check
    memory = psutil.virtual_memory()
    logger.info(f"💻 System: {memory.total/(1024**3):.1f}GB RAM, {psutil.cpu_count()} cores")
    logger.info(f"💾 Available: {memory.available/(1024**3):.1f}GB RAM")
    
    # Disk space check
    import shutil
    free_space = shutil.disk_usage(".").free / (1024**3)
    logger.info(f"💿 Free disk space: {free_space:.1f}GB")
    
    if free_space < 100:
        logger.warning("⚠️  You may need more disk space for the full dataset")
    
    try:
        extractor.extract_full_openwebtext()
    except KeyboardInterrupt:
        logger.warning("⚠️  Extraction interrupted by user")
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        raise

if __name__ == "__main__":
    main() 