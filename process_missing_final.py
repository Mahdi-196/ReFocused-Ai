#!/usr/bin/env python3
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_missing_files():
    logger.info("🚀 Processing missing files...")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("models/tokenizer/transformers_tokenizer")
    
    # Read missing files
    with open('missing_files_simple.txt', 'r') as f:
        missing_files = [line.strip() for line in f.readlines()]
    
    logger.info(f"Found {len(missing_files)} missing files")
    
    output_dir = Path("data_tokenized_production")
    
    for file_path_str in tqdm(missing_files, desc="Processing"):
        file_path = Path(file_path_str)
        
        sequences = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        
                        if text and len(text.strip()) > 10:
                            tokens = tokenizer.encode(text, max_length=512, truncation=True, padding=False)
                            if len(tokens) > 5:
                                sequences.append(tokens)
                    except:
                        continue
            
            if sequences:
                output_name = f"tokenized_{file_path.name.replace('.jsonl', '_part001.npz')}"
                output_path = output_dir / output_name
                # Convert to object array to handle variable-length sequences
                sequences_array = np.array(sequences, dtype=object)
                np.savez_compressed(output_path, sequences=sequences_array)
                logger.info(f"✅ {file_path.name}: {len(sequences)} sequences")
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
    
    logger.info("🎉 Missing files processing complete!")

if __name__ == "__main__":
    process_missing_files() 