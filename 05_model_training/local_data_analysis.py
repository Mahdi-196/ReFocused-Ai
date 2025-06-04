#!/usr/bin/env python3
"""
Local Data Analysis Script - Analyze local data or provide general recommendations
"""

import os
import numpy as np
import json
import time
import glob
import warnings
warnings.filterwarnings("ignore")

def find_local_data():
    """Find local training data files"""
    print("🔍 SEARCHING FOR LOCAL TRAINING DATA")
    print("=" * 40)
    
    # Search patterns for data files
    search_patterns = [
        "../data/**/*.npz",
        "../02_data_processing/data/**/*.npz", 
        "../04_data_tokenization/**/*.npz",
        "./data/**/*.npz",
        "../../data/**/*.npz"
    ]
    
    all_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    # Filter for tokenized/processed files
    tokenized_files = []
    for file in all_files:
        filename = os.path.basename(file).lower()
        if any(pattern in filename for pattern in ['tokenized', 'cleaned', 'processed']):
            tokenized_files.append(file)
    
    print(f"📁 Found {len(all_files)} total .npz files")
    print(f"🎯 Found {len(tokenized_files)} tokenized files")
    
    return tokenized_files

def calculate_recommendations(total_tokens):
    """Calculate recommendations based on token count"""
    # Model parameters
    sequence_length = 1024
    per_device_batch_size = 4
    gradient_accumulation_steps = 4
    gpus = 2
    
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * gpus
    tokens_per_step = effective_batch_size * sequence_length
    
    # For large datasets, we don't need to see every token multiple times
    # 1 epoch is often enough for very large datasets
    if total_tokens > 10e9:  # > 10B tokens
        target_epochs = 0.5  # See half the data once
    elif total_tokens > 1e9:  # > 1B tokens  
        target_epochs = 1.0  # See all data once
    else:
        target_epochs = 2.5  # See smaller data multiple times
    
    # Calculate steps based on target epochs
    one_epoch_steps = total_tokens // tokens_per_step
    recommended_steps = int(one_epoch_steps * target_epochs)
    
    # Cap at reasonable maximum (50k steps = ~28 hours on 2 GPUs)
    max_reasonable_steps = 50000
    if recommended_steps > max_reasonable_steps:
        recommended_steps = max_reasonable_steps
        actual_epochs = recommended_steps / one_epoch_steps
        print(f"⚠️  Dataset is very large! Capping at {max_reasonable_steps:,} steps")
        print(f"   This will process {actual_epochs:.2f} epochs of your data")
    
    # Round to nice intervals
    checkpoint_interval = max(100, recommended_steps // 20)
    recommended_steps = (recommended_steps // checkpoint_interval) * checkpoint_interval
    
    save_steps = max(100, recommended_steps // 15)
    logging_steps = max(10, recommended_steps // 100)
    estimated_hours = recommended_steps / 5.0 / 3600
    
    return {
        'recommended_steps': recommended_steps,
        'save_steps': save_steps,
        'logging_steps': logging_steps,
        'estimated_hours': estimated_hours,
        'tokens_per_step': tokens_per_step,
        'target_epochs': target_epochs if recommended_steps < max_reasonable_steps else actual_epochs,
        'one_epoch_steps': one_epoch_steps
    }

def main():
    """Main function"""
    print("🚀 REFOCUSED-AI TRAINING ANALYSIS")
    print("=" * 60)
    
    # Try local data first
    local_files = find_local_data()
    
    if local_files:
        print(f"\n✅ Analyzing {len(local_files)} local files...")
        
        total_tokens = 0
        valid_files = 0
        
        for file_path in local_files[:10]:  # Sample first 10
            try:
                data = np.load(file_path)
                if 'input_ids' in data:
                    tokens = len(data['input_ids'].reshape(-1))
                    total_tokens += tokens
                    valid_files += 1
                data.close()
            except:
                continue
        
        if valid_files > 0:
            # Extrapolate
            avg_tokens = total_tokens / valid_files
            estimated_total = int(avg_tokens * len(local_files))
            
            recommendations = calculate_recommendations(estimated_total)
            
            print(f"\n📊 LOCAL DATA ANALYSIS:")
            print(f"   Files: {len(local_files):,}")
            print(f"   Estimated tokens: {estimated_total:,}")
            print(f"   One epoch would be: {recommendations['one_epoch_steps']:,} steps")
            print(f"\n🎯 RECOMMENDATIONS (2 GPUs):")
            print(f"   max_steps = {recommendations['recommended_steps']:,}")
            print(f"   save_steps = {recommendations['save_steps']}")
            print(f"   logging_steps = {recommendations['logging_steps']}")
            print(f"   Target epochs: {recommendations['target_epochs']:.2f}")
            print(f"   Estimated time: {recommendations['estimated_hours']:.1f} hours")
        else:
            print("❌ No valid local files found")
    
    # Show general recommendations
    print(f"\n🎯 GENERAL RECOMMENDATIONS BY DATASET SIZE:")
    print("=" * 50)
    
    recommendations = {
        "Small (< 1B tokens)": {"steps": 5000, "save": 500, "log": 50},
        "Medium (1-10B tokens)": {"steps": 15000, "save": 1000, "log": 150},  
        "Large (10B+ tokens)": {"steps": 50000, "save": 2500, "log": 500}
    }
    
    for size, rec in recommendations.items():
        print(f"\n📊 {size}:")
        print(f"   max_steps = {rec['steps']:,}")
        print(f"   save_steps = {rec['save']}")
        print(f"   logging_steps = {rec['log']}")
    
    print(f"\n💡 FOR BUCKET ANALYSIS:")
    print("1. Authenticate: gcloud auth application-default login")
    print("2. Run: python analyze_training_parameters.py") 
    print("3. Or run: ./run_analysis.sh")

if __name__ == "__main__":
    main() 