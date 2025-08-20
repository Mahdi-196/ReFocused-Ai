#!/usr/bin/env python3
"""
Training Parameter Analysis Script for ReFocused-AI

This script analyzes your GCS bucket data and recommends:
1. Optimal number of training steps
2. Ideal hyperparameters for your model size
3. Data statistics and training estimates
"""

import os
import sys
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account
from tqdm import tqdm
import json
import time
import warnings
warnings.filterwarnings("ignore")

def analyze_bucket_data(bucket_name="refocused-ai", max_sample_files=30, credentials_path: str | None = None, project_id: str | None = None):
    """Analyze bucket data and return statistics"""
    print(f"🚀 REFOCUSED-AI TRAINING PARAMETER ANALYZER")
    print("=" * 60)
    print(f"🔍 Analyzing bucket: gs://{bucket_name}")
    
    try:
        if credentials_path and os.path.exists(credentials_path):
            creds = service_account.Credentials.from_service_account_file(credentials_path)
            client = storage.Client(project=project_id, credentials=creds)
        else:
            client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        print("📊 Scanning for tokenized files...")
        blobs = list(bucket.list_blobs())
        
        # Find tokenized files
        tokenized_files = []
        for blob in blobs:
            if blob.name.endswith('.npz'):
                if any(pattern in blob.name.lower() for pattern in ['tokenized', 'cleaned', 'processed']):
                    tokenized_files.append(blob)
        
        print(f"📁 Found {len(tokenized_files)} tokenized files")
        
        if len(tokenized_files) == 0:
            print("❌ No tokenized files found!")
            return None
        
        # Sample files for analysis
        sample_size = min(max_sample_files, len(tokenized_files))
        sample_files = tokenized_files[:sample_size]
        
        print(f"🔬 Analyzing {sample_size} sample files...")
        
        valid_files = 0
        corrupted_files = 0
        total_tokens_sampled = 0
        file_sizes_mb = []
        
        # Create cache directory
        os.makedirs("./cache", exist_ok=True)
        
        for i, blob in enumerate(tqdm(sample_files, desc="Analyzing")):
            try:
                # Get file size
                file_size_mb = blob.size / (1024 * 1024)
                file_sizes_mb.append(file_size_mb)
                
                # Download and analyze
                local_path = f"./cache/analysis_sample_{i}.npz"
                blob.download_to_filename(local_path)
                
                # Load data
                data = np.load(local_path)
                
                if 'input_ids' not in data:
                    print(f"   Missing 'input_ids' in {blob.name}")
                    corrupted_files += 1
                    continue
                
                input_ids = data['input_ids']
                
                # Handle different shapes
                if input_ids.ndim > 1:
                    input_ids = input_ids.reshape(-1)
                
                num_tokens = len(input_ids)
                if num_tokens == 0:
                    corrupted_files += 1
                    continue
                
                total_tokens_sampled += num_tokens
                valid_files += 1
                
                data.close()
                
                # Clean up
                if os.path.exists(local_path):
                    os.remove(local_path)
                    
            except Exception as e:
                print(f"   Error with {blob.name}: {e}")
                corrupted_files += 1
                continue
        
        # Calculate statistics
        if valid_files == 0:
            print("❌ No valid files found in sample!")
            return None
        
        # Extrapolate to full dataset
        avg_tokens_per_file = total_tokens_sampled / valid_files
        corruption_rate = corrupted_files / len(sample_files)
        estimated_valid_files = int(len(tokenized_files) * (1 - corruption_rate))
        estimated_total_tokens = int(avg_tokens_per_file * estimated_valid_files)
        avg_file_size_mb = np.mean(file_sizes_mb)
        estimated_dataset_size_gb = (avg_file_size_mb * len(tokenized_files)) / 1024
        
        stats = {
            'total_files': len(tokenized_files),
            'valid_files': estimated_valid_files,
            'corrupted_files': len(tokenized_files) - estimated_valid_files,
            'total_tokens': estimated_total_tokens,
            'avg_tokens_per_file': avg_tokens_per_file,
            'avg_file_size_mb': avg_file_size_mb,
            'estimated_dataset_size_gb': estimated_dataset_size_gb,
            'corruption_rate': corruption_rate
        }
        
        print_data_statistics(stats)
        return stats
        
    except Exception as e:
        print(f"❌ Error accessing bucket: {e}")
        return None

def print_data_statistics(stats):
    """Print comprehensive data statistics"""
    print(f"\n📈 DATA ANALYSIS RESULTS")
    print("=" * 50)
    print(f"📁 Total tokenized files: {stats['total_files']:,}")
    print(f"✅ Estimated valid files: {stats['valid_files']:,}")
    print(f"⚠️  Estimated corrupted files: {stats['corrupted_files']:,}")
    print(f"📊 Corruption rate: {stats['corruption_rate']*100:.1f}%")
    print(f"\n🔢 TOKEN STATISTICS:")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Avg tokens per file: {stats['avg_tokens_per_file']:,.0f}")
    print(f"\n💾 STORAGE STATISTICS:")
    print(f"   Dataset size: {stats['estimated_dataset_size_gb']:.2f} GB")
    print(f"   Avg file size: {stats['avg_file_size_mb']:.1f} MB")

def calculate_training_recommendations(stats, target_gpus=2):
    """Calculate optimal training parameters"""
    print(f"\n🎯 TRAINING RECOMMENDATIONS ({target_gpus} GPUs)")
    print("=" * 50)
    
    # Model parameters
    model_params = 1.2e9  # 1.2B parameters
    sequence_length = 1024
    
    # Hardware-specific batch sizes
    if target_gpus == 1:
        per_device_batch_size = 4
        gradient_accumulation_steps = 4
        estimated_steps_per_second = 2.5
    elif target_gpus == 2:
        per_device_batch_size = 4
        gradient_accumulation_steps = 4
        estimated_steps_per_second = 5.0
    elif target_gpus >= 4:
        per_device_batch_size = 6
        gradient_accumulation_steps = 4
        estimated_steps_per_second = target_gpus * 2.0
    
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * target_gpus
    tokens_per_step = effective_batch_size * sequence_length
    
    # Calculate optimal training steps
    total_tokens = stats['total_tokens']
    
    # Conservative: 1 epoch
    conservative_steps = total_tokens // tokens_per_step
    
    # Optimal: ~2.5 epochs for good convergence
    optimal_epochs = 2.5
    recommended_steps = int(conservative_steps * optimal_epochs)
    
    # Round to nice checkpoint intervals
    checkpoint_interval = max(100, recommended_steps // 20)
    recommended_steps = (recommended_steps // checkpoint_interval) * checkpoint_interval
    
    # Calculate save and logging frequencies
    save_steps = max(100, recommended_steps // 15)  # ~15 checkpoints
    logging_steps = max(10, recommended_steps // 100)  # ~100 log points
    
    # Calculate training time estimate
    estimated_training_time_hours = recommended_steps / estimated_steps_per_second / 3600
    
    recommendations = {
        'recommended_steps': recommended_steps,
        'optimal_epochs': optimal_epochs,
        'effective_batch_size': effective_batch_size,
        'per_device_batch_size': per_device_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'save_steps': save_steps,
        'logging_steps': logging_steps,
        'estimated_training_time_hours': estimated_training_time_hours,
        'tokens_per_step': tokens_per_step,
        'data_utilization': (recommended_steps * tokens_per_step) / total_tokens
    }
    
    print_training_recommendations(recommendations, target_gpus)
    return recommendations

def print_training_recommendations(rec, target_gpus):
    """Print training recommendations"""
    print(f"🎯 OPTIMAL TRAINING STEPS: {rec['recommended_steps']:,}")
    print(f"   Optimal epochs: {rec['optimal_epochs']:.1f}")
    print(f"   Total tokens processed: {rec['recommended_steps'] * rec['tokens_per_step']:,}")
    print(f"   Data utilization: {rec['data_utilization']:.1f}x")
    
    print(f"\n⚙️  BATCH SIZE CONFIGURATION:")
    print(f"   Per-device batch size: {rec['per_device_batch_size']}")
    print(f"   Gradient accumulation: {rec['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {rec['effective_batch_size']}")
    print(f"   Tokens per step: {rec['tokens_per_step']:,}")
    
    print(f"\n📊 CHECKPOINT SCHEDULE:")
    print(f"   Save every: {rec['save_steps']} steps")
    print(f"   Log every: {rec['logging_steps']} steps")
    print(f"   Total checkpoints: {rec['recommended_steps'] // rec['save_steps']}")
    
    print(f"\n⏰ TIME ESTIMATES:")
    print(f"   Training time: {rec['estimated_training_time_hours']:.1f} hours")
    print(f"   Training time: {rec['estimated_training_time_hours'] * 60:.0f} minutes")

def generate_hardware_comparison(stats):
    """Generate comparison across different GPU configurations"""
    print(f"\n🎮 MULTI-GPU PERFORMANCE COMPARISON")
    print("=" * 50)
    
    gpu_configs = [1, 2, 4, 8]
    all_recommendations = {}
    
    for gpu_count in gpu_configs:
        print(f"\n--- {gpu_count} GPU Configuration ---")
        recommendations = calculate_training_recommendations(stats, gpu_count)
        all_recommendations[f"{gpu_count}_gpu"] = recommendations
    
    return all_recommendations

def main():
    """Main analysis function"""
    print("Starting bucket analysis...")
    
    try:
        # Analyze bucket data
        # Accept optional CLI args for credentials
        cred_path = None
        project_id = None
        if len(sys.argv) > 1:
            # naive parse: --gcs-credentials <path> --gcp-project <id>
            args = sys.argv[1:]
            for i in range(len(args)):
                if args[i] == "--gcs-credentials" and i + 1 < len(args):
                    cred_path = args[i+1]
                if args[i] == "--gcp-project" and i + 1 < len(args):
                    project_id = args[i+1]

        stats = analyze_bucket_data(credentials_path=cred_path, project_id=project_id)
        
        if stats is None:
            print("❌ Failed to analyze bucket data")
            return 1
        
        # Get recommendations for different GPU configurations
        all_recommendations = generate_hardware_comparison(stats)
        
        # Use 2 GPU as default
        default_recommendations = all_recommendations["2_gpu"]
        
        # Final summary
        print(f"\n🎯 QUICK SUMMARY FOR YOUR MODEL")
        print("=" * 50)
        print(f"📊 Your dataset: {stats['total_tokens']:,} tokens in {stats['valid_files']:,} files")
        print(f"🎯 Recommended steps (2 GPUs): {default_recommendations['recommended_steps']:,}")
        print(f"⏰ Estimated time (2 GPUs): {default_recommendations['estimated_training_time_hours']:.1f} hours")
        print(f"💾 Checkpoints: Every {default_recommendations['save_steps']} steps")
        print(f"📝 Logs: Every {default_recommendations['logging_steps']} steps")
        
        print(f"\n🚀 CONFIGURATION VALUES TO UPDATE:")
        print("=" * 40)
        print(f"max_steps = {default_recommendations['recommended_steps']}")
        print(f"save_steps = {default_recommendations['save_steps']}")
        print(f"logging_steps = {default_recommendations['logging_steps']}")
        
        # Save comprehensive report
        report = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "bucket_name": "refocused-ai",
            "data_statistics": stats,
            "training_recommendations": default_recommendations,
            "gpu_comparisons": all_recommendations
        }
        
        with open("training_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Complete analysis report saved to: training_analysis_report.json")
        print(f"🎉 Analysis complete! You can now update your training configuration.")
        
        # Clean up cache
        if os.path.exists("./cache"):
            import shutil
            shutil.rmtree("./cache")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 