#!/usr/bin/env python3
"""
Storage Optimization Analysis
Shows how 2.3TB storage improves training efficiency
"""

def analyze_storage_usage():
    print("💾 STORAGE OPTIMIZATION ANALYSIS")
    print("=" * 50)
    
    # Your dataset
    dataset_size_gb = 21.7  # Your tokenized data
    available_storage_tb = 2.3
    available_storage_gb = available_storage_tb * 1024
    
    print(f"📊 Available Resources:")
    print(f"   Total storage: {available_storage_tb} TB ({available_storage_gb:,.0f} GB)")
    print(f"   Your dataset: {dataset_size_gb} GB")
    print(f"   Storage utilization: {(dataset_size_gb/available_storage_gb)*100:.1f}%")
    
    # Storage allocation for training
    allocations = {
        "Training data (/scratch/shards)": dataset_size_gb,
        "Checkpoints (/scratch/checkpoints)": 150,  # ~5 checkpoints × 30GB each
        "Logs & monitoring (/scratch/logs)": 5,
        "DeepSpeed NVMe offload": 100,  # For optimizer states
        "System cache & temp files": 50,
        "Buffer/free space": 200
    }
    
    total_used = sum(allocations.values())
    
    print(f"\n📁 Optimal Storage Allocation:")
    for purpose, size_gb in allocations.items():
        percentage = (size_gb / available_storage_gb) * 100
        print(f"   {purpose:<35} {size_gb:>6.0f} GB ({percentage:>4.1f}%)")
    
    print(f"   {'─' * 35} {'─' * 6}    {'─' * 6}")
    print(f"   {'Total used':<35} {total_used:>6.0f} GB ({(total_used/available_storage_gb)*100:>4.1f}%)")
    print(f"   {'Remaining free':<35} {available_storage_gb-total_used:>6.0f} GB ({((available_storage_gb-total_used)/available_storage_gb)*100:>4.1f}%)")
    
    print(f"\n🚀 Performance Benefits:")
    print(f"   ✅ No data transfer delays (all data local)")
    print(f"   ✅ Fast checkpoint saves (NVMe speeds)")
    print(f"   ✅ DeepSpeed NVMe offloading (reduces GPU memory pressure)")
    print(f"   ✅ Multiple checkpoint retention (recovery options)")
    print(f"   ✅ Extensive logging without storage concerns")
    
    print(f"\n💡 Without This Storage:")
    print(f"   ❌ Would need to download 21.7GB repeatedly from GCS")
    print(f"   ❌ Slower checkpoint saves to remote storage")
    print(f"   ❌ No local NVMe offloading (higher GPU memory usage)")
    print(f"   ❌ Limited checkpoint retention")
    print(f"   ❌ Potential training interruptions from I/O bottlenecks")
    
    # Cost comparison
    print(f"\n💰 Cost Impact:")
    print(f"   Current: $7.92/hour (storage included)")
    print(f"   Alternative: $7.92/hour + GCS egress + slower training")
    print(f"   Estimated savings: 10-20% faster training = $32-64 saved!")
    
    print(f"\n🎯 Bottom Line:")
    print(f"   The 2.3TB storage is FREE and makes training:")
    print(f"   • Faster (no network I/O bottlenecks)")
    print(f"   • More reliable (local data)")
    print(f"   • More efficient (NVMe offloading)")
    print(f"   • Actually SAVES money through faster completion!")

if __name__ == "__main__":
    analyze_storage_usage() 