#!/usr/bin/env python3
"""
Quick Setup Summary for ReFocused AI
"""

from pathlib import Path
import psutil

def check_system_readiness():
    """Quick system readiness check"""
    print("🔍 ReFocused AI - System Check")
    print("=" * 40)
    
    # Check data directories
    data_dirs = ["reddit_ultra_fast", "reddit_enhanced", "reddit_oauth", "unified_raw"]
    data_found = False
    total_files = 0
    
    for subdir in data_dirs:
        path = Path(f"data/{subdir}")
        if path.exists():
            txt_files = list(path.glob("*.txt"))
            if txt_files:
                total_files += len(txt_files)
                data_found = True
                print(f"   ✅ {subdir}: {len(txt_files)} files")
    
    if not data_found:
        print("   ❌ No data files found")
    
    # Check key scripts
    scripts = ["data_cleaner.py", "setup_massive_dataset.py", "validate_setup.py"]
    all_scripts = all(Path(script).exists() for script in scripts)
    status = "✅" if all_scripts else "❌"
    print(f"   {status} Processing scripts")
    
    # System resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    disk_gb = psutil.disk_usage('.').free / (1024**3)
    cpu_cores = psutil.cpu_count()
    
    print(f"\n💻 System Resources:")
    print(f"   RAM: {memory_gb:.1f}GB")
    print(f"   Disk: {disk_gb:.1f}GB free")
    print(f"   CPU: {cpu_cores} cores")
    
    # Readiness assessment
    ready = data_found and all_scripts and memory_gb >= 4 and disk_gb >= 20
    
    print(f"\n🎯 Status: {'✅ Ready' if ready else '⚠️ Setup needed'}")
    
    if ready:
        print("\n🚀 Quick Start:")
        print("   python validate_setup.py")
        print("   python setup_massive_dataset.py --help")
    else:
        print("\n📋 Next Steps:")
        if not data_found:
            print("   • Add data files to data/ subdirectories")
        if memory_gb < 4:
            print("   • Warning: Low memory may slow processing")
        if disk_gb < 20:
            print("   • Warning: Need more disk space for processing")
        print("   • Run: pip install -r requirements.txt")

if __name__ == "__main__":
    try:
        check_system_readiness()
    except ImportError:
        print("❌ Missing dependencies. Run: pip install -r requirements.txt") 