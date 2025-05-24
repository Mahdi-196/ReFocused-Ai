#!/usr/bin/env python3
"""
Setup Validation for Large Dataset Processing
Checks if everything is ready to handle 13GB Reddit dataset
"""

import sys
from pathlib import Path
from loguru import logger

def check_directory_structure():
    """Check if required directories exist"""
    logger.info("📁 Checking directory structure...")
    
    required_dirs = [
        "data",
        "data/reddit_large",
        "data/cleaned", 
        "data/processed"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists() and path.is_dir()
        status = "✅" if exists else "❌"
        logger.info(f"   {status} {dir_path}")
        
        if not exists:
            all_exist = False
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   🔧 Created {dir_path}")
    
    return all_exist

def check_required_scripts():
    """Check if all required scripts exist"""
    logger.info("🛠️ Checking required scripts...")
    
    required_scripts = [
        ("setup_large_dataset.py", "Dataset setup and extraction"),
        ("data_cleaner.py", "Core data cleaning engine"),
        ("data_processor.py", "Training data preparation"),
        ("training_prep.py", "Training environment setup")
    ]
    
    all_exist = True
    for script_name, description in required_scripts:
        path = Path(script_name)
        exists = path.exists()
        status = "✅" if exists else "❌"
        logger.info(f"   {status} {script_name} - {description}")
        
        if not exists:
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if required Python packages are available"""
    logger.info("📦 Checking dependencies...")
    
    required_packages = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical operations"),
        ("torch", "Machine learning framework"),
        ("transformers", "Pre-trained models"),
        ("nltk", "Text processing"),
        ("better_profanity", "Content filtering"),
        ("loguru", "Logging"),
        ("sklearn", "Data splitting"),
        ("psutil", "System monitoring")
    ]
    
    missing_packages = []
    for package, description in required_packages:
        try:
            __import__(package)
            logger.info(f"   ✅ {package} - {description}")
        except ImportError:
            logger.info(f"   ❌ {package} - {description}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("💡 Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_system_resources():
    """Check available system resources"""
    logger.info("💻 Checking system resources...")
    
    try:
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        memory_status = "✅" if available_gb >= 4 else "⚠️"
        logger.info(f"   {memory_status} RAM: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        # Check available disk space
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        
        disk_status = "✅" if free_gb >= 20 else "⚠️"
        logger.info(f"   {disk_status} Disk: {free_gb:.1f}GB free space")
        
        # Check CPU cores
        cpu_cores = psutil.cpu_count()
        cpu_status = "✅" if cpu_cores >= 2 else "⚠️"
        logger.info(f"   {cpu_status} CPU: {cpu_cores} cores")
        
        # Recommendations
        if available_gb < 4:
            logger.warning("⚠️  Low memory: Consider reducing batch size")
        if free_gb < 20:
            logger.warning("⚠️  Low disk space: Need ~20GB for 13GB dataset processing")
        
        return available_gb >= 2 and free_gb >= 15
        
    except ImportError:
        logger.warning("⚠️  psutil not available - cannot check system resources")
        return True

def estimate_processing_time():
    """Estimate processing time for massive datasets"""
    logger.info("⏱️ Processing time estimates for massive datasets...")
    
    # Estimates for different scenarios - more realistic rates
    scenarios = [
        ("Reddit only (13GB)", 13, 2600000),
        ("HuggingFace only (41GB)", 41, 8000000), 
        ("Both datasets (54GB)", 54, 10600000),
        ("Maximum capacity (150GB)", 150, 30000000)
    ]
    
    for scenario_name, size_gb, estimated_records in scenarios:
        # More realistic processing rates (posts per hour)
        conservative_rate = 2000  # Conservative for your 4-core system
        optimistic_rate = 8000    # Optimistic with good optimization
        
        conservative_hours = estimated_records / conservative_rate
        optimistic_hours = estimated_records / optimistic_rate
        
        logger.info(f"   📊 {scenario_name}:")
        logger.info(f"      Records: {estimated_records:,}")
        logger.info(f"      Time: {optimistic_hours:.1f}-{conservative_hours:.1f} hours")
        logger.info(f"      Output: ~{int(estimated_records * 0.65):,} clean posts")
        logger.info(f"      Size: {size_gb * 0.3:.1f}-{size_gb * 0.5:.1f}GB cleaned")
        logger.info("")

def show_quick_start():
    """Show quick start commands for massive datasets"""
    logger.info("🚀 QUICK START COMMANDS FOR MASSIVE DATASETS:")
    logger.info("-" * 50)
    logger.info("1. Test HuggingFace connection:")
    logger.info("   python setup_massive_dataset.py --huggingface --hf-samples 1000 --analyze-only")
    logger.info("")
    logger.info("2. Analyze your datasets:")
    logger.info("   python setup_massive_dataset.py --reddit YOUR_FILE_PATH --huggingface --analyze-only")
    logger.info("")
    logger.info("3. Extract and prepare (long-running):")
    logger.info("   python setup_massive_dataset.py --reddit YOUR_FILE_PATH --huggingface --extract")
    logger.info("")
    logger.info("4. Process massive dataset:")
    logger.info("   python process_massive_dataset.py")
    logger.info("")
    logger.info("5. Monitor progress:")
    logger.info("   tail -f logs/massive_processing.log")

def main():
    """Main validation function"""
    logger.info("🔍 VALIDATING SETUP FOR 13GB DATASET")
    logger.info("=" * 50)
    
    # Run all checks
    checks = [
        ("Directory Structure", check_directory_structure()),
        ("Required Scripts", check_required_scripts()),
        ("Dependencies", check_dependencies()),
        ("System Resources", check_system_resources())
    ]
    
    # Show results
    logger.info("\n📋 VALIDATION RESULTS:")
    logger.info("-" * 30)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    # Estimate processing time
    logger.info("")
    estimate_processing_time()
    
    # Final status
    logger.info("")
    if all_passed:
        logger.success("🎉 SETUP VALIDATION COMPLETE!")
        logger.info("✅ Your system is ready to process the 13GB dataset")
        logger.info("")
        show_quick_start()
    else:
        logger.error("❌ SETUP VALIDATION FAILED!")
        logger.info("🔧 Please resolve the issues above before proceeding")
        logger.info("💡 Run 'pip install -r requirements.txt' to install missing packages")

if __name__ == "__main__":
    main() 