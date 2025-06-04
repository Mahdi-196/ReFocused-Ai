#!/usr/bin/env python3
"""
GPU Debug Script for ReFocused-AI Training
Run this script to diagnose GPU and CUDA issues before training.
"""

import os
import sys
import platform
import subprocess

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed and working"""
    print("🔍 Checking NVIDIA Drivers...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA drivers are installed and working")
            print("nvidia-smi output:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed to run")
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")
        print("   NVIDIA drivers are not installed or not in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out")
        return False
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA installation"""
    print("\n🔥 Checking PyTorch CUDA...")
    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA compiled version: {torch.version.cuda}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"     Compute capability: {props.major}.{props.minor}")
                print(f"     Total memory: {props.total_memory / 1024**3:.2f} GB")
                
            # Test basic CUDA operations
            print("\n🧪 Testing basic CUDA operations...")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                print("✅ Basic CUDA tensor operations work")
                print(f"   Matrix multiplication result shape: {z.shape}")
                
                # Test memory allocation
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                print(f"   GPU memory allocated: {memory_allocated:.2f} MB")
                
                return True
            except Exception as e:
                print(f"❌ CUDA operations failed: {e}")
                return False
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def check_tensorboard():
    """Check TensorBoard installation"""
    print("\n📊 Checking TensorBoard...")
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard (PyTorch) available")
        
        # Test creating a SummaryWriter
        try:
            writer = SummaryWriter('./test_logs')
            writer.add_scalar('test', 1.0, 0)
            writer.close()
            print("✅ TensorBoard test write successful")
            
            # Clean up test logs
            import shutil
            if os.path.exists('./test_logs'):
                shutil.rmtree('./test_logs')
            return True
        except Exception as e:
            print(f"❌ TensorBoard test failed: {e}")
            return False
            
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
            print("✅ TensorboardX (fallback) available")
            return True
        except ImportError:
            print("❌ No TensorBoard implementation found")
            return False

def check_environment():
    """Check environment setup"""
    print("\n🔧 Checking Environment...")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # Check important environment variables
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER', 
        'NVIDIA_VISIBLE_DEVICES',
        'PATH'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH' and len(value) > 100:
            value = value[:100] + '...'
        print(f"{var}: {value}")

def check_vm_status():
    """Check if running in a VM"""
    print("\n🖥️  Checking VM Status...")
    try:
        # Windows-specific VM detection
        if platform.system() == 'Windows':
            try:
                import wmi
                c = wmi.WMI()
                for system in c.Win32_ComputerSystem():
                    model = system.Model.lower()
                    if any(vm in model for vm in ['vmware', 'virtualbox', 'hyper-v', 'virtual']):
                        print(f"🔍 Running in VM: {system.Model}")
                        print("   Note: GPU passthrough may need to be configured")
                        return True
                    else:
                        print(f"🔍 Physical hardware: {system.Model}")
                        return False
            except ImportError:
                print("Cannot detect VM (WMI not available)")
        
        # Generic VM detection
        vm_indicators = [
            '/proc/vz',  # OpenVZ
            '/proc/xen',  # Xen
            '/.dockerenv',  # Docker
        ]
        
        for indicator in vm_indicators:
            if os.path.exists(indicator):
                print(f"🔍 VM indicator found: {indicator}")
                return True
        
        print("🔍 No VM indicators found")
        return False
        
    except Exception as e:
        print(f"Error checking VM status: {e}")
        return False

def main():
    """Main debug function"""
    print("🚀 ReFocused-AI GPU Debug Tool")
    print("=" * 50)
    
    all_checks = []
    
    # Run all checks
    all_checks.append(("NVIDIA Drivers", check_nvidia_drivers()))
    all_checks.append(("PyTorch CUDA", check_pytorch_cuda()))
    all_checks.append(("TensorBoard", check_tensorboard()))
    
    # Environment checks (informational)
    check_environment()
    check_vm_status()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    for check_name, result in all_checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    all_passed = all(result for _, result in all_checks)
    
    if all_passed:
        print("\n🎉 All checks passed! GPU training should work.")
    else:
        print("\n⚠️  Some checks failed. Training may run on CPU only.")
        print("\nRecommendations:")
        
        if not all_checks[0][1]:  # NVIDIA drivers failed
            print("• Install NVIDIA GPU drivers")
            print("• Ensure GPU is properly connected")
            print("• Check if VM has GPU passthrough enabled")
        
        if not all_checks[1][1]:  # PyTorch CUDA failed
            print("• Reinstall PyTorch with CUDA support:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        if not all_checks[2][1]:  # TensorBoard failed
            print("• Install TensorBoard:")
            print("  pip install tensorboard tensorboardX")

if __name__ == "__main__":
    main() 