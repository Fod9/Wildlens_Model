"""
CUDA multiprocessing fixes for PyTorch training
"""

import multiprocessing as mp
import torch
import os


def setup_cuda_multiprocessing():
    """
    Setup CUDA multiprocessing to avoid 'Cannot re-initialize CUDA in forked subprocess' errors
    
    This function should be called at the beginning of training scripts before any CUDA operations.
    """
    
    # Force spawn method for multiprocessing when using CUDA
    if torch.cuda.is_available():
        try:
            mp.set_start_method('spawn', force=True)
            print("✓ CUDA multiprocessing set to 'spawn' method")
        except RuntimeError as e:
            print(f"Warning: Could not set multiprocessing start method: {e}")
    
    # Set environment variables for better CUDA stability
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async CUDA operations
    os.environ['TORCH_USE_CUDA_DSA'] = '1'     # Enable device-side assertions
    
    # Disable OpenMP threading conflicts with PyTorch
    os.environ['OMP_NUM_THREADS'] = '1'
    
    print("✓ CUDA environment configured for stable multiprocessing")


def get_safe_dataloader_config(device: torch.device) -> dict:
    """
    Get DataLoader configuration that avoids CUDA multiprocessing issues
    
    Args:
        device: The device being used for training
        
    Returns:
        Dictionary with safe DataLoader parameters
    """
    
    if device.type == 'cuda':
        # CUDA: Use single-threaded loading to avoid multiprocessing issues
        return {
            'num_workers': 0,
            'pin_memory': True,
            'persistent_workers': False
        }
    else:
        # CPU: Can use multiple workers safely
        return {
            'num_workers': min(4, mp.cpu_count()),
            'pin_memory': False,
            'persistent_workers': True
        }


def check_cuda_multiprocessing():
    """
    Check if CUDA multiprocessing is properly configured
    """
    
    print("🔍 CUDA Multiprocessing Configuration Check:")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  - CUDA device count: {torch.cuda.device_count()}")
        print(f"  - Current CUDA device: {torch.cuda.current_device()}")
        
        try:
            start_method = mp.get_start_method()
            print(f"  - Multiprocessing start method: {start_method}")
            
            if start_method != 'spawn':
                print("  ⚠️  Warning: Start method is not 'spawn' - may cause CUDA errors")
            else:
                print("  ✓ Start method is 'spawn' - CUDA multiprocessing should work")
                
        except Exception as e:
            print(f"  ❌ Error checking start method: {e}")
    
    # Check environment variables
    cuda_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not set')
    torch_dsa = os.environ.get('TORCH_USE_CUDA_DSA', 'Not set')
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'Not set')
    
    print(f"  - CUDA_LAUNCH_BLOCKING: {cuda_blocking}")
    print(f"  - TORCH_USE_CUDA_DSA: {torch_dsa}")
    print(f"  - OMP_NUM_THREADS: {omp_threads}")