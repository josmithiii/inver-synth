"""
Device detection utilities for TensorFlow training.
Handles CUDA, MPS (Metal Performance Shaders), and CPU fallback.
"""
import os
import tensorflow as tf
from typing import Tuple, Optional


def detect_best_device() -> Tuple[str, str, bool]:
    """
    Detect the best available device for TensorFlow training.
    
    Returns:
        Tuple of (device_name, device_type, use_mixed_precision)
        - device_name: Full device name (e.g., "/GPU:0", "/device:CPU:0")
        - device_type: Simple type ("cuda", "mps", "cpu")
        - use_mixed_precision: Whether mixed precision is recommended
    """
    # Check for CUDA GPU availability
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Configure GPU memory growth to avoid OOM errors
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be set before GPUs have been initialized
                pass
            
            # Check if it's CUDA
            gpu_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', '')
            if 'NVIDIA' in gpu_name or 'GeForce' in gpu_name or 'Tesla' in gpu_name or 'Quadro' in gpu_name:
                return "/GPU:0", "cuda", True
    
    # Check for MPS (Metal Performance Shaders) on Apple Silicon
    try:
        # Try to create a simple operation on MPS
        if hasattr(tf.config.experimental, 'list_devices'):
            devices = tf.config.experimental.list_devices()
            for device in devices:
                if 'GPU' in device and ('Metal' in device or 'MPS' in device):
                    return "/GPU:0", "mps", False  # Mixed precision not recommended for MPS yet
        
        # Alternative check for Apple Silicon MPS
        if tf.config.list_physical_devices('GPU'):
            # On Apple Silicon, GPU devices are MPS
            import platform
            if platform.system() == 'Darwin' and platform.processor() == 'arm':
                return "/GPU:0", "mps", False
    except Exception:
        pass
    
    # Fallback to CPU
    return "/device:CPU:0", "cpu", False


def configure_tensorflow_device(force_device: Optional[str] = None) -> Tuple[str, str, bool]:
    """
    Configure TensorFlow to use the best available device.
    
    Args:
        force_device: Force a specific device type ("cuda", "mps", "cpu", or None for auto-detect)
        
    Returns:
        Tuple of (device_name, device_type, use_mixed_precision)
    """
    if force_device:
        force_device = force_device.lower()
        if force_device == "cpu":
            return "/device:CPU:0", "cpu", False
        elif force_device == "cuda":
            if tf.config.list_physical_devices('GPU'):
                gpus = tf.config.list_physical_devices('GPU')
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
                return "/GPU:0", "cuda", True
            else:
                print("WARNING: CUDA requested but no GPU available, falling back to CPU")
                return "/device:CPU:0", "cpu", False
        elif force_device == "mps":
            if tf.config.list_physical_devices('GPU'):
                return "/GPU:0", "mps", False
            else:
                print("WARNING: MPS requested but no GPU available, falling back to CPU")
                return "/device:CPU:0", "cpu", False
    
    return detect_best_device()


def print_device_info():
    """Print detailed information about available devices."""
    device_name, device_type, use_mixed_precision = detect_best_device()
    
    print("=" * 50)
    print("DEVICE CONFIGURATION")
    print("=" * 50)
    print(f"Selected Device: {device_name}")
    print(f"Device Type: {device_type.upper()}")
    print(f"Mixed Precision: {'Enabled' if use_mixed_precision else 'Disabled'}")
    
    # Print all available devices
    print("\nAvailable Physical Devices:")
    for device_type_name in ['CPU', 'GPU']:
        devices = tf.config.list_physical_devices(device_type_name)
        if devices:
            for i, device in enumerate(devices):
                print(f"  {device_type_name}:{i} - {device}")
                if device_type_name == 'GPU':
                    try:
                        details = tf.config.experimental.get_device_details(device)
                        if details:
                            print(f"    Details: {details}")
                    except:
                        pass
        else:
            print(f"  No {device_type_name} devices found")
    
    print("=" * 50)
    

def get_device_from_env() -> Optional[str]:
    """Get device preference from environment variable TF_DEVICE."""
    return os.environ.get('TF_DEVICE', None)


def setup_tensorflow_for_training() -> Tuple[str, str, bool]:
    """
    Complete TensorFlow setup for training with device detection.
    
    Returns:
        Tuple of (device_name, device_type, use_mixed_precision)
    """
    # Check for environment override
    env_device = get_device_from_env()
    
    # Configure device
    device_name, device_type, use_mixed_precision = configure_tensorflow_device(env_device)
    
    # Setup mixed precision if recommended
    if use_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision (float16) enabled for faster training")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")
            use_mixed_precision = False
    
    # Print configuration
    print_device_info()
    
    return device_name, device_type, use_mixed_precision