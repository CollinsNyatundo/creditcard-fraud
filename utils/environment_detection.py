import torch
import sys
def check_gpu_and_rapids():
    """
    Checks for GPU availability with PyTorch and RAPIDS library installations.
    """
    print("--- GPU Availability (PyTorch) ---")
    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        current_device_idx = torch.cuda.current_device()
        print(f"Current GPU device index: {current_device_idx}")
        print(f"Current GPU device name: {torch.cuda.get_device_name(current_device_idx)}")
        # List all GPUs
        print("\nAll available GPUs:")
        for i in range(gpu_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA-enabled GPU found. PyTorch will use CPU.")
    print("\n--- RAPIDS Installation Check ---")
    rapids_libraries = {
        "cuDF": "cudf",
        "cuML": "cuml",
        "cuGraph": "cugraph",
        "RMM": "rmm"  # RAPIDS Memory Manager, often a dependency
    }
    installed_rapids = {}
    for name, module_name in rapids_libraries.items():
        try:
            __import__(module_name)
            installed_rapids[name] = True
            print(f"{name} ({module_name}): INSTALLED")
        except ImportError:
            installed_rapids[name] = False
            print(f"{name} ({module_name}): NOT INSTALLED")
    print("\n--- Summary ---")
    if gpu_available:
        print("GPU is available and configured for PyTorch.")
        if all(installed_rapids.values()):
            print("All common RAPIDS libraries are installed.")
        elif any(installed_rapids.values()):
            print("Some RAPIDS libraries are installed.")
        else:
            print("No common RAPIDS libraries are installed.")
    else:
        print("No GPU available. PyTorch will run on CPU. RAPIDS libraries, if installed, will not function on GPU.")
        if any(installed_rapids.values()):
            print("Warning: RAPIDS libraries are installed but cannot utilize GPU without a detected CUDA device.")
    # Determine processing environment
    if gpu_available and all(installed_rapids.get(lib) for lib in ["cuDF", "cuML"]):
        environment = "gpu_rapids"
    elif gpu_available:
        environment = "gpu_pytorch"
    else:
        environment = "cpu_optimized"
    print(f"\n--- Detected Processing Environment: {environment} ---")
    # Basic test computation
    if environment == "gpu_rapids":
        try:
            import cudf
            s = cudf.Series([1, 2, 3, 4, 5])
            print(f"cuDF Series created: {s}")
            print("cuDF is functional.")
        except Exception as e:
            print(f"Error using cuDF: {e}")
            environment = "gpu_pytorch"  # Fallback to PyTorch if cuDF fails
    elif environment == "gpu_pytorch":
        try:
            s = torch.tensor([1, 2, 3, 4, 5]).cuda()
            print(f"PyTorch GPU tensor created: {s}")
            print("PyTorch GPU is functional.")
        except Exception as e:
            print(f"Error using PyTorch GPU: {e}")
            environment = "cpu_optimized"  # Fallback to CPU if PyTorch GPU fails
    else:  # cpu_optimized
        try:
            import pandas as pd
            s = pd.Series([1, 2, 3, 4, 5])
            print(f"Pandas Series created: {s}")
            print("CPU-based processing is functional.")
        except Exception as e:
            print(f"Error using Pandas: {e}")
            print("Critical error: No working computation environment found.")
    return environment, gpu_available, installed_rapids
if __name__ == "__main__":
    environment, gpu_status, rapids_status = check_gpu_and_rapids()
    print(f"\nFinal detected environment: {environment}")
