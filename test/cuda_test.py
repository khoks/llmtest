import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)



# If CUDA is available, check the CUDA device
if cuda_available:
    print("CUDA device_count:", torch.cuda.device_count())
    print("CUDA max_memory_allocated:", torch.cuda.max_memory_allocated())
    print("CUDA current_device:", torch.cuda.current_device())
    print("CUDA get_device_name:", torch.cuda.get_device_name())
else:
    print("No CUDA device found.")


# Create a tensor and move it to the GPU
x = torch.tensor([1.0, 2.0, 3.0])
x = x.to('cuda')

print("Tensor on CUDA:", x)

# Perform a simple operation
y = x * 2
print("Result:", y)