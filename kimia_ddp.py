import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple linear layer to verify matrix multiplication works on the GPU
        self.net = nn.Linear(2048, 2048)
    def forward(self, x):
        return self.net(x)

def run_test():
    # 1. Check if torchrun injected the environment variables
    if "LOCAL_RANK" not in os.environ:
        print("ERROR: This script must be launched using 'torchrun'!")
        print("Run it with: torchrun --nproc_per_node=4 test_ddp.py")
        sys.exit(1)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 2. Initialize the distributed environment
    # Note: If NCCL hangs, change backend="nccl" to backend="gloo"
    dist.init_process_group(backend="nccl")

    # 3. Explicitly pin this process to its specific GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Print status from each process
    gpu_name = torch.cuda.get_device_name(local_rank)
    print(f"[Process {local_rank}/{world_size-1}] Successfully tied to GPU {local_rank}: {gpu_name}")

    # 4. Move model and data to this specific GPU
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    # Create fake inputs and targets directly on this GPU
    inputs = torch.randn(64, 2048).to(device)
    targets = torch.randn(64, 2048).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5. Run a fake forward and backward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Synchronize all processes to make sure they finished together
    dist.barrier()

    if local_rank == 0:
        print("\n🎉 SUCCESS! All 4 processes initialized, communicated, and ran a training step.")

    dist.destroy_process_group()

if __name__ == "__main__":
    run_test()