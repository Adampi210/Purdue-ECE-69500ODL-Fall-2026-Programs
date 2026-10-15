import torch
import time

device_mps = torch.device('mps' if torch.mps.is_available() else 'cpu')
device_cpu = torch.device('cpu')

A = torch.normal(mean=torch.zeros([10, 3, 4, 4]), std=torch.ones([10, 3, 4, 4]))
B = torch.normal(mean=torch.zeros([10, 1, 4, 4]), std=torch.ones([10, 1, 4, 4]))

A = A.to(device_cpu)
B = B.to(device_cpu)

t0 = time.time()
C = A @ B
t1 = time.time()
print("Time on CPU:", t1 - t0)

A = A.to(device_mps)
B = B.to(device_mps)

t0_mps = time.time()
C = A @ B
t1_mps = time.time()
print("Time on MPS:", t1_mps - t0_mps)

speedup = (t1 - t0) / (t1_mps - t0_mps)
if speedup > 1:
    print(f"MPS is faster by {speedup}x")
else:
    print(f"CPU is faster by {1/speedup}x")
