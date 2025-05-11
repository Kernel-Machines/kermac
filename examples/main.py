import kermac
import torch

M = 30000
N = 30000
K = 1000
# Change this to i.e. 1.0, 1.3, 2.0 etc...
p = 1.0

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

print(f'Running p-norm={p} with size ({M},{K}) by ({N},{K})')

print('Generating random data..')

start_event.record()
a = torch.randn(K, M).cuda()
b = torch.randn(K, N).cuda()
c = torch.randn(N, M).cuda()
end_event.record()
torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"Generated random data in: {elapsed_time_ms:.3f} ms")

start_event.record()
c = kermac.cdist_transposed(a, b, p=p,out=c)
end_event.record()

torch.cuda.synchronize()  # Wait for GPU to finish

# Calculate and print elapsed time
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"kermac.cdist_transposed elapsed time: \t{elapsed_time_ms:.3f} ms")

a_t = a.T.contiguous()
b_t = b.T.contiguous()

torch.cuda.synchronize()
start_event.record()
c_pytorch = torch.cdist(a_t, b_t, p=p)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"torch.cdist elapsed time: \t\t{elapsed_time_ms:.3f} ms")

print("kermac.cdist_transposed:")
print(c)
print("torch.cdist:")
print(c_pytorch.T)