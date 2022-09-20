import torch
from tqdm import tqdm
from opensimplex_pytorch.simplex3d import Simplex3D

DEVICE = 'cuda:0'

for lod in range(4, 11):
    resolution = 2**lod
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=100,
            active=900,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/{DEVICE}-{resolution}'),
        record_shapes=True,
        with_modules=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as profiler:
        extent = resolution / 4
        device = torch.device(DEVICE)
        simplex3d = Simplex3D(device)
        xs = torch.linspace(-extent, extent, steps=resolution, device=device)
        ys = torch.linspace(-extent, extent, steps=resolution, device=device)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        x = x.to(device)
        y = y.to(device)
        t = torch.zeros((resolution, resolution), device=device)
        pbar = tqdm(total=1000, desc=f'{resolution}x{resolution}')
        for dt in torch.arange(0, 1, 1/1000, device=device):
            xyt = torch.dstack((x, y, t + dt))
            flat_xyt = torch.reshape(xyt, (resolution**2, -1))
            image = simplex3d(flat_xyt)
            pbar.set_postfix(std_mean="%0.2f/%0.2f" % torch.std_mean(image, unbiased=False))
            pbar.update()
            profiler.step()
        del x, y, t

