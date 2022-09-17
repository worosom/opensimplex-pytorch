import torch

with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile/'),
    record_shapes=True,
    with_modules=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as profiler:
    from opensimplex_pytorch.simplex3d import simplex3d

    extent = 20
    resolution = 2048
    for _ in range(10):

        with torch.no_grad():
            xs = torch.linspace(-extent, extent, steps=resolution)
            ys = torch.linspace(-extent, extent, steps=resolution)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            t = torch.zeros((resolution, resolution))
            for dt in range(0, 1, 100000):
                xyt = torch.dstack((x, y, t + dt))
                flat_xyt = torch.reshape(xyt, (resolution**2, -1))
                image = simplex3d(flat_xyt)
        profiler.step()

print(profiler.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

