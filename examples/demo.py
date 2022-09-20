import vispy.app
from vispy import visuals
from vispy.visuals.transforms import STTransform

import torch
from opensimplex_pytorch.simplex3d import Simplex3D
from opensimplex_pytorch.misc import rand_rotation_matrix


torch.random.manual_seed(0)
canvas_size = 1024

device = torch.device('cuda:0')
simplex3d = Simplex3D(device)

extent = 20
resolution = 1024
with torch.no_grad():
    xs = torch.linspace(-extent, extent, steps=resolution)
    ys = torch.linspace(-extent, extent, steps=resolution)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    t = torch.zeros((resolution, resolution))
    rotation_matrix = rand_rotation_matrix().type(torch.float32)

def get_image(dt):
    with torch.no_grad():
        xyt = torch.dstack((x, y, t + dt))
        flat_xyt = torch.reshape(xyt, (resolution**2, -1))
        rot_xyt = torch.einsum('in,mn->im', flat_xyt, rotation_matrix).to(device)
        image = simplex3d(rot_xyt)
        image = torch.reshape(image, (resolution, resolution, -1)).cpu().numpy()
    return image * .5 + .5

class Canvas(vispy.app.Canvas):
    def __init__(self, **kwargs):
        self.t = 1e-4
        vispy.app.Canvas.__init__(self, **kwargs)
        self.image = visuals.ImageVisual(get_image(self.t), method='subdivide', cmap='grays')

        # scale and center image in canvas
        s = canvas_size / max(self.image.size)
        t = 0.5 * (canvas_size - (self.image.size[0] * s))
        self.image.transform = STTransform(scale=(s, s), translate=(t, 0))

        self.show()

    def on_draw(self, ev):
        self.image.draw()
        self.t += .01
        i = get_image(self.t)
        self.image.set_data(i)
        self.update()

    def on_mouse_press(self, event):
        import imageio
        img = self.render()[:, :, :3]
        imageio.imwrite('noise.png', img)

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.image.transforms.configure(canvas=self, viewport=vp)


if __name__ == '__main__':
    canvas = Canvas(keys='interactive', size=(canvas_size, canvas_size))
    canvas.measure_fps()
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
