# opensimplex_pytorch
![](https://media.githubusercontent.com/media/worosom/opensimplex-pytorch/main/assets/noise.png)

## Requirements
- pytorch

To run demo.py:
- vispy
- imageio (optional)

## Usage

Currently, only a 3d variant is implemented.
`simplex3d` accepts a tensor of shape `(batch_size, 3)` and returns a tensor of shape `(batch_size,)`.

Values are centered around `0` with a range of `-1/1`.

```python
import torch
from opensimplex_pytorch import simplex3d

xs = torch.linspace(-2, 2, steps=20)
ys = torch.linspace(-2, 2, steps=10)
x, y = torch.meshgrid(xs, ys, indexing='xy')
t = torch.zeros((10, 20))
xyt = torch.dstack((x, y, t))
flat_xyt = torch.reshape(xyt, (-1, 3))
noise = simplex3d(flat_xyt)
noise_img = torch.reshape(noise, (10, 20))

charset = ['🁣', '🁫', '🁳', '🁻', '🂃', '🂋', '🂓']
noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min()) # range to 0/1
quant_img = (noise_img * (len(charset) - 1)).type(torch.int8)
for line in quant_img:
  for col in line:
    print(charset[col], end='')
  print()

🂃🂓🂃🁫🁫🁻🂃🁳🁳🁳🂃🂃🁳🁣🁣🁫🁻🁳🁫🁳
🁫🁻🁻🁻🁻🁻🁻🁻🁻🁻🁳🁫🁫🁳🁻🂃🂃🂃🂃🂃
🁻🁳🁻🂋🂃🁳🁳🂃🂃🂃🂃🁻🁻🂋🂃🁳🁳🁻🁻🂃
🁳🁳🁻🁻🁻🁳🁳🁫🁳🁻🁻🁳🁳🁳🁳🁫🁫🁳🁳🁫
🁳🁳🁻🁳🁳🁳🁳🁳🁳🁳🁳🁳🁳🁳🁳🁻🁻🂃🂃🂃
🂃🁳🁻🁻🂃🂃🁳🁫🁳🁻🁻🁻🁳🁳🁳🂃🂋🂃🁻🁻
🁳🁳🁳🁻🂃🁻🁻🁻🁻🁫🁫🁳🁳🁫🁳🁻🁻🁻🁻🂃
🁣🁻🂃🂃🁻🁳🁫🁻🂃🁻🁳🁳🁳🁳🁳🂃🂋🂃🁻🁳
🁻🂋🂃🁻🁳🁫🁳🁻🁻🁳🁻🁻🁳🁫🁫🁳🁻🁳🁳🁳
🁳🂃🁻🁳🁫🁳🁳🁻🂃🂋🂋🁻🁣🁣🁳🂃🁻🁫🁣🁫
```

## References

[3d simplex noise by nikat](https://www.shadertoy.com/view/XsX3zB)

[Mark Jarzynski and Marc Olano, Hash Functions for GPU Rendering, Journal of Computer Graphics Techniques (JCGT), vol. 9, no. 3, 21-38, 2020](http://jcgt.org/published/0009/03/02/)
