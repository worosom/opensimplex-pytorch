# opensimplex_pytorch
![](https://media.githubusercontent.com/media/worosom/opensimplex_pytorch/main/assets/noise.png)

## ToDo

- ~~[PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)~~
  - [opensimplex_pytorch/examples/profile.py](https://github.com/worosom/opensimplex_pytorch/examples/profile.py)
- Implement [pcg.py](https://github.com/worosom/opensimplex_pytorch/opensimplex_pytorch/pcg.py) as [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension)
- 4d

## Requirements

- pytorch

To run demo.py:

- vispy

## Install

```bash
pip install git+https://github.com/worosom/opensimplex_pytorch
```

For development:

```bash
git clone https://github.com/worosom/opensimplex_pytorch
cd opensimplex_pytorch
pip install -e .
```

## Usage

```python
from opensimplex_pytorch.simplex3d import simplex3d
```

`simplex3d` accepts a `torch.Tensor` of shape `(batch_size, 3)` and returns a tensor of shape `(batch_size,)` with values centered around `0` a range of `-1/1`.

> Large `>1e4` input values may lead to undesireable artifacts.

```python
import torch
from opensimplex_pytorch.simplex3d import simplex3d

x = torch.zeros((1, 3))
print(simplex3d(x))
# tensor([[0.1294]])
```

```python
import torch
from opensimplex_pytorch.simplex3d import simplex3d

# create a grid of pixel coordinates
xs = torch.linspace(-2, 2, steps=20)
ys = torch.linspace(-2, 2, steps=10)
x, y = torch.meshgrid(xs, ys, indexing='xy')
# add a third dimension for time
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
