import os
from distutils.core import setup

__version__ = '0.1.0'

proj_root=os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(proj_root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opensimplex_pytorch',
    packages=['opensimplex_pytorch'],
    license='MIT',
    version=__version__,
    author='Alexander Morosow',
    description='OpenSimplex Noise - implemented in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/worosom/opensimplex_pytorch',
    project_urls={
        'Homepage': 'https://github.com/worosom/opensimplex_pytorch',
        'Bug Tracker': 'https://github.com/worosom/opensimplex_pytorch/issues'
    },
    requires=[
        'torch',
        'torchvision',
        'torchaudio',
    ],
)
