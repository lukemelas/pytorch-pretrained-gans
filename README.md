<div align="center">

## PyTorch Pretrained GANs
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/CVPR-2021-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->

</div>

<!-- TODO: Add video -->

### Quick Start
This repository provides a standardized interface for pretrained GANs in PyTorch. You can install it with:
```bash
pip install git+https://github.com/lukemelas/pytorch-pretrained-gans
```
It is then easy to generate an image with a GAN:
```python
import torch
from pytorch_pretrained_gans import make_gan

# Sample a class-conditional image from BigGAN with default resolution 256
G = make_gan(gan_type='biggan')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])
```

### Motivation
Over the past few years, great progress has been made in generative modeling using GANs. As a result, a large body of research has emerged that uses GANs and explores/interprets their latent spaces. I recently worked on a project in which I wanted to apply the same technique to a bunch of different GANs (here's the [paper](https://github.com/lukemelas/unsupervised-image-segmentation) if you're interested). This was quite a pain because all the pretrained GANs out there are in completely different formats. So I decided to standardize them, and here's the result. I hope you find it useful. 

### Installation
Install with `pip` directly from GitHub: 
```
pip install git+https://github.com/lukemelas/pytorch-pretrained-gans
```

### Available GANs

The following GANs are available. If you would like to add a new GAN to the repo, please submit a pull request -- I would love to add to this list: 
 - [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
 - [BigBiGAN](https://arxiv.org/abs/1907.02544)
 - [StyleGAN-2-ADA](https://arxiv.org/abs/1912.04958)
 - [Self-Conditioned GANs](https://arxiv.org/abs/2006.10728)
 - [Many GANs from StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN):
   - [BigGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) (a reimplementation)
   - [ContraGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
   - [SAGAN](https://arxiv.org/abs/1805.08318)
   - [SNGAN](https://arxiv.org/abs/1802.05957)



### Structure

This repo supports both conditional and unconditional GANs. The standard GAN interface is as follows:

```python
class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format."""

    def __init__(self, G, num_classes=None):
        super().__init__()
        self.G : nn.Module =    # GAN generator
        self.dim_z : int =      # dimensionality of latent space
        self.conditional =      # True / False

    def forward(self, z, y=None):  # y is for conditional GAN only
        x =  # ... generate image from latent with self.G
        return x  # returns image

    def sample_latent(self, batch_size, device='cpu'):
        z =  # ... samples latent vector of size self.dim_z
        return z

    def sample_class(self, batch_size=None, device='cpu'):
        y =  # ... samples class y (for conditional GAN only)
        return y
```

Each type of GAN is contained in its own folder and has a `make_GAN_TYPE` function. For example, `make_bigbigan` creates a BigBiGAN with the format of the `GeneratorWrapper` above. 

The weights of all GANs except those in PyTorch-StudioGAN and are downloaded automatically. To download the PyTorch-StudioGAN weights, use the `download.sh` scripts in the corresponding folders (see the file structure below). 

#### Code Structure
The structure of the repo is below. Each type of GAN has an `__init__.py` file that defines its `GeneratorWrapper` and its `make_GAN_TYPE` file. 

```bash
pytorch_pretrained_gans
├── __init__.py
├── BigBiGAN
│   ├── __init__.py
│   ├── ...
│   └── weights
│       └── download.sh   # (use this to download pretrained weights)
├── BigGAN
│   ├── __init__.py   # (pretrained weights are auto-downloaded)
│   ├── ...
├── StudioGAN
│   ├── __init__.py   # (pretrained weights are auto-downloaded)
│   ├── ...
│   ├── configs
│   │   ├── ImageNet
│   │   │   ├── BigGAN2048
│   │   │   │   └── ...
│   │   │   └── download.sh  # (use this to download pretrained weights)
│   │   └── TinyImageNet
│   │       ├── ACGAN
│   │       │   └── ACGAN.json
│   │       ├── ...
│   │       └── download.sh  # (use this to download pretrained weights)
├── self_conditioned
│   ├── __init__.py   # (pretrained weights are auto-downloaded)
│   └── ...
└── stylegan2_ada_pytorch
    ├── __init__.py   # (pretrained weights are auto-downloaded)
    └── ...
```

### GAN-Specific Details

Naturally, there are some details that are specific to certain GANs. 

**BigGAN:** For BigGAN, you should specify a resolution with `model_name`. For example:
 * `G = make_gan(gan_type='biggan', model_name='biggan-deep-512')`

**StudioGAN:** For StudioGAN, you should specify a model with `model_name`. For example:
 * `G = make_gan(gan_type='studiogan', model_name='SAGAN')`
 * `G = make_gan(gan_type='studiogan', model_name='ContraGAN256')`

**Self-Conditioned GAN:** For StudioGAN, you should specify a model (either `self_conditioned` or `unconditional`) with `model_name`. For example:
 * `G = make_gan(gan_type='selfconditionedgan', model_name='self_conditioned')`

**StyleGAN 2:** 
 * StyleGAN2's `sample_latent` method returns `w`, not `z`, because this is usually what is desired. `w` has shape `torch.Size([1, 18, 512])`.
 * StyleGAN2 is currently not implemented on `CPU`

### Citation
Please cite the following if you use this repo in a research paper:
```bibtex
@inproceedings{melaskyriazi2021finding,
  author    = {Melas-Kyriazi, Luke and Rupprecht, Christian and Laina, Iro and Vedaldi, Andrea},
  title     = {Finding an Unsupervised Image Segmenter in each of your Deep Generative Models},
  booktitle = arxiv,
  year      = {2021}
}
```
