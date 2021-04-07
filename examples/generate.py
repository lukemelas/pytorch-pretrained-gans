"""
Example usage:
python generate.py
"""
import torch
from pytorch_pretrained_gans import make_gan

# BigGAN (unconditional)
G = make_gan(gan_type='biggan', model_type='biggan-deep-512')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])

# BigBiGAN (unconditional)
G = make_gan(gan_type='bigbigan')  # -> nn.Module
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 120])
x = G(z=z)  # -> torch.Size([1, 3, 128, 128])

# Self-Conditioned GAN (unconditional)
G = make_gan(gan_type='selfconditionedgan', model_name='self_conditioned')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 128, 128])

# StudioGAN (unconditional)
G = make_gan(gan_type='studiogan', model_name='SAGAN')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 128, 128])

# StyleGAN2 (unconditional)
G = make_gan(gan_type='stylegan2')  # -> nn.Module
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 120])
x = G(z=z)  # -> torch.Size([1, 3, 128, 128])
