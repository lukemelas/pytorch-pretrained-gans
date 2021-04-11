"""
Example usage:
python generate.py
"""
import torch
from pytorch_pretrained_gans import make_gan

# BigGAN (unconditional)
G = make_gan(gan_type='biggan', model_name='biggan-deep-256')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])
assert z.shape == torch.Size([1, 128])
assert x.shape == torch.Size([1, 3, 256, 256])

# BigBiGAN (unconditional)
G = make_gan(gan_type='bigbigan')  # -> nn.Module
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 120])
x = G(z=z)  # -> torch.Size([1, 3, 128, 128])
assert z.shape == torch.Size([1, 120])
assert x.shape == torch.Size([1, 3, 128, 128])

# Self-Conditioned GAN (unconditional)
G = make_gan(gan_type='selfconditionedgan', model_name='self_conditioned')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 256])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 128, 128])
assert z.shape == torch.Size([1, 256])
assert x.shape == torch.Size([1, 3, 128, 128])

# StyleGAN2 (unconditional)
G = make_gan(gan_type='stylegan2').to('cuda')  # -> nn.Module
z = G.sample_latent(batch_size=1, device='cuda')  # -> torch.Size([1, 18, 512])
x = G(z=z)  # -> torch.Size([1, 3, 1024, 1024])
assert z.shape == torch.Size([1, 18, 512])
assert x.shape == torch.Size([1, 3, 1024, 1024])

try:
    # StudioGAN (unconditional)
    G = make_gan(gan_type='studiogan', model_name='SAGAN')  # -> nn.Module
    y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
    z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
    x = G(z=z, y=y)  # -> torch.Size([1, 3, 128, 128])
    assert z.shape == torch.Size([1, 128])
    assert x.shape == torch.Size([1, 3, 128, 128])
except:
    print('Please download StudioGAN models as specified in the repo')