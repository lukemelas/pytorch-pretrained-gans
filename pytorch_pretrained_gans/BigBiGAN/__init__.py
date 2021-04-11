from pathlib import Path
import torch
from torch import nn
from torch.utils import model_zoo
from .model import BigGAN


_WEIGHTS_URL = "https://github.com/greeneggsandyaml/tmp/releases/download/0.0.1/BigBiGAN_x1.pth"


class GeneratorWrapper(nn.Module):
    """ A wrapper to put the GAN in a standard format -- here, a modified
        version of the old UnconditionalBigGAN class """

    def __init__(self, big_gan):
        super().__init__()
        self.big_gan = big_gan
        self.dim_z = self.big_gan.dim_z
        self.conditional = False

    def forward(self, z):
        classes = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        return self.big_gan(z, self.big_gan.shared(classes))

    def sample_latent(self, batch_size, device='cpu'):
        z = torch.randn((batch_size, self.dim_z), device=device)
        return z


def make_biggan_config(resolution):
    attn_dict = {128: '64', 256: '128', 512: '64'}
    dim_z_dict = {128: 120, 256: 140, 512: 128}
    config = {
        'G_param': 'SN', 'D_param': 'SN',
        'G_ch': 96, 'D_ch': 96,
        'D_wide': True, 'G_shared': True,
        'shared_dim': 128, 'dim_z': dim_z_dict[resolution],
        'hier': True, 'cross_replica': False,
        'mybn': False, 'G_activation': nn.ReLU(inplace=True),
        'G_attn': attn_dict[resolution],
        'norm_style': 'bn',
        'G_init': 'ortho', 'skip_init': True, 'no_optim': True,
        'G_fp16': False, 'G_mixed_precision': False,
        'accumulate_stats': False, 'num_standing_accumulations': 16,
        'G_eval_mode': True,
        'BN_eps': 1e-04, 'SN_eps': 1e-04,
        'num_G_SVs': 1, 'num_G_SV_itrs': 1, 'resolution': resolution,
        'n_classes': 1000}
    return config


def make_bigbigan(model_name='bigbigan-128'):
    assert model_name == 'bigbigan-128'
    config = make_biggan_config(resolution=128)
    G = BigGAN.Generator(**config)
    checkpoint = model_zoo.load_url(_WEIGHTS_URL, map_location='cpu')
    G.load_state_dict(checkpoint)  # , strict=False)
    G = GeneratorWrapper(G)
    return G
