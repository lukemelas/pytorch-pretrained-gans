import os
from os.path import join
from pathlib import Path
import numpy as np
import torch
import json
from collections.abc import MutableMapping

from .models import resnet
from .models import big_resnet
from .models import big_resnet_deep


# Download here: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN#imagenet-3x128x128
ROOT = Path(__file__).parent
ARCHS = {
    'resnet': resnet,
    'big_resnet': big_resnet,
    'big_resnet_deep': big_resnet_deep,
}


class Config(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def flatten(d):
    items = []
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            items.extend(flatten(v).items())
        else:
            items.append((k, v))
    return dict(items)


def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format """

    def __init__(self, Gen, cfgs):
        super().__init__()
        self.G = Gen
        self.dim_z = Gen.z_dim
        self.conditional = True
        self.num_classes = cfgs.num_classes

        self.truncation = 1.0

    def forward(self, z, y=None, return_y=False):
        if y is not None:
            # the model is conditional and the user gives us a class
            y = y.to(z.device)
        elif self.num_classes is not None:
            # the model is conditional but the user does not give us a class
            y = self.sample_class(batch_size=z.shape[0], device=z.device)
        else:
            # the model is unconditional
            y = None
        x = self.G(z, label=y, evaluation=True)
        x = torch.clamp(x, min=-1, max=1)  # this shouldn't really be necessary
        return (x, y) if return_y else x

    def sample_latent(self, batch_size, device='cpu'):
        z = torch.randn((batch_size, self.dim_z), device=device)
        return z

    def sample_class(self, batch_size, device='cpu'):
        y = torch.randint(low=0, high=self.num_classes, size=(batch_size,), device=device)
        return y


def get_config_and_checkpoint(root):
    paths = list(map(str, root.iterdir()))
    checkpoint_path = [p for p in paths if '.pth' in p]
    config_path = [p for p in paths if '.json' in p]
    assert len(checkpoint_path) == 1, f'no checkpoint found in {root}'
    assert len(config_path) == 1, f'no config found in {root}'
    checkpoint_path = checkpoint_path[0]
    config_path = config_path[0]
    with open(config_path) as f:
        cfgs = json.load(f)
        cfgs = Config(flatten(cfgs))
        cfgs.mixed_precision = False
    return cfgs, checkpoint_path


def make_studiogan(model_name='SAGAN', dataset='ImageNet') -> torch.nn.Module:

    # Get configs and model checkpoint path
    cfgs, checkpoint_path = get_config_and_checkpoint(ROOT / 'configs' / dataset / model_name)

    # From: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/loader.py#L90
    Generator = ARCHS[cfgs.architecture].Generator
    Gen = Generator(
        cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
        cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
        cfgs.g_init, cfgs.G_depth, cfgs.mixed_precision)

    # Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    Gen.load_state_dict(checkpoint['state_dict'])

    # Wrap
    G = GeneratorWrapper(Gen, cfgs)
    return G.eval()


if __name__ == '__main__':
    # Testing
    device = 'cuda'
    G = make_studiogan('BigGAN2048').to(device)
    print('Created G')
    print(f'Params: {sum(p.numel() for p in G.parameters()):_}')
    z = torch.randn([1, G.dim_z]).to(device)
    print(f'z.shape: {z.shape}')
    x = G(z)
    print(f'x.shape: {x.shape}')
    print(x.max(), x.min())
