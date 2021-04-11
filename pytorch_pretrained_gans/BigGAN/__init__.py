import contextlib
import os
import torch
import numpy as np

from .model import BigGAN


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

    def __init__(self, G, truncation=0.4):
        super().__init__()
        self.G = G
        self.dim_z = G.config.z_dim
        self.conditional = True
        self.num_classes = 1000

        self.truncation = truncation

    def forward(self, z, y=None, return_y=False):
        """ In original code, z -> noise_vector, y -> class_vector """
        if y is None:
            y = self.sample_class(batch_size=z.shape[0], device=z.device)
        elif y.dtype == torch.long:
            y = torch.eye(self.num_classes, dtype=torch.float, device=y.device)[y]
        else:
            y = y.to(z.device)
        x = self.G(z, y, truncation=self.truncation)
        x = torch.clamp(x, min=-1, max=1)  # this shouldn't really be necessary
        return (x, y) if return_y else x

    def sample_latent(self, batch_size, device='cpu'):
        z = truncated_noise_sample(truncation=self.truncation, batch_size=batch_size)
        z = torch.from_numpy(z).to(device)
        return z

    def sample_class(self, batch_size, device='cpu'):
        y = torch.randint(low=0, high=self.num_classes, size=(batch_size,), device=device)
        y = torch.eye(self.num_classes, dtype=torch.float, device=device)[y]
        return y


def make_biggan(model_name='biggan-deep-256') -> torch.nn.Module:
    G = BigGAN.from_pretrained(model_name).eval()
    G = GeneratorWrapper(G)
    return G.eval()


if __name__ == '__main__':
    # Testing
    device = torch.device('cuda')
    G = make_pretrained_biggan('biggan-deep-512')
    G.to(device).eval()
    print('Created G')
    print(f'Params: {sum(p.numel() for p in G.parameters()):_}')
    z = torch.randn([1, G.dim_z]).to(device)
    print(f'z.shape: {z.shape}')
    x = G(z)
    print(f'x.shape: {x.shape}')
    print(f'x.max(): {x.max()}')
    print(f'x.min(): {x.min()}')
