import os
import sys
import tarfile
from typing import Mapping
import numpy as np
from pathlib import Path
import torch
from torch.hub import urlparse, get_dir, download_url_to_file, load_state_dict_from_url
import time

from . import utils as inference_utils
from .BigGAN_PyTorch import utils as biggan_utils


MODELS = {
    'icgan_biggan_imagenet_res256': {
        'checkpoint': 'https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_imagenet_res256.tar.gz',
        'conditioning': 'https://github.com/greeneggsandyaml/tmp/releases/download/v0.0.2/imagenet_res256_rn50_selfsupervised_kmeans_k1000_instance_features.npy'
    },  # TODO: https://dl.fbaipublicfiles.com/ic_gan/stored_instances.tar.gz
    'icgan_biggan_imagenet_res128': {
        'checkpoint': 'https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_imagenet_res128.tar.gz',
        'conditioning': 'https://github.com/greeneggsandyaml/tmp/releases/download/v0.0.2/imagenet_res128_rn50_selfsupervised_kmeans_k1000_instance_features.npy'
    }
}


def download_url(url, download_dir=None, filename=None):
    parts = urlparse(url)
    if download_dir is None:
        hub_dir = get_dir()
        download_dir = os.path.join(hub_dir, 'checkpoints')
    else:
        if not os.path.exists(download_dir):
            print(f'Creating because it does not exist: {download_dir}')
            os.makedirs(download_dir)
    if filename is None:
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(download_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file)
    return cached_file


def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., variance: float = 1.0, seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    if truncation == 1.0:
        return np.random.randn(batch_size, dim_z) * variance
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


def get_model(exp_name, root_path, backbone, device="cuda"):
    
    # Create config
    parser = biggan_utils.prepare_parser()
    parser = biggan_utils.add_sample_parser(parser)
    parser = inference_utils.add_backbone_parser(parser)
    args = ["--experiment_name", exp_name]
    args += ["--base_root", root_path]
    args += ["--model_backbone", backbone]
    config = vars(parser.parse_args(args=args))

    # Load model and overwrite configuration parameters if stored in the model
    config = biggan_utils.update_config_roots(config, change_weight_folder=False)
    generator, config = inference_utils.load_model_inference(config, device=device)
    biggan_utils.count_parameters(generator)
    generator.eval()

    return generator, config


class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format """

    def __init__(self, G, instance_features: Mapping, dim_z: int = 120, truncation: float = 1.0, conditional: bool = False):
        super().__init__()
        self.G = G
        self.instance_features = instance_features
        self.dim_z = dim_z
        self.conditional = conditional
        self.num_classes = 1000
        self.truncation = truncation

    def forward(self, z, feats=None, y=None, return_y=False):
        """ In original code, z -> noise_vector, y -> class_vector """
        if feats is None:
            feats = self.sample_features(batch_size=z.shape[0], device=z.device)
        if self.conditional:
            raise NotImplementedError()
            if y is None:
                y = self.sample_class(batch_size=z.shape[0], device=z.device)
            elif y.dtype == torch.long:
                y = torch.eye(self.num_classes, dtype=torch.float, device=y.device)[y]
            else:
                y = y.to(z.device)
            x = self.G(z, y)
            # x = torch.clamp(x, min=-1, max=1)  # this shouldn't really be necessary
            return (x, y) if return_y else x
        else:
            return self.G(z, None, feats)

    def sample_latent(self, batch_size, device='cpu'):
        z = truncated_noise_sample(truncation=self.truncation, batch_size=batch_size)
        z = torch.from_numpy(z).to(device).float()
        return z
    
    def sample_features(self, batch_size, device='cpu'):
        indices = np.random.choice(range(1000), batch_size, replace=False)
        feats = self.instance_features['instance_features'][indices]
        feats = torch.from_numpy(feats).to(device).float()
        return feats

    def sample_class(self, batch_size, device='cpu'):
        y = torch.randint(low=0, high=self.num_classes, size=(batch_size,), device=device)
        y = torch.eye(self.num_classes, dtype=torch.float, device=device)[y]
        return y


def make_icgan(model_name='icgan_biggan_imagenet_res256') -> torch.nn.Module:
    
    # Create download dir for our files
    torch_hub_root_dir = get_dir()
    download_dir = os.path.join(torch_hub_root_dir, 'ic_gan')
    experiment_dir = os.path.join(torch_hub_root_dir, 'ic_gan', model_name)

    # Download checkpoints
    urls = MODELS[model_name]
    instances_file = download_url(urls['conditioning'], download_dir=download_dir)
    if not os.path.exists(os.path.join(experiment_dir, 'state_dict_best0.pth')):
        checkpoint_file = download_url(urls['checkpoint'], download_dir=download_dir)
        with tarfile.open(checkpoint_file) as f:
            f.extractall(path=download_dir)
        os.remove(checkpoint_file)  # delete the checkpoint file when done
        print('Finished extracting')
    
    # Load instance features
    instance_features = np.load(instances_file, allow_pickle=True).item()

    # Create model and load checkpoint
    G, config = get_model(exp_name=model_name, root_path=download_dir, backbone="biggan", device="cpu")
    G = GeneratorWrapper(G, instance_features=instance_features, dim_z=config['dim_z'])
    return G.eval()


if __name__ == '__main__':
    # Testing
    device = torch.device('cuda')
    G = make_icgan()
    G.to(device).eval()
    print('Created G')
    print(f'Params: {sum(p.numel() for p in G.parameters()):_}')
    z = G.sample_latent(batch_size=2, device=device)
    print(f'Starting generation...')
    print(f'z.shape: {z.shape}')
    x = G(z)
    print(f'x.shape: {x.shape}')
    print(f'x.max(): {x.max()}')
    print(f'x.min(): {x.min()}')
    print(z)
