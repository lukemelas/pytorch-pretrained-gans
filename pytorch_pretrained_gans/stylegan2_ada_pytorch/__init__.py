import os
import sys
import torch
from torch.hub import urlparse, get_dir, download_url_to_file
import pickle


MODELS = {
    'ffhq': ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', None),
    'afhqwild': ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl', None),
}


def download_url(url, download_dir=None, filename=None):
    parts = urlparse(url)
    if download_dir is None:
        hub_dir = get_dir()
        download_dir = os.path.join(hub_dir, 'checkpoints')
    if filename is None:
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(download_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file)
    return cached_file


class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format. This wrapper takes
        w as input, rather than (z, c) """

    def __init__(self, G, num_classes=None):
        super().__init__()
        self.G = G  # NOTE! This takes in w, rather than z
        self.dim_z = G.synthesis.w_dim
        self.conditional = (num_classes is not None)
        self.num_classes = num_classes

        self.num_ws = G.synthesis.num_ws
        self.truncation_psi = 0.5
        self.truncation_cutoff = 8

    def forward(self, z):
        r"""The input `z` is expected to be `w`, not `z`, in the notation
            of the original StyleGAN 2 paper"""
        if len(z.shape) == 2:  # expand to 18 layers
            z = z.unsqueeze(1).repeat(1, self.num_ws, 1)
        return self.G.synthesis(z)

    def sample_latent(self, batch_size, device='cpu'):
        z = torch.randn([batch_size, self.dim_z], device=device)
        c = None if self.conditional else None  # not implemented for conditional models
        w = self.G.mapping(z, c, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff)
        return w


def add_utils_to_path():
    import sys
    from pathlib import Path
    util_path = str(Path(__file__).parent)
    if util_path not in sys.path:
        sys.path.append(util_path)
        print(f'Added {util_path} to path')


def make_stylegan2(model_name='ffhq') -> torch.nn.Module:
    """G takes as input an image in NCHW format with dtype float32, normalized 
    to the range [-1, +1]. Some models also take a conditioning class label, 
    which is passed as img = G(z, c)"""
    add_utils_to_path()  # we need dnnlib and torch_utils in the path
    url, num_classes = MODELS[model_name]
    cached_file = download_url(url)
    assert cached_file.endswith('.pkl')
    with open(cached_file, 'rb') as f:
        G = pickle.load(f)['G_ema']
    G = GeneratorWrapper(G, num_classes=num_classes)
    return G.eval()


if __name__ == '__main__':
    # Testing
    G = make_stylegan2().cuda()
    print('Created G')
    print(f'Params: {sum(p.numel() for p in G.parameters()):_}')
    z = torch.randn([1, G.dim_z]).cuda()
    print(f'z.shape: {z.shape}')
    x = G(z)
    print(f'x.shape: {x.shape}')
