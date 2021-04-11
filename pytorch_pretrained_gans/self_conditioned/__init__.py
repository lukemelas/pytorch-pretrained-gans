import torch
from torch.utils import model_zoo
from .gan_training.models import generator_dict

# Config, adapted from:
# - https://github.com/stevliu/self-conditioned-gan/blob/master/configs/imagenet/default.yaml
# - https://github.com/stevliu/self-conditioned-gan/blob/master/configs/imagenet/unconditional.yaml
# - https://github.com/stevliu/self-conditioned-gan/blob/master/configs/imagenet/selfcondgan.yaml
configs = {
    'unconditional': {
        'generator': {
            'name': 'resnet2',
            'kwargs': {},
            # Unconditional
            'nlabels': 1,
            'conditioning': 'unconditional',
        },
        'z_dist': {
            'dim': 256
        },
        'data': {
            'img_size': 128
        },
        'pretrained': {
            'model': 'http://selfcondgan.csail.mit.edu/weights/uncondgan_i_model.pt'
        }
    },
    'self_conditioned': {
        'generator': {
            'name': 'resnet2',
            'kwargs': {},
            # Self-conditional
            'nlabels': 100,
            'conditioning': 'embedding',
        },
        'z_dist': {
            'dim': 256
        },
        'data': {
            'img_size': 128
        },
        'pretrained': {
            'model': 'http://selfcondgan.csail.mit.edu/weights/selfcondgan_i_model.pt'
        }
    }
}


class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format and add metadata (dim_z) """

    def __init__(self, generator, dim_z, nlabels):
        super().__init__()
        self.G = generator
        self.dim_z = dim_z
        self.conditional = True
        self.num_classes = nlabels

    def forward(self, z, y=None, return_y=False):
        if y is None:
            y = self.sample_class(batch_size=z.shape[0], device=z.device)
        else:
            y = y.to(z.device)
        x = self.G(z, y)
        return (x, y) if return_y else x

    def sample_latent(self, batch_size, device='cpu'):
        z = torch.randn(size=(batch_size, self.dim_z), device=device)
        return z

    def sample_class(self, batch_size=None, device='cpu'):
        y = torch.randint(low=0, high=self.num_classes, size=(batch_size,), device=device)
        return y


def make_selfcond_gan(model_name='self_conditioned'):
    """ A helper function for loading a (pretrained) GAN """

    # Get generator configuration
    assert model_name in {'self_conditioned', 'unconditional'}
    config = configs[model_name]

    # Create GAN
    Generator = generator_dict[config['generator']['name']]
    generator = Generator(
        z_dim=config['z_dist']['dim'],
        nlabels=config['generator']['nlabels'],
        size=config['data']['img_size'],
        conditioning=config['generator']['conditioning'],
        **config['generator']['kwargs']
    )

    # Load checkpoint
    checkpoint = model_zoo.load_url(config['pretrained']['model'], map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    print(f"Loaded pretrained GAN weights (iteration: {checkpoint['it']})")

    # Wrap GAN
    G = GeneratorWrapper(
        generator=generator,
        dim_z=config['z_dist']['dim'],
        nlabels=config['generator']['nlabels']
    ).eval()

    return G


if __name__ == "__main__":

    # Load model
    G = make_selfcond_gan('self-conditioned')
    print(f'Parameters: {sum(p.numel() for p in G.parameters()) / 10**6} million')

    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    with torch.no_grad():
        z = torch.randn(7, G.dim_z, requires_grad=False, device=device)
        x = G(z)
        print(f'Input shape: {z.shape}')
        print(f'Output shape: {x.shape}')
