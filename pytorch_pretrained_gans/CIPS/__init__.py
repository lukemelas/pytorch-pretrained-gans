import torch
from typing import NamedTuple

try:
    from .GeneratorsCIPS import CIPSskip
    model_available = True
except:
    CIPSskip = None
    model_available = False


class Churches256Arguments(NamedTuple):
    """CIPSskip for LSUN-Churches-256"""
    Generator = CIPSskip
    size = 256
    coords_size = 256
    fc_dim = 512
    latent = 512
    style_dim = 512
    n_mlp = 8
    activation = None
    channel_multiplier = 2
    coords_integer_values = False


MODELS = {
    # Download from https://github.com/saic-mdal/CIPS#pretrained-checkpoints
    'churches': ('/home/luke/projects/experiments/gan-seg/src/segmentation/gans/CIPS/churches_g_ema.pt', Churches256Arguments),
}


class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format """

    def __init__(self, g_ema, args, truncation=0.7, device='cpu'):
        super().__init__()
        self.G = g_ema.to(device)
        self.dim_z = g_ema.style_dim
        self.conditional = False

        self.truncation = truncation
        self.truncation_latent = get_latent_mean(g_ema, args, device)
        self.x_channel, self.y_channel = convert_to_coord_format_unbatched(
            args.coords_size, args.coords_size, device,
            integer_values=args.coords_integer_values)
        self.coords_size = args.coords_size

    def forward(self, z):
        x_channel = self.x_channel.repeat(z.size(0), 1, self.coords_size, 1).to(z.device)
        y_channel = self.y_channel.repeat(z.size(0), 1, 1, self.coords_size).to(z.device)
        converted_full = torch.cat((x_channel, y_channel), dim=1)
        sample, _ = self.G(
            coords=converted_full,
            latent=[z],
            return_latents=False,
            truncation=self.truncation,
            truncation_latent=self.truncation_latent,
            input_is_latent=True)
        sample = torch.clamp(sample, min=-1, max=1)  # I don't know if this is needed, I think it is though
        return sample


def convert_to_coord_format_unbatched(h, w, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1)
    else:
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1)
    return (x_channel, y_channel)


def make_cips(model_name='churches', **kwargs) -> torch.nn.Module:
    if not model_available:
        raise Exception('Could not load model. Do you have CUDA?')

    checkpoint_path, args = MODELS[model_name]
    g_ema = args.Generator(
        size=args.size,
        hidden_size=args.fc_dim,
        style_dim=args.latent,
        n_mlp=args.n_mlp,
        activation=args.activation,
        channel_multiplier=args.channel_multiplier)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    g_ema.load_state_dict(ckpt)
    G = GeneratorWrapper(g_ema, args, **kwargs)
    return G.eval()


@torch.no_grad()
def get_latent_mean(g_ema, args, device):

    # Get sample input
    n_sample = 1
    sample_z = torch.randn(n_sample, args.latent, device=device)
    x_channel, y_channel = convert_to_coord_format_unbatched(args.coords_size, args.coords_size, device,
                                                             integer_values=args.coords_integer_values)
    x_channel = x_channel.repeat(sample_z.size(0), 1, args.coords_size, 1).to(device)
    y_channel = y_channel.repeat(sample_z.size(0), 1, 1, args.coords_size).to(device)
    converted_full = torch.cat((x_channel, y_channel), dim=1)

    # Generate a bunch of times and
    latents = []
    samples = []
    for _ in range(100):
        sample_z = torch.randn(n_sample, args.latent, device=device)
        sample, latent = g_ema(converted_full, [sample_z], return_latents=True)
        latents.append(latent.cpu())
        samples.append(sample.cpu())
    samples = torch.cat(samples, 0)
    latents = torch.cat(latents, 0)
    truncation_latent = latents.mean(0).cuda()
    assert len(truncation_latent.shape) == 1 and truncation_latent.size(0) == 512, 'smt wrong'
    return truncation_latent


if __name__ == '__main__':
    # Testing
    device = torch.device('cuda')
    G = make_cips(device=device)
    print('Created G')
    print(f'Params: {sum(p.numel() for p in G.parameters()):_}')
    z = torch.randn([1, G.dim_z]).to(device)
    print(f'z.shape: {z.shape}')
    x = G(z)
    print(f'x.shape: {x.shape}')
    import pdb
    pdb.set_trace()
