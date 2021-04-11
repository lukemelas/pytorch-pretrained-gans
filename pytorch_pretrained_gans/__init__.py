from .BigGAN import make_biggan
from .BigBiGAN import make_bigbigan
from .self_conditioned import make_selfcond_gan
from .stylegan2_ada_pytorch import make_stylegan2
from .StudioGAN import make_studiogan
from .CIPS import make_cips


def make_gan(*, gan_type, **kwargs):
    t = gan_type.lower()
    if t == 'bigbigan':
        G = make_bigbigan(**kwargs)
    elif t == 'selfconditionedgan':
        G = make_selfcond_gan(**kwargs)
    elif t == 'studiogan':
        G = make_studiogan(**kwargs)
    elif t == 'stylegan2':
        G = make_stylegan2(**kwargs)
    elif t == 'cips':
        G = make_cips(**kwargs)
    elif t == 'biggan':
        G = make_biggan(**kwargs)
    else:
        raise NotImplementedError(f'Unrecognized GAN type: {gan_type}')
    return G
