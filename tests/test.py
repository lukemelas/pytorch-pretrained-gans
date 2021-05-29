"""
pytest tests/test.py --disable-pytest-warnings -s
"""
import pytest
import torch

@pytest.mark.skip(reason="disabled")
def test_load_models():
    
    from pytorch_pretrained_gans import make_gan

    # # BigGAN (conditional)
    # G = make_gan(gan_type='biggan', model_name='biggan-deep-128')
    # y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
    # z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
    # x = G(z=z, y=y)  # -> torch.Size([1, 3, 128, 128])
    # assert list(x.shape) == [1, 3, 128, 128]

    # SelfCondGAN (conditional)
    G = make_gan(gan_type='selfconditionedgan', model_name='self_conditioned')
    y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
    z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
    x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])
    assert list(x.shape) == [1, 3, 256, 256]


