import os
import sys
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
import torch
import torchvision.transforms as transforms
from torch.hub import urlparse, get_dir, download_url_to_file, load_state_dict_from_url
import time

# from .model import BigGAN
from . import utils as inference_utils
from .BigGAN_PyTorch import utils as biggan_utils
from .data_utils import utils as data_utils
from .data_utils.datasets_common import pil_loader


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



def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., variance: float = None, seed=None):
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


# def get_data(root_path, model, resolution, which_dataset, visualize_instance_images):
#     data_path = os.path.join(root_path, "stored_instances")
#     if model == "cc_icgan":
#         feature_extractor = "classification"
#     else:
#         feature_extractor = "selfsupervised"
#     filename = "%s_res%i_rn50_%s_kmeans_k1000_instance_features.npy" % (
#         which_dataset,
#         resolution,
#         feature_extractor,
#     )
#     # Load conditioning instances from files
#     data = np.load(os.path.join(data_path, filename), allow_pickle=True).item()
#     return data


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


# def get_conditionings(test_config, generator, data):
#     # Obtain noise vectors
#     z = torch.empty(
#         test_config["num_imgs_gen"] * test_config["num_conditionings_gen"],
#         generator.z_dim if config["model_backbone"] == "stylegan2" else generator.dim_z,
#     ).normal_(mean=0, std=test_config["z_var"])

#     # Subsampling some instances from the 1000 k-means centers file
#     if test_config["num_conditionings_gen"] > 1:
#         total_idxs = np.random.choice(
#             range(1000), test_config["num_conditionings_gen"], replace=False
#         )

#     # Obtain features, labels and ground truth image paths
#     all_feats, all_img_paths, all_labels = [], [], []
#     for counter in range(test_config["num_conditionings_gen"]):
#         # Index in 1000 k-means centers file
#         if test_config["index"] is not None:
#             idx = test_config["index"]
#         else:
#             idx = total_idxs[counter]
#         # Image paths to visualize ground-truth instance
#         all_img_paths.append(data["image_path"][idx])
#         # Instance features
#         all_feats.append(
#             torch.FloatTensor(data["instance_features"][idx : idx + 1]).repeat(
#                 test_config["num_imgs_gen"], 1
#             )
#         )
#         # Obtain labels
#         if test_config["swap_target"] is not None:
#             # Swap label for a manually specified one
#             label_int = test_config["swap_target"]
#         else:
#             # Use the label associated to the instance feature
#             label_int = int(data["labels"][idx])
#         # Format labels according to the backbone
#         labels = None
#         if test_config["model_backbone"] == "stylegan2":
#             dim_labels = 1000
#             labels = torch.eye(dim_labels)[torch.LongTensor([label_int])].repeat(
#                 test_config["num_imgs_gen"], 1
#             )
#         else:
#             if test_config["model"] == "cc_icgan":
#                 labels = torch.LongTensor([label_int]).repeat(
#                     test_config["num_imgs_gen"]
#                 )
#         all_labels.append(labels)
#     # Concatenate all conditionings
#     all_feats = torch.cat(all_feats)
#     if all_labels[0] is not None:
#         all_labels = torch.cat(all_labels)
#     else:
#         all_labels = None
#     return z, all_feats, all_labels, all_img_paths


class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format """

    def __init__(self, G, dim_z: int = 120, truncation: float = 1.0, conditional: bool = False):
        super().__init__()
        self.G = G
        self.dim_z = dim_z
        self.conditional = conditional
        self.num_classes = 1000
        self.truncation = truncation

    def forward(self, z, y=None, return_y=False):
        """ In original code, z -> noise_vector, y -> class_vector """
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
            return self.G(z)

    def sample_latent(self, batch_size, device='cpu'):
        z = truncated_noise_sample(truncation=self.truncation, batch_size=batch_size)
        z = torch.from_numpy(z).to(device)
        return z

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

    # Create model and load checkpoint
    G, config = get_model(exp_name=model_name, root_path=download_dir, backbone="biggan", device="cpu")
    G = GeneratorWrapper(G, dim_z=config['dim_z'])
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
