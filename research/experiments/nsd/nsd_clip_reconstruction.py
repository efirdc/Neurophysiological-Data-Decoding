import argparse
import math
import sys
import os
from pathlib import Path
from functools import partial
import random
import json
from typing import Optional, Dict

import pandas
import pandas as pd

sys.path.insert(1, str(Path(os.getcwd()) / 'taming-transformers'))

import fire
import h5py
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from taming.models import cond_transformer, vqgan
from taming.models.vqgan import VQModel
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
import clip
from clip.model import VisionTransformer
import kornia.augmentation as K
import numpy as np
import imageio
from einops import rearrange
from urllib.request import urlopen
from shutil import copyfile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def crowson_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return torch.arcsin((x - y).norm(dim=-1) / 2).pow(2) * 2.


def cosine_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1. - (x * y).sum(dim=-1)


def surface_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    cos_angle = (x * y).sum(dim=-1)
    angle = torch.acos(cos_angle)
    return angle


def square_surface_distance(x, y):
    distance = surface_distance(x, y)
    return distance * distance


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7, p=0.7),
        )

        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []

        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))
        return batch


class EmbeddingLoss(nn.Module):
    def __init__(self, resblocks=False):
        self.resblocks = resblocks
        super().__init__()

    def forward(self, x, y, t):
        #loss = torch.stack([
        #    ((x[f'transformer.resblocks.{i}'] - y[f'transformer.resblocks.{i}']) ** 2).mean()
        #    for i in range(12)
        #]).mean()

        i = int(t * 13)
        if i <= 11 and self.resblocks:
           loss = ((x[f'transformer.resblocks.{i}'] - y[f'transformer.resblocks.{i}']) ** 2).mean()
        else:
           loss = crowson_distance(x['embedding'], y['embedding']).mean()
        # loss = ((x['scratch.refinenet4'] - y['scratch.refinenet4']) ** 2).mean()

        loss = loss * x[f'embedding'].shape[0]
        return loss


def main(
        nsd_path: str,
        model_name: str,
        group_name: str,
        subject: str,
        fold_name: str,
        fold_subset: Optional[str] = None,
        lr: float = 0.1,
        max_iterations: int = 500,
        embedding_iterations: int = 200,
):
    params = dict(locals())
    nsd_path = Path(nsd_path)
    derivatives_path = nsd_path / 'derivatives'

    Y_pred = {}
    decoded_embeddings_file = h5py.File(derivatives_path / f'decoded_features/{model_name}/{group_name}.hdf5', 'r')
    subject_embeddings = decoded_embeddings_file[subject]

    Y_pred.update({
        embedding_name: embedding[fold_name]['Y_pred']
        for embedding_name, embedding in subject_embeddings.items()
    })

    stimulus_ids = list(subject_embeddings.values())[0][fold_name]['stimulus_ids'][:]
    stimulus_order = np.argsort(stimulus_ids)
    if fold_subset:
        subset_file = nsd_path / f'nsddata/stimuli/nsd/{fold_subset}.tsv'
        subset_stimulus_ids = set((np.array(pd.read_csv(subset_file, header=None)[0]) - 1).tolist())
        subset_mask = np.array([i in subset_stimulus_ids for i in stimulus_ids[stimulus_order]])
        stimulus_order = stimulus_order[subset_mask]

    def load_embedding(i):
        if isinstance(i, int):
            i = [i]
        y = {
            k: torch.from_numpy(v[i]).to(torch.float16).to(device)
            for k, v in Y_pred.items()
        }
        return y

    results_root = derivatives_path / 'reconstructions' / model_name / group_name / subject
    results_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    perceptor, preprocess = clip.load('ViT-B/32', device=device)
    perceptor = perceptor.visual
    perceptor = perceptor.eval().requires_grad_(False).to(device)

    hook_modules = {
        **{f'transformer.resblocks.{i}': f'transformer.resblocks.{i}' for i in range(12)},
        '': 'embedding'
    }
    modules = dict(perceptor.named_modules())
    embeddings = {}

    run_id = hex(np.random.randint(2 ** 31)).split('x')[1]
    run_note = '''
    
    '''
    run_folder = results_root / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    with open(run_folder / 'note.txt', 'w') as f:
        f.write(run_note)

    with open(run_folder / 'params.json', 'w') as f:
        f.write(json.dumps(params))

    batch_size = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_name = "vqgan_imagenet_f16_16384"
    vqgan_checkpoint = f"{model_name}.ckpt"
    vqgan_config = f"{model_name}.yaml"
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)

    seed = torch.seed()
    torch.manual_seed(seed)
    print('Using seed:', seed)

    for stimulus_batch in tqdm(np.split(stimulus_order, stimulus_order.shape[0] // batch_size)):
        stimulus_embeddings = load_embedding(stimulus_batch)
        images = reconstruct(
            stimulus_embeddings,
            hook_modules=hook_modules,
            model=model,
            vqgan_checkpoint=vqgan_checkpoint,
            perceptor=perceptor,
            device=device,
            max_iterations=max_iterations,
            embedding_iterations=embedding_iterations,
            lr=lr,
        )

        stim_image_path = results_root / run_id / 'images'
        #stim_video_path.mkdir(exist_ok=True, parents=True)
        stim_image_path.mkdir(exist_ok=True, parents=True)

        #for image_bytes in tqdm(images):
        #    imageio.imwrite('/content/steps/' + str(i) + '.png', np.array(image_bytes))

        batch_stim_ids = stimulus_ids[stimulus_batch]
        for i, image_id in enumerate(stimulus_batch):
            stim_id = batch_stim_ids[i]
            file_name = f'image-{image_id}_stim-{stim_id}_seed-{seed}'
            #video_path = str(stim_video_path / f'{file_name}.mp4')
            image_path = str(stim_image_path / f'{file_name}.png')
            this_images = [image[i] for image in images]
            #create_video(this_images, video_path)
            imageio.imwrite(image_path, this_images[-1])


def reconstruct(
        stimulus_embeddings: Dict[str, torch.Tensor],
        hook_modules: Dict[str, str],
        model: VQModel,
        vqgan_checkpoint: str,
        perceptor: VisionTransformer,
        device: torch.device,
        max_iterations: int,
        embedding_iterations: Optional[int] = None,
        lr: float = 0.1,
        cutn: int = 32,
        cut_pow: float = 1.,
        width: int = 224,
        height: int = 224,
        resblocks: bool = False,
):
    modules = dict(perceptor.named_modules())
    embeddings = {}

    def forward_hook(module_name, module, x_in, x_out):
        embeddings[module_name] = x_out

    hook_handles = []
    for module_name, feature_name in hook_modules.items():
        module = modules[module_name]
        hook_handle = module.register_forward_hook(partial(forward_hook, feature_name))
        hook_handles.append(hook_handle)

    batch_size = stimulus_embeddings['embedding'].shape[0]
    stimulus_embeddings = {
        k: v.repeat(cutn, *[1 for _ in v.shape[1:]])
        for k, v in stimulus_embeddings.items()
    }
    f = 2 ** (model.decoder.num_resolutions - 1)
    cut_size = perceptor.input_resolution
    make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)

    toksX, toksY = width // f, height // f
    sideX, sideY = toksX * f, toksY * f

    if vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
        e_dim = 256
        n_toks = model.quantize.n_embed
        vqgan_embedding_weights = model.quantize.embed.weight
    else:
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e
        vqgan_embedding_weights = model.quantize.embedding.weight

    z_min = vqgan_embedding_weights.min(dim=0).values[None, :, None, None]
    z_max = vqgan_embedding_weights.max(dim=0).values[None, :, None, None]

    selection = (vqgan_embedding_weights.norm(dim=1) > 1).nonzero()[:, 0]
    embedding_indices = torch.randint(selection.shape[0], [toksY * toksX, batch_size], device=device)
    embedding_indices = selection[embedding_indices]
    one_hot = F.one_hot(embedding_indices, n_toks).float()
    z = one_hot @ vqgan_embedding_weights
    z = rearrange(z, '(w h) n e -> n e w h', w=toksX, h=toksY)

    # with torch.no_grad():
    #    x2 = torch.stack([
    #        TF.to_tensor(image.resize((sideX, sideY), Image.LANCZOS))
    #        for image in image_data
    #    ]).to(device)
    #    z, *_ = model.encode(x2 * 2. - 1.)

    # z = torch.rand_like(z) * 2

    z.requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)
    criterion = EmbeddingLoss(resblocks=resblocks)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    images = []
    for i in range(max_iterations):
        optimizer.zero_grad()

        t = (i + 1) / (max_iterations - embedding_iterations + 1)

        z_q = vector_quantize(z.movedim(1, 3), vqgan_embedding_weights).movedim(3, 1)
        image = clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

        if t >= 1:
            cutouts = make_cutouts(image)
            cutouts = normalize(cutouts)
        else:
            cutouts = normalize(image)

        perceptor(cutouts.to(torch.float16)).float()

        loss = criterion(stimulus_embeddings, embeddings, t)

        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #    z.copy_(z.maximum(z_min).minimum(z_max))

        image_bytes = image.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        image_bytes = rearrange(image_bytes, 'n c w h -> n w h c')
        images.append(image_bytes)

    for hook in hook_handles:
        hook.remove()

    return images


if __name__ == '__main__':
    fire.Fire(main)
