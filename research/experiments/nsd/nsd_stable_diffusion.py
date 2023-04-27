from pathlib import Path
import json
import gc
from typing import Optional, List, Union, Sequence

import pandas as pd
import fire
import h5py
import imageio
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main(
        nsd_path: str,
        sd_config_path: str,
        sd_ckpt_path: str,
        sd_plms_sampling: bool = False,
        sd_precision: str = 'autocast',
        sd_ddim_steps: int = 50,
        sd_ddim_eta: float = 0.0,
        sd_n_iter: int = 2,
        sd_H: int = 512,
        sd_W: int = 512,
        sd_C: int = 4,
        sd_f: int = 8,
        sd_scale: float = 7.5,
        sd_batch_size: int = 1,
        decoded_embeddings: bool = False,
        subject: Union[str, List] = None,
        group_name: Optional[str] = None,
        fold_name: Optional[str] = None,
        fold_subset: Optional[str] = None,
        average_repetitions: bool = False,
        seed: int = 0,
        run_number: Optional[int] = None,
        rois: Optional[str] = None,
        fixed_code: bool = False,
):
    if rois is None:
        rois = ['']
    elif rois == 'hccp_cortices_rois':
        rois = [
            'Primary_Visual', 'Early_Visual', 'Dorsal_Stream_Visual', 'Ventral_Stream_Visual',
            'MT+_Complex_and_Neighboring_Visual_Areas', 'Medial_Temporal', 'Lateral_Temporal',
            'Temporo-Parieto-Occipital_Junction', 'Superior_Parietal', 'Inferior_Parietal',
            'Posterior_Cingulate', 'Frontal',
        ]
    else:
        raise ValueError()

    seed_everything(seed)

    if isinstance(subject, list):
        subjects = subject
    elif subject == 'all':
        subjects = [f'subj0{i}' for i in range(1, 9)]
    else:
        subjects = [subject]

    model_name = 'clip-vit-large-patch14-text'
    params = dict(locals())
    print(params)
    nsd_path = Path(nsd_path)
    derivatives_path = nsd_path / 'derivatives'

    if decoded_embeddings:
        results_root = derivatives_path / 'reconstructions' / model_name / group_name
        results_root.mkdir(parents=True, exist_ok=True)
    else:
        results_root = derivatives_path / 'reconstructions' / model_name / 'ground_truth'
    results_root.mkdir(parents=True, exist_ok=True)

    if run_number is None:
        run_number = 1
        run_folders = [p.name for p in results_root.iterdir()]
        while f'run-{run_number:03}' in run_folders:
            run_number += 1
    run_id = f'run-{run_number:03}'

    run_path = results_root / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    run_note = '''
        
    '''
    with open(run_path / 'note.txt', 'w') as f:
        f.write(run_note)

    with open(run_path / 'params.json', 'w') as f:
        f.write(json.dumps(params))

    for subject in subjects:
        for roi in rois:
            gc.collect()
            torch.cuda.empty_cache()
            Y_pred = {}

            if decoded_embeddings:
                embeddings_file = h5py.File(derivatives_path / f'decoded_features/{model_name}/{group_name}.hdf5', 'r')
                subject_embeddings = embeddings_file[f'{subject}/embedding_unpooled/{roi}/{fold_name}']
                Y_pred = subject_embeddings['Y_pred']
                stimulus_ids = subject_embeddings['stimulus_ids'][:]
                run_path = results_root / run_id / subject / roi

            else:
                embeddings_file = h5py.File(derivatives_path / f'stimulus_embeddings/{model_name}.hdf5')
                Y_pred = embeddings_file['embedding_unpooled']
                stimulus_ids = np.arange(Y_pred.shape[0])

            run_path.mkdir(parents=True, exist_ok=True)

            if average_repetitions:
                unique_stimulus_ids = np.unique(stimulus_ids)
                Y_pred = np.stack([Y_pred[i == stimulus_ids].mean(axis=0) for i in unique_stimulus_ids])
                stimulus_ids = unique_stimulus_ids

            stimulus_order = np.argsort(stimulus_ids)
            if fold_subset:
                subset_file = nsd_path / f'nsddata/stimuli/nsd/{fold_subset}.tsv'
                subset_stimulus_ids = set((np.array(pd.read_csv(subset_file, header=None)[0]) - 1).tolist())
                subset_mask = np.array([i in subset_stimulus_ids for i in stimulus_ids[stimulus_order]])
                stimulus_order = stimulus_order[subset_mask]

            config = OmegaConf.load(sd_config_path)
            model = load_model_from_config(config, sd_ckpt_path)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model = model.to(device)

            if sd_plms_sampling:
                sampler = PLMSSampler(model)
            else:
                sampler = DDIMSampler(model)

            precision_scope = autocast if sd_precision == "autocast" else nullcontext

            with torch.no_grad():
                with precision_scope('cuda'):
                    with model.ema_scope():
                        for i in stimulus_order:
                            uc = None
                            if sd_scale != 1.0:
                                uc = model.get_learned_conditioning(sd_batch_size * [""])
                            c = torch.from_numpy(Y_pred[i])[None].to(device)
                            c = torch.cat(sd_batch_size * [c])

                            stim_id = stimulus_ids[i]
                            if fixed_code:
                                torch.random.manual_seed(stim_id)
                                start_code = torch.randn([sd_batch_size, sd_C, sd_H // sd_f, sd_W // sd_f], device=device)
                            else:
                                start_code = None

                            shape = [sd_C, sd_H // sd_f, sd_W // sd_f]
                            samples_ddim, _ = sampler.sample(
                                S=sd_ddim_steps,
                                conditioning=c,
                                batch_size=sd_batch_size,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=sd_scale,
                                unconditional_conditioning=uc,
                                eta=sd_ddim_eta,
                                x_T=start_code
                            )

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            for batch_id in range(sd_batch_size):
                                x_sample = x_samples_ddim.cpu().numpy()[batch_id]

                                x_sample = rearrange((x_sample * 255).astype(np.uint8), 'c h w -> h w c')
                                img = Image.fromarray(x_sample)

                                stim_image_path = run_path / 'images'
                                stim_image_path.mkdir(exist_ok=True, parents=True)

                                file_name = f'image-{i}_stim-{stim_id}_seed-{seed}_v-{batch_id}'
                                image_path = str(stim_image_path / f'{file_name}.png')
                                imageio.imwrite(image_path, img)


if __name__ == '__main__':
    fire.Fire(main)
