docker container run \
--gpus all \
--mount type=bind,source="/remote/cirrus-home/efird/NSD/",target=/NSD \
--mount type=bind,source="/remote/cirrus-home/efird/Deep-Image-Reconstruction/",target=/Deep-Image-Reconstruction \
--mount type=bind,source="/remote/cirrus-home/efird/code/",target=/code \
--interactive \
--tty --rm nvidia/python:3.9 bash


python3 -m research.experiments.nsd.nsd_clip_reconstruction \
  --nsd_path "/NSD" \
  --model_name "ViT-B=32" \
  --group_name "group-1" \
  --subject "subj01" \
  --fold_name "val"

parallel --jobs 2 python3 -m research.experiments.nsd.nsd_clip_reconstruction \
  --nsd_path "/NSD" \
  --model_name "ViT-B=32" \
  --group_name "group-4" \
  --subject subj0{} \
  --fold_name "test" \
  --fold_subset "special100" \
  ::: {1..8}

python3 -m research.experiments.nsd.nsd_encoding clip_fracridge \
--nsd_path "/NSD"

parallel python3 -m research.experiments.nsd.nsd_run_decoding clip \
--nsd_path "/NSD" \
--subject subj0{} ::: {1..8}

parallel python3 -m research.experiments.nsd.nsd_run_decoding clip \
--nsd_path "/NSD" \
--subject subj0{} ::: {1..8}

python3.9 -m research.experiments.nsd.nsd_encoding clip_fracridge \
--nsd_path "/NSD" \
--model_name "ViT-L=14-336px"


python3.9 -m research.experiments.nsd.nsd_encoding sd_clip_fracridge \
--nsd_path "/NSD" \
--embedding True

python3.9 -m research.experiments.nsd.nsd_run_decoding clip \
--nsd_path "/NSD" \
--subject "all" \
--model_name "ViT-L=14-336px"

python3.9 -m research.experiments.nsd.nsd_stable_diffusion main \
--nsd_path "/NSD" \
--sd_config_path "/code/v1-inference.yaml" \
--sd_ckpt_path "/code/sd-v1-4.ckpt" \
--sd_n_iter 1 \
--fold_name "test" \
--fold_subset "special100" \
--sd_batch_size 3 \
--fixed_code True


parallel --jobs 1 python3.9 -m research.experiments.nsd.nsd_stable_diffusion main \
--nsd_path "/NSD" \
--sd_config_path "/code/v1-inference.yaml" \
--sd_ckpt_path "/code/sd-v1-4.ckpt" \
--sd_n_iter 1 \
--fold_name "test" \
--fold_subset "special100" \
--decoded_embeddings True \
--group_name "group-5" \
--subject subj0{} ::: {6..8} \
--average_repetitions True \
--run_number 1

python3.9 -m research.experiments.nsd.nsd_stable_diffusion main \
--nsd_path "/NSD" \
--sd_config_path "/code/v1-inference.yaml" \
--sd_ckpt_path "/code/sd-v1-4.ckpt" \
--sd_n_iter 1 \
--fold_name "test" \
--fold_subset "special100" \
--decoded_embeddings True \
--group_name "group-6" \
--subject "subj01" \
--average_repetitions False \
--run_number 5 \
--sd_batch_size 1 \
--fixed_code True \
--rois "hccp_cortices_rois"


python3.9 -m research.experiments.nsd.nsd_run_decoding sd_clip \
--nsd_path "/NSD" \
--hccp_cortices_rois True \
--subject "all"

python3.9 -m research.experiments.nsd.nsd_encoding sd_clip_fracridge \
--nsd_path "/NSD" \
--subject "subj01" \
--permutation_test True


python -m research.experiments.nsd.nsd_run_decoding clip --nsd_path "D:\\Datasets\\NSD" --subject "all" --model_name "ViT-B=32"

