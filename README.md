## Fruit Classification with Vision Transformers and Self‑Supervised Learning
Author: Mingke Zhao 

## Project Overview
This project explores the combination of Vision Transformers and Masked Autoencoders for self‑supervised pretraining to achieve high‑accuracy fruit classification on the Fruit‑360 dataset. 
I also compare the classification performance of:
MAE‑pretrained ViT  
Randomly initialized ViT  
Classical CNN baselines (ResNet‑18/34/50/101)  

## Requirements
Python  3.11  
PyTorch  2.3.1  
torchvision 0.18.1  
timm 0.3.2  
numpy,pandas,matplotlib  
CUDA  12.4
NVIDIA-SMI Version: 550.54.15

initialization
-pip install timm==0.3.2
-pip install submitit
-pip install GPUtil
## dataset spliting 
 see dataset.ipynb
## Self‑Supervised Pretraining (MAE)

Pretrain order:
python mae-main/submitit_pretrain.py \
    --job_dir pretrain \
    --nodes 1 \
    --ngpus 1 \
    --use_volta32 \
    --batch_size 32 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 100 \
    --warmup_epochs 40 \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --resume pretrain/checkpoint-80.pth \
    --data_path mae-main/fruit\
    

## Fine‑Tuning (ViT)

To fine‑tune MAE‑pretrained ViT on Fruit‑360:

python mae-main/submitit_finetune.py \
    --job_dir finetune \
    --nodes 1 \
    --ngpus 1 \
    --batch_size 128\
    --model vit_base_patch16 \
    --finetune  pretrain/checkpoint-50.pth\
    --nb_classes 141\
    --epochs 40\
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path pretrain/40%/finetune\
## ResNet Model
see file CNN.ipynb
   
## Result Analysis
see file figure.ipynb

##  Acknowledgement
Adapted from @Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}; licensed under CC BY‑NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)







