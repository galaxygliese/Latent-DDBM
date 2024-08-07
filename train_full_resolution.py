#-*- coding:utf-8 -*-

from diffusion import create_model, DdbmEdmDenoiser
from dataset import FolderDataset, FolderPairDataset, expand2square
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
from torchvision import transforms as T
from tqdm.auto import tqdm

import torchvision.transforms as transforms
from functools import partial
from PIL import Image
import numpy as np
import argparse
import torch.nn as nn
import torch
import wandb
import os 

parser = argparse.ArgumentParser()

# General options
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batchsize', type=int, default=8)
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training
parser.add_argument('--ema_power', type=float, default=0.75)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--export_folder', type=str, default="./checkpoints")
parser.add_argument('--warmup_steps', type=int, default=50)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--save-per-epoch', type=int, default=50)
parser.add_argument('-d', '--device', type=int, default=0)

# Dataset options
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)

# Karras (EDM) options
parser.add_argument('--sigma_data', type=float, default=0.5)
parser.add_argument('--sigma_sample_density_mean', type=float, default=-1.2)
parser.add_argument('--sigma_sample_density_std', type=float, default=1.2)
parser.add_argument('--sigma_max', type=float, default=80)
parser.add_argument('--sigma_min', type=float, default=0.0002)
parser.add_argument('--rho', type=float, default=7.0)

# Resuming options
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_checkpoint', type=str)
parser.add_argument('--resume_epochs', type=int, default=0)
opt = parser.parse_args()

# LATENT_DIM = opt.image_size // 8

def rand_log_normal(shape, loc=0., scale=1., device='cuda', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()

@torch.no_grad()
def generate(
        model:DdbmEdmDenoiser, 
        y: torch.Tensor,
        num_diffusion_iters:int, 
        export_name:str, 
        # sample_num:int, 
        device:str='cuda'
    ):
    # B = sample_num
    with torch.no_grad():
        # initialize action from Guassian noise
        nimage = model.sample(y, steps=num_diffusion_iters)
        
        imgs = nimage.detach().to('cpu')
        imgs = 0.5*(imgs+1)
        imgs = (imgs*255).clip(0, 255)
        
        img = make_grid(imgs)
        img = transforms.functional.to_pil_image(img)
        # (B, 3, H, W)
    img = Image.fromarray(np.asarray(img))
    img.save(export_name)

def train():
    with tqdm(range(opt.resume_epochs, opt.epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            model.train()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    x, y = nbatch[0].to(device), nbatch[1].to(device)
                    B = x.shape[0]
                    
                    # sample a diffusion iteration for each data point
                    loss = model.get_loss(x_start=x, x_T=y).mean()

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                run.log({"train-loss": np.mean(epoch_loss)})
                tglobal.set_postfix(loss=np.mean(epoch_loss))   
            
                if (epoch_idx + 1) % opt.save_per_epoch == 0:
                    torch.save(model.state_dict(), f'{opt.export_folder}/training-ddbm-edm-latent-diffusion-epoch{epoch_idx+1}'+'.pt')
                    model.eval()
                    generate(
                        model=model,
                        y=test_y,
                        num_diffusion_iters=num_diffusion_iters,
                        export_name=f"{opt.export_folder}/epoch{epoch_idx+1}.png",
                        # sample_num=4
                    )

if __name__ == '__main__':
    device = opt.device
    image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Lambda(lambda img: expand2square(img)),
        T.Resize(opt.image_size),
        # T.CenterCrop(opt.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    # TODO: add data sampler for multi-gpu
    dataset = FolderPairDataset(
        folder_path=opt.dataset_path,
        image_size=opt.image_size,
        transform=image_transform
    )
    
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchsize,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True, 
        # don't kill worker process afte each epoch
        persistent_workers=True 
    )

    num_diffusion_iters = opt.diffusion_timesteps    
    unet = create_model(
        image_size=opt.image_size,
        num_channels=opt.num_channels,
        num_res_blocks=opt.num_res_blocks,
        in_channels=opt.in_channels,
    ).to(device)
    model = DdbmEdmDenoiser(
        unet=unet,
        sigma_data=opt.sigma_data,
        sigma_min=opt.sigma_min,
        sigma_max=opt.sigma_max,
        rho=opt.rho,
        device=device,
    )
    print("Model Loaded!")
    
    if opt.resume:
        state_dict = torch.load(opt.resume_checkpoint, map_location='cuda')
        model.load_state_dict(state_dict)
        print("Pretrained Model Loaded")
            
    batch = next(iter(dataloader))
    print("batch.shape:", batch[0].shape, batch[1].shape)
    print("batch x range:", torch.max(batch[0]), torch.min(batch[0]))
    print("batch y range:", torch.max(batch[1]), torch.min(batch[1]))
    test_x, test_y = batch[0], batch[1]
    test_y = test_y.to(device)
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=opt.lr, weight_decay=1e-6
    )
    
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=opt.warmup_steps,
        num_training_steps=len(dataloader) * opt.epochs
    )
    
    sample_density = partial(rand_log_normal, loc=opt.sigma_sample_density_mean, scale=opt.sigma_sample_density_std)
    
    # TODO: fix wandb resume
    run = wandb.init(project = 'latent_ddbm', resume = opt.resume)
    config = run.config
    config.epochs = opt.epochs
    config.batchsize = opt.batchsize
    config.learning_rate = opt.lr 
    config.diffusion_timesteps = opt.diffusion_timesteps

    config.sigma_data = opt.sigma_data
    config.sigma_sample_density_mean = opt.sigma_sample_density_mean
    config.sigma_sample_density_std = opt.sigma_sample_density_std
    
    train()