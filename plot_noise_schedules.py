#-*- coding:utf-8 -*-

from diffusion import create_model, DdbmEdmDenoiser
from dataset import FolderDataset, FolderPairDataset, expand2square
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
from torchvision import transforms as T

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()

# General options
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batchsize', type=int, default=8)
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training
parser.add_argument('--ema_power', type=float, default=0.75)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=50)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('-d', '--device', type=int, default=0)

# Dataset options
parser.add_argument('--dataset_path', type=str, default="./datas/val")
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
opt = parser.parse_args()

device = opt.device

def main():
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
    
    image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Lambda(lambda img: expand2square(img)),
        T.Resize(opt.image_size),
        # T.CenterCrop(opt.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    dataset = FolderPairDataset(
        folder_path=opt.dataset_path,
        image_size=opt.image_size,
        transform=image_transform
    )
    
    sample = dataset[1]
    image = sample[0]
    image = 0.5*(image + 1).cpu().permute(1,2,0).data.numpy()
    plt.imshow(image)
    plt.savefig("plots/sample.png")
    
    n = 20
    sigmas = model.get_sigmas(n) # sigma_max to sigma_min
    x_start = sample[0].to(device)
    x_T = sample[1].to(device)
    for i, sigma in enumerate(sigmas):
        noise = torch.randn_like(x_start)
        x_t = model.get_ddbm_sample(x_0=x_start, x_T=x_T, noise=noise, sigmas=sigma)
        im = 0.5*(x_t + 1).cpu().permute(1,2,0).data.numpy()
        print(">", sigma)
        plt.imshow(im)
        plt.savefig("plots/noise-step-{:2f}.png".format(i/n))
    print('Done!')
    
if __name__ == '__main__':
    main()