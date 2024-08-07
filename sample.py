#-*- coding:utf-8 -*-

from diffusion import create_model, DdbmEdmDenoiser
from dataset import FolderDataset, FolderPairDataset, expand2square
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
from torchvision import transforms as T

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()

# General options
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=50)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-w', '--weight_path', type=str, default="./checkpoints/")
parser.add_argument('--sample_num', type=int, default=4)

# Dataset options
parser.add_argument('--dataset_path', type=str, default="./datas/val")
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)

# Karras (EDM) options
parser.add_argument('--sigma_data', type=float, default=0.5)
parser.add_argument('--sigma_sample_density_mean', type=float, default=-1.2)
parser.add_argument('--sigma_sample_density_std', type=float, default=1.2)
parser.add_argument('--sigma_max', type=float, default=1)
parser.add_argument('--sigma_min', type=float, default=0.0001)
parser.add_argument('--rho', type=float, default=7.0)
opt = parser.parse_args()

device = opt.device

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
        print(">>", torch.max(imgs), torch.min(imgs))
        imgs = (imgs*255).clip(0, 255)
        
        img = make_grid(imgs)
        img = transforms.functional.to_pil_image(img)
        # (B, 3, H, W)
    img = Image.fromarray(np.asarray(img))
    img.save(export_name)

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
    model.load_state_dict(torch.load(opt.weight_path))
    model.eval()
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
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.sample_num,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True, 
        # don't kill worker process afte each epoch
        persistent_workers=True 
    )
    
    batch = next(iter(dataloader))
    print("batch.shape:", batch[0].shape, batch[1].shape)
    print("batch x range:", torch.max(batch[0]), torch.min(batch[0]))
    print("batch y range:", torch.max(batch[1]), torch.min(batch[1]))
    test_x, test_y = batch[0], batch[1]
    test_y = test_y.to(device)
    
    generate(
        model=model,
        y=test_y,
        num_diffusion_iters=opt.diffusion_timesteps,
        export_name=f"plots/sample.png",
    )
    print('Done!')
    
if __name__ == '__main__':
    main()