#-*- coding:utf-8 -*-

from diffusion import create_model, create_unet_model, DdbmEdmDenoiser
from diffusion.fp16_util import MixedPrecisionTrainer
from dataset import FolderDataset, FolderPairDataset, expand2square
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
from torchvision import transforms as T
from tqdm.auto import tqdm

import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
import dill as pickle
import numpy as np
import argparse
import random
import torch.nn as nn
import torch
import wandb
import copy
import os 

# deepspeed
import deepspeed
from deepspeed.accelerator import get_accelerator


parser = argparse.ArgumentParser()

# General options
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batchsize', type=int, default=8)
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--export_folder', type=str, default="./checkpoints")
parser.add_argument('--warmup_steps', type=int, default=50)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=3)
parser.add_argument('--save-per-epoch', type=int, default=50)
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('--ema-rate', type=float, default=0.99)
parser.add_argument('--half', action='store_true')

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
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--deepspeed_config', type=str, default="./configs/ds_config.json")
opt = parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# LATENT_DIM = opt.image_size // 8
seed_everything(opt.seed)

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
        nimage, path = model.sample(y, steps=num_diffusion_iters)
        
        nimage = ((nimage + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        imgs = nimage.detach().to('cpu')
        
        img = make_grid(imgs)
        img = transforms.functional.to_pil_image(img)
        # (B, 3, H, W)
    img = Image.fromarray(np.asarray(img))
    img.save(export_name)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def update_ema(mp_trainer:MixedPrecisionTrainer, ema_params, ema_rate:float=0.99):
    for targ, src in zip(ema_params, mp_trainer.master_params):
        targ.detach().mul_(ema_rate).add_(src, alpha=1 - ema_rate)

def anneal_lr(step:int, optimizer, lr_anneal_steps:float = 0):
    if lr_anneal_steps == 0:
        return
    frac_done = (step) / lr_anneal_steps
    lr = opt.lr * (1 - frac_done)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(args, run):
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()
    
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

    num_diffusion_iters = opt.diffusion_timesteps   
    attention_type = 'flash' if opt.half else 'vanilla' 
    unet = create_unet_model(
        image_size=opt.image_size,
        num_channels=opt.num_channels,
        num_res_blocks=opt.num_res_blocks,
        in_channels=opt.in_channels,
        use_fp16=opt.half,
        attention_type=attention_type
    )
    print("UNET dtype:", unet.dtype)
    if opt.half:
        mp_trainer = MixedPrecisionTrainer(
            model=unet,
            use_fp16=True,
            fp16_scale_growth=1e-3,
        )
    model = DdbmEdmDenoiser(
        unet=unet,
        sigma_data=opt.sigma_data,
        sigma_min=opt.sigma_min,
        sigma_max=opt.sigma_max,
        rho=opt.rho,
    )
    
    if opt.half:
        ema_params = copy.deepcopy(mp_trainer.master_params)
    print("Model Loaded!")
    
    if opt.resume:
        state_dict = torch.load(opt.resume_checkpoint, map_location='cuda')
        model.load_state_dict(state_dict)
        
        # TODO: load EMA params
        print("Pretrained Model Loaded")
            
    
    # Define the network with DeepSpeed.
    print(">", args)
    model_engine, optimizer , trainloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset
    )
    
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    rank = dist.get_rank()
    device = rank
    step = 0
    
    if rank == 0:
        batch = dataset[0]
        print("batch.shape:", batch[0].shape, batch[1].shape)
        print("batch x range:", torch.max(batch[0]), torch.min(batch[0]))
        print("batch y range:", torch.max(batch[1]), torch.min(batch[1]))
        test_y = torch.cat([batch[1].unsqueeze(0)]*opt.generate_batchsize, 0).to(device)
        print("test y:", test_y.shape)
    
    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half
    
    model_engine.train()
    global_step = 0
    
    with tqdm(range(opt.resume_epochs, opt.epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            model.unet.train()
            # batch loop
            with tqdm(trainloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    global_step += 1
                    # data normalized in dataset
                    # device transfer
                    x, y = nbatch[0].to(device), nbatch[1].to(device)
                    B = x.shape[0]
                    
                    # sample a diffusion iteration for each data point
                    with torch.cuda.amp.autocast(cache_enabled=False):
                        loss = model_engine(x_start=x, x_T=y)

                    # optimize
                    model_engine.backward(loss)
                    model_engine.step()
                    
                    if opt.half:
                        took_step = mp_trainer.optimize(optimizer)
                        if took_step:
                            update_ema(mp_trainer, ema_params, ema_rate=opt.ema_rate)
                    # anneal_lr(step, optimizer)
                    step += 1

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                if rank == 0:
                    run.log({"train-loss": np.mean(epoch_loss)})
                tglobal.set_postfix(loss=np.mean(epoch_loss))   
            
                if (epoch_idx + 1) % opt.save_per_epoch == 0 and rank == 0:
                    # torch.save(model.unet.module.state_dict(), f'{opt.export_folder}/training-3dddbm-edm-latent-diffusion-epoch{epoch_idx+1}'+'.pt')
                    model_engine.save_checkpoint(save_dir=opt.export_folder, tag=global_step)

                    # TODO: generate with ema model?
                    # generate(
                    #     model=model,
                    #     y=test_y,
                    #     num_diffusion_iters=num_diffusion_iters,
                    #     export_name=f"{opt.export_folder}/epoch{epoch_idx+1}.nii.gz",
                    #     # sample_num=4
                    # )
    model_engine.save_checkpoint(save_dir=opt.export_folder, tag=global_step)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    
    # TODO: fix wandb resume
    run = wandb.init(project = '3dddbm', resume = opt.resume)
    config = run.config
    config.epochs = opt.epochs
    config.batchsize = opt.batchsize
    config.learning_rate = opt.lr 
    config.diffusion_timesteps = opt.diffusion_timesteps

    config.sigma_data = opt.sigma_data
    config.sigma_sample_density_mean = opt.sigma_sample_density_mean
    config.sigma_sample_density_std = opt.sigma_sample_density_std
    

    # mp.spawn(train, args=(world_size, run), nprocs=world_size, join=True)
    train(opt, run)