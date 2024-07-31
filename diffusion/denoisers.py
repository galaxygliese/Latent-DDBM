#-*- coding:utf-8 -*-
# Based on: https://github.com/crowsonkb/k-diffusion

import math
import torch
from torch import nn
from tqdm.auto import trange
from typing import Union, List
from .model import SongUNet

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cuda'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1.):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, mask = None, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * append_dims(sigma, input.ndim)
        if mask is not None:
            noised_input[mask] = input[mask]
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        loss = (model_output - target).pow(2)
        if mask is not None:
            loss_mask = ~mask
            loss = loss * loss_mask.type(loss.dtype)
        return loss.flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, sigma, **kwargs) * c_out + input * c_skip
    
class DdbmEdmDenoiser(nn.Module):
    def __init__(
        self,
        unet: SongUNet,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: int = 7,
        c: float = 1,
        cov_xy: float = 0.0,
        device: Union[int, str] = 'cuda',
        ):
        # NN model
        self.unet = unet
        
        # Scheduler parameters
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
        # Scaling factor parameters
        self.c = c 
        self.cov_xy = cov_xy
        
        self.device = device
        
    def get_sigmas(self, n:int):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return append_zero(sigmas).to(self.device)
    
    def get_snrs(self, sigmas:torch.Tensor):
        # sigmas : (B, )
        return sigmas ** -2
    
    def get_ddbm_scalings(self, sigma:torch.Tensor):
        a_t = sigma**2 / self.sigma_max**2
        b_t = (1 - sigma**2 / self.sigma_max**2)
        c_buff = a_t**2 * self.sigma_data**2 + b_t**2 * (self.sigma_data**2 + self.c*sigma**2) \
                + 2 * a_t * b_t * self.cov_xy 
        c_in = 1 / (c_buff ** 0.5)
        c_skip = (b_t * self.sigma_data**2 + a_t*self.cov_xy) / c_buff 
        c_out = ((a_t**2) * (self.sigma_data**4 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * b_t * sigma**2)**0.5 * c_in
        return c_in, c_skip, c_out
    
    def get_denoised(self, x_t:torch.Tensor, sigmas:torch.Tensor, **model_kwargs):
        c_in, c_skip, c_out = [
            append_dims(x_t, x_t.ndim) for x in self.get_ddbm_scalings(sigmas)
        ]
        c_noises = 1000 * sigmas.log() / 4
        F = self.unet(c_in * x_t, c_noises, **model_kwargs)
        D = x_t * c_skip + c_out * F
        return D
    
    def get_ddbm_sample(self, x_0: torch.Tensor, x_T:torch.Tensor, n:torch.Tensor, sigmas:torch.Tensor):
        a_t = sigmas**2 / self.sigma_max**2
        b_t = (1 - sigmas**2 / self.sigma_max**2)
        mu_t = a_t * x_T + b_t * x_0 
        std_t = sigmas * torch.sqrt(b_t)
        return mu_t + std_t * n
    
    def get_loss_weightings(self, sigmas:torch.Tensor):
        a_t = sigmas**2 / self.sigma_max**2
        b_t = (1 - sigmas**2 / self.sigma_max**2)
        c_buff = a_t**2 * self.sigma_data**2 + b_t**2 * (self.sigma_data**2 + self.c*sigmas**2) \
                + 2 * a_t * b_t * self.cov_xy 
        weights = c_buff / ((a_t)**2 * (self.sigma_data**4 - self.cov_xy**2) + self.sigma_data**2 * self.c * b_t * sigmas**2)
        return weights
    
    def get_loss(self, x_start:torch.Tensor, x_T:torch.Tensor, sigmas:torch.Tensor, model_kwargs=None):
        noise = torch.randn_like(x_start)
        x_t = self.get_ddbm_sample(x_start, x_T, sigmas, n=noise)
        D = self.get_denoised(x_t, sigmas)
        lamgdas = self.get_loss_weightings(sigmas)
        lambdas = append_dims(lambdas, x_start.ndim)
        loss = mean_flat((D - x_start)**2)
        return loss
        