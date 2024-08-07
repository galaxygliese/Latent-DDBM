#-*- coding:utf-8 -*-
# Based on: https://github.com/crowsonkb/k-diffusion

import math
import torch
import numpy as np
from torch import nn
from tqdm.auto import trange
from typing import Union, List
from functools import partial
from enum import Enum
from .model import SongUNet

class SDEType(Enum):
    VP = "VP"
    VE = "VE"

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
    
def rand_log_normal(shape, loc=0., scale=1., device='cuda', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()    

def vp_logsnr(sigmas:torch.Tensor, beta_d, beta_min):
    sigmas = torch.as_tensor(sigmas)
    return - torch.log((0.5 * beta_d * (sigmas ** 2) + beta_min * sigmas).exp() - 1)
    
def vp_logs(sigmas:torch.Tensor, beta_d, beta_min):
    sigmas = torch.as_tensor(sigmas)
    return -0.25 * sigmas ** 2 * (beta_d) - 0.5 * sigmas * beta_min

def vp_snr_sqrt_reciprocal(sigmas:torch.Tensor, beta_d, beta_min):
    return (np.e ** (0.5 * beta_d * (sigmas ** 2) + beta_min * sigmas) - 1) ** 0.5

def vp_snr_sqrt_reciprocal_deriv(sigmas:torch.Tensor, beta_d, beta_min):
    return 0.5 * (beta_min + beta_d * sigmas) * (vp_snr_sqrt_reciprocal(sigmas, beta_d, beta_min) + 1 / vp_snr_sqrt_reciprocal(sigmas, beta_d, beta_min))

class DdbmEdmDenoiser(nn.Module):
    def __init__(
        self,
        unet: SongUNet,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: int = 7,
        c: float = 1,
        num_sampling_steps:int=40,
        cov_xy: float = 0.0,
        device: Union[int, str] = 'cuda',
        sde_type: SDEType = SDEType.VP,
        ):
        super().__init__()
        # NN model
        self.unet = unet
        
        # Scheduler parameters
        self.sigma_data = sigma_data
        self.sigma_data_end = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_sampling_steps = num_sampling_steps
        self.sigma_sample_density_mean = -1.2
        self.sigma_sample_density_std = 1.2
        
        # Scaling factor parameters
        self.c = c 
        self.cov_xy = cov_xy
        self.beta_d = 2
        self.beta_min = 0.1
        
        self.device = device
        self.sde_type = sde_type
        
    def get_sigmas(self, n:int):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return append_zero(sigmas).to(self.device)
    
    def sample_sigmas_uniform(self, x:torch.Tensor):
        """ Uniform schedule sampler """
        ramp = torch.randint(1, self.num_sampling_steps, (x.shape[0],), device=x.device) / (self.num_sampling_steps - 1)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas
    
    def sample_sigmas(self, x:torch.Tensor):
        """ Schedule sampler """        
        B = x.shape[0]
        sample_density = partial(rand_log_normal, loc=self.sigma_sample_density_mean, scale=self.sigma_sample_density_std)
        sigmas = sample_density([B], device=self.device)
        sigmas = torch.minimum(sigmas, torch.ones_like(sigmas)* self.sigma_max)
        return sigmas
    
    def get_snrs(self, sigmas:torch.Tensor):
        # sigmas : (B, )
        if self.sde_type == SDEType.VE:
            return sigmas ** -2
        elif self.sde_type == SDEType.VP:
            return vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            raise NotImplementedError
        
    def get_condition_weight(self, sigmas:torch.Tensor): # = a_t
        if self.sde_type == SDEType.VE:
            return sigmas**2 / self.sigma_max**2
        elif self.sde_type == SDEType.VP:
            logsnr_t = vp_logsnr(sigmas, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigmas, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)
            return (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        else:
            raise NotImplementedError
    
    def get_sample_weight(self, sigmas:torch.Tensor): # = b_t
        if self.sde_type == SDEType.VE:
            return 1 - sigmas**2 / self.sigma_max**2
        elif self.sde_type == SDEType.VP:
            logsnr_t = vp_logsnr(sigmas, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigmas, self.beta_d, self.beta_min)
            return - torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        else:
            raise NotImplementedError
    
    def get_ddbm_scalings(self, sigma:torch.Tensor):
        a_t = self.get_condition_weight(sigma)
        b_t = self.get_sample_weight(sigma)
        if self.sde_type == SDEType.VE:
            c_buff = a_t**2 * self.sigma_data**2 + b_t**2 * (self.sigma_data**2 + self.c*sigma**2) \
                    + 2 * a_t * b_t * self.cov_xy 
            c_in = 1 / (c_buff ** 0.5)
            c_skip = (b_t * self.sigma_data**2 + a_t*self.cov_xy) / c_buff 
            c_out = ((a_t**2) * (self.sigma_data**4 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * b_t * sigma**2)**0.5 * c_in
            return c_in, c_skip, c_out
        elif self.sde_type == SDEType.VP:
            logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()
            A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
            
            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy)/ A
            c_out = (a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * c_t )**0.5 * c_in
            return c_in, c_skip, c_out
        else:
            raise NotImplementedError
    
    def get_denoised(self, x_t:torch.Tensor, sigmas:torch.Tensor, **model_kwargs):
        c_in, c_skip, c_out = [
            append_dims(x_t, x_t.ndim) for x in self.get_ddbm_scalings(sigmas)
        ]
        c_noises = 1000 * sigmas.log() / 4
        # c_noises = sigmas.log() / 4
        F = self.unet(c_in * x_t, c_noises, **model_kwargs)
        D = x_t * c_skip + c_out * F
        return D
        
    def get_ddbm_sample(self, x_0: torch.Tensor, x_T:torch.Tensor, noise:torch.Tensor, sigmas:torch.Tensor):
        sigmas = append_dims(sigmas, x_0.ndim)
        a_t = self.get_condition_weight(sigmas)
        b_t = self.get_sample_weight(sigmas)
        mu_t = a_t * x_T + b_t * x_0 
        if self.sde_type == SDEType.VE:
            std_t = sigmas * torch.sqrt(b_t)
        elif self.sde_type == SDEType.VP:
            logsnr_t = vp_logsnr(sigmas, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigmas, self.beta_d, self.beta_min)
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
        else:
            raise NotImplementedError
        x_t = mu_t + std_t * noise
        return torch.clamp(x_t, min=-self.sigma_max, max=self.sigma_max)
    
    def get_loss_weightings(self, sigmas:torch.Tensor):
        a_t = self.get_condition_weight(sigmas)
        b_t = self.get_sample_weight(sigmas)
        if self.sde_type == SDEType.VE:
            c_buff = a_t**2 * self.sigma_data**2 + b_t**2 * (self.sigma_data**2 + self.c*sigmas**2) \
                    + 2 * a_t * b_t * self.cov_xy 
            weights = c_buff / ((a_t)**2 * (self.sigma_data**4 - self.cov_xy**2) + self.sigma_data**2 * self.c * b_t * sigmas**2)
            return weights
        elif self.sde_type == SDEType.VP:
            logsnr_t = vp_logsnr(sigmas, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigmas, self.beta_d, self.beta_min)
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()
            A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
            weights = A / (a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * c_t )
            return weights
        else:
            raise NotImplementedError
    
    def get_loss(self, x_start:torch.Tensor, x_T:torch.Tensor, model_kwargs=None):
        noise = torch.randn_like(x_start)
        sigmas = self.sample_sigmas(x_start)
        x_t = self.get_ddbm_sample(x_0=x_start, x_T=x_T, noise=noise, sigmas=sigmas)
        D = self.get_denoised(x_t, sigmas)
        
        lambdas = self.get_loss_weightings(sigmas)
        lambdas = append_dims(lambdas, x_start.ndim)
        loss = mean_flat(lambdas*(D - x_start)**2)
        return loss
    
    def get_dxdt(self, x_t:torch.Tensor, sigma_t:torch.Tensor, denoised_t:torch.Tensor, x_T:torch.Tensor, stochastic:bool=True, w:float=1):
        if self.sde_type == SDEType.VE:
            s = (denoised_t - x_t) / append_dims(sigma_t**2, x_t.ndim)
            h = (x_T - x_t) / (append_dims(torch.ones_like(sigma_t)*self.sigma_max**2, x_t.ndim) - append_dims(sigma_t**2, x_t.ndim))
            dxdt = -0.5 * (2*sigma_t)*s
            return dxdt
        elif self.sde_type == SDEType.VP:
            """Converts a denoiser output to a Karras ODE derivative."""
            a_t = self.get_condition_weight(sigma_t)
            b_t = self.get_sample_weight(sigma_t)
            mu_t = a_t * x_T + b_t * denoised_t 
            # std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
            
            
            # x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] )
            # x, denoised, x_T, std_t,logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv
            s = (1 + vp_snr_sqrt_reciprocal(sigma_t, self.beta_d, self.beta_min) ** 2).rsqrt()
            std_t = vp_snr_sqrt_reciprocal(sigma_t, self.beta_d, self.beta_min) * s
            s_t_deriv = -vp_snr_sqrt_reciprocal(sigma_t, self.beta_d, self.beta_min) * vp_snr_sqrt_reciprocal_deriv(sigma_t, self.beta_d, self.beta_min) * (s ** 3)
            sigma_t_deriv = vp_snr_sqrt_reciprocal_deriv(sigma_t, self.beta_d, self.beta_min)
            sigma_t_hat = vp_snr_sqrt_reciprocal(sigma_t, self.beta_d, self.beta_min)
            
            logsnr_t = vp_logsnr(sigma_t, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma_t, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)
            # logs = lambda t: -0.25 * t ** 2 * (self.beta_d) - 0.5 * t * self.beta_min
            # logsnr = lambda t :  - 2 * torch.log(vp_snr_sqrt_reciprocal(t, self.beta_d, self.beta_min))
            # logsnr_T = logsnr(torch.as_tensor(self.sigma_max))
            # logs_T = logs(torch.as_tensor(self.sigma_max))
            # logsnr_t = logsnr(torch.as_tensor(sigma_t))
            # logs_t = logs(torch.as_tensor(sigma_t))
            
            grad_logq = - (x_t - mu_t)/std_t**2 / (-torch.expm1(logsnr_T - logsnr_t))
            grad_logpxTlxt = -(x_t - torch.exp(logs_t-logs_T)*x_T) /std_t**2  / torch.expm1(logsnr_t - logsnr_T)
            f = s_t_deriv * (-logs_t).exp() * x_t
            gt2 = 2 * (logs_t).exp()**2 * sigma_t_hat * sigma_t_deriv 
            
            dxdt = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
            # grad_pxtlx0 = (denoised_t - x_t) / append_dims(sigma_t**2, x_t.ndim)
            # grad_pxTlxt = (x_T - x_t) / (append_dims(torch.ones_like(sigma_t)*self.sigma_max**2, x_t.ndim) - append_dims(sigma_t**2, x_t.ndim))
            # gt2 = 2*sigma_t
            # dxdt = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
            return dxdt 
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def sample(self, y:torch.Tensor, steps:int, guidance:float=1) -> torch.Tensor:
        """ DDBM Heun Sampler """
        sigmas = self.get_sigmas(steps)
        x_T = y
        x = y
        indices = range(len(sigmas) - 1)
        for i, index in enumerate(indices):
            sigma_t = sigmas[index].unsqueeze(0)
            D = self.get_denoised(x, sigma_t).clamp(-1, 1)
            dxdt = self.get_dxdt(x, sigma_t, D, x_T)
            dt = sigmas[index + 1].unsqueeze(0) - sigma_t
            if sigmas[index + 1] == 0:
                x = x + dxdt * dt
            else:
                x_heun = x + dxdt * dt 
                sigma_heun = sigmas[index + 1].unsqueeze(0)
                D_heun = self.get_denoised(x_heun, sigma_heun).clamp(-1, 1)
                dxdt_heun = self.get_dxdt(x_heun, sigma_heun, D_heun, x_T)
                x = x + (dxdt + dxdt_heun) / 2 * dt
        return x
        