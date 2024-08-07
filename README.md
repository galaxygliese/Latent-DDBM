# Latent-DDBM

Simplified implementation of DDBM.

### Train VP models

```
python3 train_full_resolution.py --dataset_path /path/to/datas -e 500 -b 16 --save-per-epoch 50 --sigma_max 1 --sigma_min 0
```

### Train VE models

:construction: [WIP] unstable.

Appropriate hyperparameters should be chosen.

# Citations

```
@misc{zhou2023denoisingdiffusionbridgemodels,
      title={Denoising Diffusion Bridge Models}, 
      author={Linqi Zhou and Aaron Lou and Samar Khanna and Stefano Ermon},
      year={2023},
      eprint={2309.16948},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.16948}, 
}
```
