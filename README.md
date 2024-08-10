# Latent-DDBM

Simplified implementation of DDBM. 

Original Repogitory : [github](https://github.com/alexzhou907/DDBM)

### VE models

training code:

```
python3 train_full_resolution.py --dataset_path /path/to/datas -e 500 -b 16 --save-per-epoch 50 --sigma_max 1 --sigma_min 0
```

sampling code:

```
python3 sample.py --weight_path /path/to/weight_file --dataset_path /path/to/test_datas
```

input `x_T`:

<image src="./assets/sample-0.jpg" />

generated result `x_0` (epoch=1500):

<image src="./assets/sample-80.jpg" />


### VP models

:construction: [WIP] (not debugged).


### Latent models

:construction: [WIP] (developing).

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
