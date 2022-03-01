# CMU-02699-Image-Segmentation-via-GANs

Spring 2022 Bioimage Informatics (Self-Study) project

## Summary
This project explores using a deep generative adversarial network (GAN) to perform semi-supervised image segmentation on the 2015 MICCAI Gland Challenge dataset.

The following generative model architectures are explored:
- ZhengSN
- ConcatenationFCN
- (future) SummationFCN
- (future) U-Net


The following discriminative model architectures are explored:
- ZhengEN
- FlexVGG
- (future) VGG16

## Run single run

Executes `octopus` for a single configuration file.

One time setup

```commandline
bash /path/to/octopus/bin/mount_drive
bash /path/to/octopus/bin/setup_wandb
```

For each run

```commandline
python /path/to/octopus/run_octopus.py --filename=/path/to/octopus/configs/remote/config001.txt
```

## Run sweep

wandb conducts a search over your hyperparameters. Set the configuration file within the sweep yaml file.

One time setup

```commandline
bash /path/to/octopus/bin/mount_drive
bash /path/to/octopus/bin/setup_wandb
```

For each sweep

```commandline
wandb sweep /path/to/octopus/sweeps/remote/sweep001.yaml

# execute wandb agent as specified by wandb
```

## configs

`octopus` is designed to handle ConfigParser configuration files. For this project, ConfigParser config files are
organized into local and remote configs.

## sweeps

Sweeps are a tool wandb uses to search over a hyperparameter space. For this project, sweep configuration files are
organized into local and remote sweeps.

