# Experiment 012 - optimize discriminator LR - sweep
Try to optimize the value of en_lr

## Notes
- sweep: sweep_remote_grid_008,sweep_remote_grid_009

## Hyperparameters
- To optimize:
	- en_lr: [0.1,0.01,0.001,0.0001,0.00001]
- Second attempt:
	- en_lr: [0.01,0.001,0.0001]




## Overall summary
- learning rate of 0.1 is too high, causes val scores to dramatically decrease and training loss to jump around
- learning rate of 0.01-0.001 resulting in more oscillations than smaller learning rates, allows discriminator to be adversarial against generator
- perform test again in [0.01, 0.001, 0.0001] range on a larger number of epochs
- found that 0.001 and 0.0001 performed similarly. 0.0001 had more variation in discriminator accuracy longer into the training process.