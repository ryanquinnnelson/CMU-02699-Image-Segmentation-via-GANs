# Experiment 010 - When to start GAN training - sweep
So far I've been training SN model a bit first, then turning on the GAN process. This sweep is trying to determine whether starting the GAN process at different epochs makes a differences on the overall outcome

## Notes
- sweep: sweep_remote_grid_006


## Hyperparameters
- To optimize:
	- gan_start_epoch: [1,20,40,60,80]


## Overall Summary
- with start_epoch=1, val_loss and val_acc were able to achieve almost as good of metrics as when start_epoch=40, even for a doubled batch size. val_iou_score was only able to achieve 0.088, but this may be because of the batch_size change from 4 to 8. Results need to be compared against a SN baseline without a GAN.
- compared against a baseline (GAN_Concat_Flex_014_baseline), modifying the start_epoch had little effect on val_loss, val_acc, or iou_score