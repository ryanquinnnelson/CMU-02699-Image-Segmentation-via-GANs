# Experiment 8 - Rerun ConcatenationFCN with optimal configurations
Selected the best hyperparameters from previous runs, now that analysis has been performed on them.

## Notes
- Differences with previous version
	- sn_block_depth: 1 -> 2
	- sn_first_layer_out_channels: 8 -> 64


## Best Model
- Run Name: GAN_Concat_010_baseline


## Overall Summary
- Previous version slightly (<2%) outperforms the new version
- New version uses 75% of the GPU memory of the previous version. If we use the new version, we can increase the batch size

