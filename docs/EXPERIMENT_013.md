# Experiment 013 - alternative ConcatenationFCN architecture
Trying a model that uses fcn block depth of 2 instead of 1, which converts model from only using one size of channel for entire model into a model that uses an increasing number of channels per block

## Notes
- Run name: GAN_Concat_Flex_018
- Compare against baseline
- Had to decrease batch size to 4 to fit within GPU memory, should rerun baseline model with batch size of 4 to make comparison fair

## Hyperparameters
- Testing:
	- sn_num_fcn_blocks=5
	- sn_block_depth=2
	- sn_first_layer_out_channels=16
	- sn_block_pattern=double_run


## Overall Summary
- model with larger number of channels takes much longer to train (4x)
