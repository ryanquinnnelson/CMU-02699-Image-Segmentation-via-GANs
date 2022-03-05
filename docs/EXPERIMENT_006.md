# Experiment 6 - GAN optimization 1 
Try controlling learning rate for EN and SN differently. Set EN learning rate at 10% of initial SN rate to start. 

## Notes
- (GAN_Concat_Flex_008)
- Load SN model that has been trained for 40 epochs (GAN_Concat_Flex_006)

## Hyperparameters
- Set LR of EN to 10% of LR of SN
	- sn_lr=0.001
	- en_lr=0.0001
- Fix the following hyperparameters:
	- sn_model_type=ConcatenationFCN
	- sn_num_fcn_blocks=5
	- sn_block_depth=2
	- sn_input_channels=3
	- sn_output_channels=2
	- sn_first_layer_out_channels=8
	- sn_block_pattern=double_run
	- sn_upsampling_pattern=last_three
	- en_model_type=FlexVGG
	- en_num_fcn_blocks=5
	- en_depth_fcn_block=2
	- en_input_channels=4
	- en_first_layer_out_channels=64
	- en_fcn_block_pattern=double_run
	- en_depth_linear_block=1
	- en_linear_block_pattern=single_run
	- en_first_linear_layer_out_features=128
	- en_out_features=1
	- num_epochs = 100
	- sn_criterion=CrossEntropyLoss
	- en_criterion=BCELoss
	- optimizer_type=adam
	- scheduler_type=ReduceLROnPlateau
	- scheduler_factor=0.8
	- scheduler_patience=10
	- scheduler_mode=min
	- scheduler_verbose=True
	- scheduler_plateau_metric=val_loss
	- scheduler_min_lr=0.00001
	- use_gan = True
	- sigma=0.1
	- sigma_weight=300
	- gan_start_epoch=40

## Best model
- Run Name: GAN_Concat_Flex_008_BEST
- Epoch: 100


## overall summary
- differential learning rates didn't improve results for models
- generator loss decreased steadily after GAN start, but plateaued and reversed around epoch 80
- validation metrics all were about the same as before
