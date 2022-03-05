# Experiment 5 - FlexVGG optimization - sweep
Fixing the SN model in place, perform hyperparameter tuning on the EN model.



## Notes
- Load SN model that has been trained for 40 epochs (GAN_Concat_Flex_006)
- sweep: sweep_remote_grid_004
- Ran into memory issues with smaller number of fcn blocks because it would result in a massive number of features when flattening to linear layers. Best result has a deeper model in which starting number of channels is set to minimize the number of features when flattening, and double_run block pattern ensures number of channels doesn't increase too quickly.
- Both EN and SN start with the same learning rate


## Hyperparameters
- Optimize the following hyperparameters:
	- en_num_fcn_blocks: [4,5,6]
	- en_depth_linear_block: [1,2,4]
	- en_first_linear_layer_out_features: [128,512,1024]
	- sigma: [0.1,0.5,1.0]
	- sigma_weight: [30,300]
- Fix the following hyperparameters:
	- num_workers=0
	- pin_memory=False
	- batch_size=4
	- sn_model_type=ConcatenationFCN
	- sn_num_fcn_blocks=5
	- sn_block_depth=2
	- sn_input_channels=3
	- sn_output_channels=2
	- sn_first_layer_out_channels=8
	- sn_block_pattern=double_run
	- sn_upsampling_pattern=last_three
	- en_model_type=FlexVGG
	- en_depth_fcn_block=2
	- en_first_layer_out_channels=64
	- en_fcn_block_pattern=double_run
	- en_linear_block_pattern=single_run
	- num_epochs = 100
	- sn_criterion=CrossEntropyLoss
	- en_criterion=BCELoss
	- optimizer_type=adam
	- lr=0.001
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

## Memory Issues
- GPU (14.76 GiB total capacity) runs out of memory for the following combinations (en_num_fcn_blocks, en_depth_linear_block, en_first_linear_layer_out_features):
	- (4, 1, 512) all 6
	- (5, 1, 512) all 6
	- (6, 1, 512) all 6
	- (4, 1, 1024) all 6
	- (5, 1, 1024) all 6
	- (6, 1, 1024) all 6
	- (4, 2, 512) all 6
	- (5, 2, 512) all 6
	- (6, 2, 512) all 6
	- (4, 2, 1024) all 6
	- (5, 2, 1024) all 6
	- (6, 2, 1024) all 6
	- (4, 4, 512) all 6
	- (5, 4, 512) all 6
	- (6, 4, 512) all 6
	- (4, 4, 1024) all 6
	- (5, 4, 1024) all 6
	- (6, 4, 1024) all 6


## Sections
(en_num_fcn_blocks, en_depth_linear_block, en_first_linear_layer_out_features, sigma, sigma_weight)
(4, 1, 128, X, Y)
X: sigma_weight of 0.1 outperforms 0.5 or 1, 0.1 much better for discriminator loss
Y: sigma_weight of 300 significantly better than 30


(5, 1, 128, X, Y)
5 blocks just as good as 4 blocks, uses roughly the same memory


(6, 1, 128, X, Y)
5 blocks just as good as 4 blocks, uses roughly the same memory



(4, 2, 128, X, Y)
linear depth of 2 performs about as well as 1

(5, 2, 128, X, Y)
performs as well as 4

(6, 2, 128, X, Y)
performs as well as 4


(4, 4, 128, X, Y)
performs worse than 2 or 1

(5, 4, 128, X, Y)
performs as well as (4, 1,...)


(6, 4, 128, X, Y)
performs much worse

## Best Model
- Run Name: GAN_Concat_Flex_007_BEST
- Epoch: 100
- d_train_loss_unannotated: 1.222
- d_train_loss_annotated: 0.3543
- d_train_loss: 0.6832
- val_iou_score: 0.2012
- val_acc: 0.8922
- val_loss: 0.4163



## overall summary
- one linear layer is sufficient
- four fcn blocks is sufficient
- requiring more than 128 linear features resulted in out of memory, could try 256
- **todo try leaving all channels the same for EN too
- **try even smaller values for sigma_weight
- **try even smaller values for sigma
- Best configurations
	- en_num_fcn_blocks: 4
	- en_depth_fcn_block=2
	- en_first_layer_out_channels=64
	- en_fcn_block_pattern=double_run
	- en_depth_linear_block: 1
	- en_first_linear_layer_out_features: 128
	- en_linear_block_pattern=single_run


