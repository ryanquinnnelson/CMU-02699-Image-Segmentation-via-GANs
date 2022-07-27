# Experiment 4 - ConcatenationFCN optimization 2 - sweep
After identifying good values for most of the hyperparameters, I explored different optimizers and learning rates.

## Notes
- sweep: sweep_remote_grid_002

## Hyperparameters
- Optimize the following hyperparameters:
	- lr: [0.1,0.01,0.001,0.0001,0.00001]
	- optimizer_type: [adam,sgd]
	- sn_block_depth: [1,2]
	- sn_first_layer_out_channels: [8, 16, 32]
	- sn_block_pattern: ["single_run","double_run"]


- Fix the following hyperparameters:
	- num_workers=0
	- pin_memory=False
	- batch_size=4
	- sn_model_type=ConcatenationFCN
	- sn_num_fcn_blocks=5
	- sn_input_channels=3
	- sn_output_channels=2
	- sn_upsampling_pattern=last_three
	- en_model_type=FlexVGG
	- num_epochs = 40
	- sn_criterion=CrossEntropyLoss
	- scheduler_type=ReduceLROnPlateau
	- scheduler_factor=0.75
	- scheduler_patience=5
	- scheduler_mode=min
	- scheduler_verbose=True
	- scheduler_plateau_metric=val_loss
	- scheduler_min_lr=0.00001
	- use_gan = False

## Memory Issues
- GPU (14.76 GiB total capacity) runs out of memory for the following combinations (num_fcn_blocks, block_depth, first_layer_out_channels, block_pattern):
	- (5, 1, 32, single_run) all 10 failed
	- (5, 2, 32, single_run) *
	- (5, 2, 32, double_run) all 10 failed
	- (5, 2, 16, single_run) all 10 failed
	- (5, 2, 8, single_run) all 10 failed


## Sections
(lr, optimizer, num_fcn_blocks, block_depth, first_layer_out_channels, block_pattern)
(0.1,adam, 5, 1, X, single_run)
8 vs 16 channels 
8 channels resulted in no learning, 16 better


(0.1,adam, 5, 1, X, double_run)
8 vs 16, both resulted in learning, 8 better
32 channels double_run performed best - same number of channels for all layers
16, 8 entire model performed well too; 8 performed best in some cases

** todo build deeper model with same channels per layer


(0.1,sgd, 5, 1, X, single_run)


(0.1,sgd, 5, 1, X, double_run)
sgd performs as well as adam
less variance in scores


(0.01,adam, 5, 1, X, double_run)
Smaller LR didn't improve results


(0.01,sgd, 5, 1, X, double_run)
worse performance than adam


(0.001,adam, 5, 1, X, double_run)
learning rate resulted in better performance across the board than 0.01 or 0.1

(0.001,adam, 5, 2, X, double_run)
double_run, block depth of 2 resulted in top performance of sweep
8 or 16 channels
but uses vastly more memory (2x-3x that of block_depth=1)
makes models infeasible for GAN without running out of memory




(0.001,sgd, 5, 2, X, double_run)
adam vastly outperforms sgd



(0.0001,adam, 5, 1, X, double_run)
outperformed by 0.001 learning rate


## Best Model
- Run Name: FlexVGG-003-BEST
- Epoch: 40
- val_acc: 0.8327
- val_loss: 0.4734
- val_iou_score: 0.1766




## overall takeaways
- adam outperforms sgd
- lr=0.001 outperforms other rates
- models built entirely of a single channel size perform really well and use little memory
