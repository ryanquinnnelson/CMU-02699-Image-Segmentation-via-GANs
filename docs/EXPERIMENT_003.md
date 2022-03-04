# Experiment 3 - ConcatenationFCN optimization 1 - sweep
I rebuilt the framework, added a flexible version of the paper's SN model, and I turned off the discriminator while optimizing the SN.

## Notes
- Tried writing every channel hyperparameter set as a dictionary (flexgan_sn_001, 002, 003), got to be too hard to sweep
- Redesigned `octopus` to allow for wandb sweeps
- Ran sweep (sweep_remote_grid_001)

## Hyperparameters
- Optimize the following hyperparameters:
	- sn_num_fcn_blocks: [3,4,5,6]
	- sn_block_depth: [1,2]
	- sn_first_layer_out_channels: [64,32]
	- sn_block_pattern: ["single_run", "double_run"]


- Fix the following hyperparameters:
	- num_workers=8
	- pin_memory=True
	- batch_size=2
	- sn_criterion_type=CrossEntropyLoss
	- en_criterion=BCELoss
	- num_epochs=20
	- optimizer_type=adam
	- lr=0.002
	- scheduler_type=ReduceLROnPlateau
	- scheduler_factor=0.75
	- scheduler_patience=5
	- scheduler_mode=min
	- scheduler_verbose=True
	- scheduler_plateau_metric=val_loss
	- scheduler_min_lr=0.00001
	- sn_upsampling_pattern=last_three

## Memory Issues
- GPU (14.76 GiB total capacity) runs out of memory for the following combinations (num_fcn_blocks, block_depth, first_layer_out_channels, block_pattern):
	- (5,1,64,single_run)
	- (6,1,64,single_run)
	- (6,1,32,single_run)
	- (3,2,64,single_run)
	- (4,2,64,single_run)
	- (5,2,64,single_run)
	- (6,2,64,single_run)
	- (4,2,32,single_run)
	- (5,2,32,single_run)
	- (6,2,32,single_run)
	- (5,2,64,double_run)
	- (6,2,64,double_run)
	- (6,2,32,double_run)

## Run sections
(X,1,64,single_run)
- Outcome is approx. the same between 3,4 fcn_blocks, all other hyp. held constant (X,1,64,single_run). 3 fcn_blocks had less swinging up and down in scores. 4 fcn blocks uses much more memory, almost twice as much. 5 and 6 fcn blocks run out of memory.

(X,1,32,single_run)
- Outcome improves when reducing first_layer_out_channels to 32, model uses memory based on fcn_blocks
- 5 blocks has the best scores overall, most steady improvement
- very close to memory limit
- **try 16 and 8 first_layer_out_channels next with varying fcn_block


(X,2,64,single_run)
- All fcn_block fail due to out of memory issues

(X,2,32,single_run)
- doubling block depth slightly reduces scores, but doesn't really hurt
- doubling block depth significantly increases memory usage
- block depth doesn't help
- very close to memory limit


(X,1,64,double_run)
- more blocks is better during double run
- uses a very small amount of memory (20%), which will work well with GAN

(X,1,32,double_run)
- switching to 32 out channels results in similar performance as 64 for larger number of fcn_blocks
- uses slightly less memory than 64

(X,2,64,double_run)
- performs similar to block_deph=1

(X,2,32,double_run)
- more fcn_blocks performs better
- uses almost all GPU memory with 5 blocks, 4 blocks still reasonable memory usage

## Best model
- Run Name: FCNOnly-001_BEST
- Epoch: 15
- val_iou_score: 0.3411
- val_loss: 0.5121
- val_acc: 0.788


## overall takeaways
- smaller batch size resulted in better performance. but we need a min batch of 4 for the gan to work.
- smaller number of out channels to start is better, allows for deeper number of fcn blocks
- double run doesn't seem to perform better but reduces memory usage
- 5 fcn blocks seems to be max for memory
- best combinations:
	- (5,1,32,single_run)
	- (5,2,32,double_run)
	- (3,2,64,double_run)*runner-up, 1/2 memory of best
	- (5,1,64,double_run)*runner-up, 1/4 memory of best, model with all 64 channel layers, performed extremely well
	- (6,1,64,double_run)*runner-up, same as 5,1,64... but smoother learning and scores

- same number of channels across the board seems just as good as larger number each time, can allow for very deep models, larger number of fcn_blocks is better
-**try very deep models with same number of channels for each fcn_block and more than 6 fcn_blocks





