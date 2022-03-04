# Experiments
This document tracks the experiments I performed for this project.


### Experiment 1 - ZhengSN (zhang_sn_002)
I built the generator model described in the paper "Deep Adversarial Networks for Biomedical Image Segmentation Utilizing Unannotated Images" and ran training and validation on the dataset.

- Downsampled images from 522 x 774 -> 224 x 332 to avoid running out of memory
- Models used 90% GPU Memory
- Set batch_size=4 to avoid running out of memory
- Optimize the following hyperparameters:
	- lr
	- scheduler_factor


- Found that LR=0.001 slightly outperformed LR=0.0001
- Not enough information about scheduler_factor to report whether 0.5 was better than 0.75


- Fix the following hyperparameters:
	- num_workers=8
	- pin_memory=True
	- batch_size=2
	- criterion_type=CrossEntropyLoss
	- num_epochs=30
	- weight_decay=0.000005
	- optimizer_type=adam
	- scheduler_type=ReduceLROnPlateau
	- scheduler_patience=5
	- scheduler_mode=min
	- scheduler_verbose=True
	- scheduler_plateau_metric=val_loss

- Measured the following metrics:
	- train_loss: Min 0.404 at epoch 30 
	- val_loss: Min 0.4311 at epoch 30
	- val_acc: Max 0.8763 at epoch 30
	- iou_score: Max 0.1959 at epoch 30


### Experiment 2 - ZhengSN + ZhengEN (zhang_sn_006)
I added the discriminator model and ran the GAN process.

- Ran for 100 epochs
- discriminator loss oscillates until it stabilizes after 40 epochs
- discriminator d_train_loss keeps increasing with every epoch
- not the behavior I want to see with an adversarial process with increasing sigma
- no real improvement in generator loss/accuracy due to GAN process

- Fix the following hyperparameters:
	- sigma=0.1
	- sigma_weight=300


- Measured the following metrics:
	- d_train_annotated_loss: 8.938 at epoch 100
	- d_train_unannotated_loss: 22.292 at epoch 100
	- d_train_loss: 0.6444 at epoch 100
	- g_train_loss: 0.3519 at epoch 100
	- val_loss: 0.4105 at epoch 100
	- val_acc: 0.8979 at epoch 100
	- iou_score: 0.2028 at epoch 100



### Experiment 3 - ConcatenationFCN optimization 1 - sweep
I rebuilt the framework, added a flexible version of the paper's SN model, and I turned off the discriminator while optimizing the SN.

- Tried writing every channel hyperparameter set as a dictionary (flexgan_sn_001, 002, 003), got to be too hard to sweep
- Redesigned `octopus` to allow for wandb sweeps
- Ran sweep (sweep_remote_grid_001)
- GPU (14.76 GiB total capacity) runs out of memory for the following combinations (num_fcn_blocks, block_depth, first_layer_out_channels, block_pattern):
	- (5,1,64,single_run)
	- (6,1,64,single_run)
	- (6,1,32,single_run)

	- (3,2,64,single_run)
	- (4,2,64,single_run)
	- (5,2,64,single_run)
	- (6,2,64,single_run)

	- (4,2,32,single_run)
	- (5,2,32,single_run)*
	- (6,2,32,single_run)

	- (5,2,64,double_run)
	- (6,2,64,double_run)

	- (6,2,32,double_run)

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


overall takeaways
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




### Experiment 4 - ConcatenationFCN optimization 2 - sweep
After identifying good values for most of the hyperparameters, I explored different optimizers and learning rates.
- sweep: sweep_remote_grid_002
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


- GPU (14.76 GiB total capacity) runs out of memory for the following combinations (num_fcn_blocks, block_depth, first_layer_out_channels, block_pattern):
	- (5, 1, 32, single_run) all 10 failed
	- (5, 2, 32, single_run) *
	- (5, 2, 32, double_run) all 10 failed
	- (5, 2, 16, single_run) all 10 failed
	- (5, 2, 8, single_run) all 10 failed

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





overall takeaways
- adam outperforms sgd
- lr=0.001 outperforms other rates
- models built entirely of a single channel size perform really well and use little memory

















### Experiment 5 - FlexVGG optimization - sweep
Fixing the SN model in place, perform hyperparameter tuning on the EN model.
- Load SN model that has been trained for 40 epochs (GAN_Concat_Flex_006)
- sweep: sweep_remote_grid_004
- Ran into memory issues with smaller number of fcn blocks because it would result in a massive number of features when flattening to linear layers. Best result has a deeper model in which starting number of channels is set to minimize the number of features when flattening, and double_run block pattern ensures number of channels doesn't increase too quickly.
- Both EN and SN start with the same learning rate

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


overall summary
- one linear layer is sufficient
- four fcn blocks is sufficient
- requiring more than 128 linear features resulted in out of memory, could try 256
- **todo try leaving all channels the same for EN too
- **try even smaller values for sigma_weight
- **try even smaller values for sigma







### Experiment 6 - GAN optimization 1 
Try controlling learning rate for EN and SN differently. Set EN learning rate at 10% of initial SN rate to start. 
- (GAN_Concat_Flex_008)
- Load SN model that has been trained for 40 epochs (GAN_Concat_Flex_006)
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


overall summary








### Experiment 6 - GAN optimization 2
- Set LR of EN to 10% of LR of SN
- Remove sigma from discriminator loss calculations

### Experiment 7 - GAN optimization 3
- Set LR of EN to 10% of LR of SN
- Remove sigma from discriminator loss calculations
- Train both SN and EN from epoch 1, rather than SN first, then SN + EN

### Experiment 8 - GAN optimization 4
- Use SN output as a mask on the EN input, rather than concatenating SN channels with input channels

### Experiment 9 - GAN optimization 5
- Optimize the following hyperparameters:
  - sigma
  - sigma_weight