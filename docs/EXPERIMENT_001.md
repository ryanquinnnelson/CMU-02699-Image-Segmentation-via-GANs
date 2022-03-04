# Experiment 1 - ZhengSN (zhang_sn_002)
I built the generator model described in the paper "Deep Adversarial Networks for Biomedical Image Segmentation Utilizing Unannotated Images" and ran training and validation on the dataset.

## Notes
- Downsampled images from 522 x 774 -> 224 x 332 to avoid running out of memory
- Models used 90% GPU Memory
- Set batch_size=4 to avoid running out of memory
- Optimize the following hyperparameters:
	- lr
	- scheduler_factor


## Hyperparameters
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

## Best Model
- Run name: zhang_sn_002_BEST
- Epoch: 30 
- train_loss: Min 0.404 
- val_loss: Min 0.4311
- val_acc: Max 0.8763
- iou_score: Max 0.1959


## Overall Summary
- Found that LR=0.001 slightly outperformed LR=0.0001
- Not enough information about scheduler_factor to report whether 0.5 was better than 0.75