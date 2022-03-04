# Experiment 2 - ZhengSN + ZhengEN (zhang_sn_006)
I added the discriminator model and ran the GAN process.

## Notes
- Ran for 100 epochs

## Hyperparameters
- Fix the following hyperparameters:
	- sigma=0.1
	- sigma_weight=300




## Best Model
- Run Name: zhang_sn_006_BEST
- epoch: 100
- d_train_annotated_loss: 8.938
- d_train_unannotated_loss: 22.292
- d_train_loss: 0.6444
- g_train_loss: 0.3519
- val_loss: 0.4105
- val_acc: 0.8979
- iou_score: 0.2028


## Overall Summary
- discriminator loss oscillates until it stabilizes after 40 epochs
- discriminator d_train_loss keeps increasing with every epoch
- not the behavior I want to see with an adversarial process with increasing sigma
- no real improvement in generator loss/accuracy due to GAN process