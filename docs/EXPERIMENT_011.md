# Experiment 011 - weights between models - sweep 
Try to optimize the value of sigma and sigma_weight.

## Notes
- sweep: sweep_remote_grid_007

## Hyperparameters
- To optimize:
	- sigma: [0.05,0.1,0.2,0.5,0.8,1.0]
	- sigma_weight: [100,300,500]

## Sections
(0.05,X)
- Smaller sigma_weight enables discriminator to correct its mistakes more quickly, has little effect on val scores or val loss due to small sigma

(Y,100)
- Larger sigma results in smaller discriminator loss, no oscillations, but a steady decrease. Has little effect on val scores or val loss. Larger sigma results in higher starting loss for generator in training, but resulting loss matches that of lower sigma values.


(Y, 300)
- Little effect on val scores
- Smaller sigma values result in smaller discriminator training loss

(Y, 500)
- sigma_weight is too small to have an impact over a short term



## Overall summary
- smaller sigma and sigma_weight values are better for val_iou_score, smaller sigma matters a lot more; same for val_acc
- smaller sigma and larger sigma_weight values are better for val_loss
- sigma of 0.05 or 0.1 is best
- sigma_weight is less clear, but smaller values mean more sigma is added every time. 100 or 300 is probably okay.