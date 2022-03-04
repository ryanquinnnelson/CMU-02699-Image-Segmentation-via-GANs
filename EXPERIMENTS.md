
##

### Experiment 8 - GAN optimization 3
- Set LR of EN to 10% of LR of SN
- Remove sigma from discriminator loss calculations
- Train both SN and EN from epoch 1, rather than SN first, then SN + EN

### Experiment 9 - GAN optimization 4
- Use SN output as a mask on the EN input, rather than concatenating SN channels with input channels

### Experiment 10 - GAN optimization 5
- Optimize the following hyperparameters:
  - sigma
  - sigma_weight