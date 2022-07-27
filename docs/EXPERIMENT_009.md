# Experiment 9 - Rerun GAN  - sweep
To check if we can increase batch size using the smaller SN model

## Notes
- sweep: sweep_remote_grid_005

## Hyperparameters
- experimented with batch_size between 4 and 20


## Overall Summary
- GPU memory is now (GAN_Concat_Flex_011) hitting the 60% capacity with batch_size=4, up from 30% when running SN alone, but lower than the 80% capacity when using the previous version of the SN
- Consider using smaller number of sn channels to reduce model memory even more, or increase number of fcn blocks
- optimal batch size for this model is 8. This batch size uses on average 80% of memory capacity, and had a single spike to 97%. Could push for 10, but it might result in out of memory errors from time to time.
