# Experiment 015 - updated FlexVGG architecture
Added ability to have multiple CNN layers in the input block. Require that all FCN blocks start with pooling layer. Going to test against DAN EN model to see if designs still match.

## Notes
- Run name: GAN_Flex_001_ZhengEN, GAN_Flex_002_FlexVGG

## Overall Summary
- both models performed equally well regarding validation scores
- FlexVGG ended with increasing generator loss
- FlexVGG had more variation in discriminator accuracy for a longer period of time

