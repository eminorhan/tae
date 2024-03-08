## Discrete Autoencoder (DAE)

A new simple discrete autoencoder model.

- Encoder and decoder are both ViT models.
- Output of encoder at each patch is a one-hot vector with size `vocab_size`.
- Gradient is estimated with the straight-through estimator at this discretization step.
- The skeleton of the code is recycled from Facebook's [MAE](https://github.com/facebookresearch/mae) repository with several simplifications.