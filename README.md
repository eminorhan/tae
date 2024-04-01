## Transformer Autoencoder (TAE)

A new simple transformer-based autoencoder model.

- Encoder and decoder are both vanilla ViT models.
- The skeleton of the code is recycled from Facebook's [MAE](https://github.com/facebookresearch/mae) repository with several simplifications.
- Work in progress.

### Why transformer-based autoencoders?

- Better representational alignment with transformer models used in downstream tasks, *e.g.* diffusion transformers.
- Possibly achieving better spatial compression by trading off embedding dimensionality, *e.g.* being able to train diffusion transformers with a 4x4 spatial grid = 16 spatial tokens. In transformers, complexity scales quadratically with the number of spatial tokens and linearly with dimensionality, so this potentially leads to more compute efficient models. This can in principle be done with convnet-based autoencoders too, but is more natural and convenient with transformers.
- Current "first stage models" used for image/video compression are too complicated, *e.g.* using adversarial losses (among others). I'd like to simplify this process by showing simple plain autoencoders are performant as first stage models.