import torch

from dgmr import DGMR
model = DGMR(
        forecast_steps=10,
        input_channels=3,
        output_channels=3,
        output_shape=256,
        latent_channels=384*2,
        context_channels=384,
        num_samples=6,
    )

x = torch.rand((4, 4, 1, 256, 256))
out = model(x)
pass