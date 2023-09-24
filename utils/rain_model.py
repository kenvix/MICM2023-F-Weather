import torch
import numpy as np
from tqdm import tqdm


class QuantitativeRainModel(torch.nn.Module):
    def __init__(self, vector_length=65536, device='cpu'):
        super(QuantitativeRainModel, self).__init__()
        self.vector_length = vector_length
        self.device = device

        self.a = torch.nn.Parameter(torch.ones(vector_length, device=device))
        self.b = torch.nn.Parameter(torch.ones(vector_length, device=device))
        self.c = torch.nn.Parameter(torch.ones(vector_length, device=device))

    # Define the target function to fit
    @staticmethod
    def target_function(x, a, b, c):
        """

        :param x: x0 dbz, x1 zdr
        :param a:
        :param b:
        :param c:
        :return:
        """
        return a * torch.log10(torch.nn.functional.relu(x[:, 0]) + 1) + b * x[:, 1] + c

    @staticmethod
    def loss_of(y, y_pred):
        return torch.mean((y - y_pred)**2)

    def forward(self, x):
        return QuantitativeRainModel.target_function(x, self.a, self.b, self.c)


def _entry():
    epoch_num = 10000
    device = 'cuda:0'

    # Generate some sample data to fit
    x_data = torch.linspace(0, 4, 50, device=device)

    # Train the model to fit the data and solve for the unknown parameters
    model = QuantitativeRainModel(vector_length=65536, device=device)

    # Generate some sample data to fit
    y_data = model.target_function(x_data, 2.5, 1.3, 0.5) + 0.2 * torch.randn(len(x_data), device=device)

    # Define the optimizer to use for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    pbar = tqdm(range(epoch_num))
    for i in pbar:
        optimizer.zero_grad()
        y_pred = model(x_data)
        loss = model.loss_of(y_data, y_pred)
        pbar.set_postfix({
            'Loss': loss.item(),
        })
        loss.backward()
        optimizer.step()

    # Print the optimized parameters
    print("Optimized parameters:", model.a, model.b, model.c)


if __name__ == '__main__':
    _entry()
