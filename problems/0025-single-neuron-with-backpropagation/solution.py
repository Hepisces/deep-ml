import torch
import torch.nn as nn


def train_neuron(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    weights = initial_weights.clone().detach().requires_grad_(True)
    bias = torch.tensor(initial_bias, requires_grad=True)
    optimizer = torch.optim.SGD([weights, bias], lr=learning_rate)

    mse_loss=nn.MSELoss()
    mse_values: list[float] = []


    labels =labels.to(dtype=features.dtype)

    for _ in range(epochs):
        optimizer.zero_grad()

        z = features @ weights + bias
        predictions=torch.sigmoid(z)

        loss = mse_loss(predictions, labels)

        loss.backward()
        optimizer.step()

        mse_values.append(round(loss.item(),4))

    updated_weights=[round(w,4) for w in weights.detach().tolist()]
    updated_bias=round(bias.item(),4)

    return updated_weights, updated_bias, mse_values