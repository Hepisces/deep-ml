import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    max_score = score_tensor.max()
    logit = torch.round(
        torch.exp(score_tensor - max_score)/torch.sum(torch.exp(score_tensor - max_score)
    ),decimals=4)
    return logit.tolist()
