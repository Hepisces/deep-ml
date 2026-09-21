import torch

def model_fit_quality(training_accuracy: torch.Tensor, test_accuracy: torch.Tensor) -> torch.Tensor:
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: torch.Tensor or float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: torch.Tensor or float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: torch.Tensor scalar, one of 1 (overfitting), -1 (underfitting), or 0 (good fit).
    """
    # Your code here
    if training_accuracy-test_accuracy>0.2:
        return torch.tensor(1)
    elif training_accuracy<0.7 and test_accuracy<0.7:
        return torch.tensor(-1)
    else:return torch.tensor(0)