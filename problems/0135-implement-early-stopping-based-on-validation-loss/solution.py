import torch
from typing import Tuple

def early_stopping(val_losses: torch.Tensor, patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine when to stop training early based on validation losses.
    
    Args:
        val_losses: A 1D tensor of validation losses for each epoch
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum decrease in loss to qualify as an improvement
    
    Returns:
        Tuple of (stop_epoch, best_epoch)
    """
    best_loss=float("inf")
    best_epoch=0
    wait=0

    for epoch,loss in enumerate(val_losses):
        current_loss=loss.item()

        if best_loss-current_loss> min_delta:
            best_loss=current_loss
            best_epoch=epoch
            wait=0
        else:
            wait+=1

            if wait>=patience:
                return epoch,best_epoch


    return val_losses.shape[0]-1,best_epoch
