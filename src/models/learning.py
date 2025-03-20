import torch
import torch.nn as nn
import torch.nn.functional as F

class OHEM:
    def __init__(self, use_ohem: bool = False, ohem_ratio: float = 0.5, method: str = 'ratio'):
        """
        Initialize OHEM module.

        Args:
            use_ohem (bool): Whether to use OHEM.
            ohem_ratio (float): Fraction of hardest samples to keep. Used only if method='ratio'.
            method (str): Method to apply. Options: 'ratio' (top-k hardest samples), 'mse' (mean squared error),
                          or 'rmse' (root mean squared error).
        """
        all_methods = ['ratio', 'mse', 'rmse']
        assert method in all_methods, f"Expect method in {all_methods}, but got {method}."
        if method == 'ratio':
            assert ohem_ratio is not None, "ohem_ratio must be provided when method='ratio'."

        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio
        self.method = method

    def apply(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply OHEM to the loss tensor.

        Args:
            loss (torch.Tensor): Per-sample loss values.

        Returns:
            torch.Tensor: Mean loss, MSE, RMSE, or mean of top-k hardest samples after applying OHEM.
        """
        if not self.use_ohem:
            return torch.mean(loss)  # Default behavior: return mean loss

        if self.method == 'ratio':
            # Use top-k hard samples
            num_hard_samples = int(self.ohem_ratio * len(loss))
            hard_loss, _ = torch.topk(loss, num_hard_samples)
            return torch.mean(hard_loss)
        else:
            # Use squared approach on all samples
            squared_loss = torch.square(loss)
            mean_squared_loss = torch.mean(squared_loss)
            if self.method == 'rmse':
                return torch.sqrt(mean_squared_loss)  # RMSE
            else:
                return mean_squared_loss  # MSE