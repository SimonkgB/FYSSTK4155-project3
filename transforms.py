
# the func in the test example for log res VVVVV
#standardise func
# if we pool aswell as flatten
# AAAAAAAA DONE

# different augmentations like time shifts and gaussian noise to freq


import torch
import torch.nn.functional as tnF


class Standardize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X_train: torch.Tensor) -> "Standardize":
        # (1, C, 1, 1): one value per view, broadcasts over (N, C, H, W)
        self.mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
        self.std = X_train.std(dim=(0, 2, 3), keepdim=True)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class AvgPoolFlatten:
    """kxk average pooling and flattening, ie. for k=4: 140x170 -> 35x42 -> 1470."""
    def __init__(self, k: int):
        self.k = k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return tnF.avg_pool2d(x, self.k).flatten(1)