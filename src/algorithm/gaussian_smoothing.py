"""Gaussian smoothing."""

from math import ceil
from typing import Sequence

from torch import Tensor
from torch import device as torch_device
from torch import linspace
from torch.distributions import Normal
from torch.nn.functional import conv1d


def gaussian_smoothing(volume: Tensor, stds: Sequence[float], truncate_at_n_stds: float) -> Tensor:
    """Apply Gaussian smoothing"""
    smoothed_volume = volume
    for dim, std in enumerate(stds):
        kernel = _gaussian_kernel_1d(std, truncate_at_n_stds, device=volume.device)
        volume_dim = volume.ndim - len(stds) + dim
        smoothed_volume = _conv1d(
            smoothed_volume,
            kernel,
            dim=volume_dim,
        )
    return smoothed_volume


def _gaussian_kernel_1d(std: float, truncate_at_n_stds: float, device: torch_device) -> Tensor:
    ceil_coordinate = int(ceil(truncate_at_n_stds * std))
    steps = 2 * ceil_coordinate + 1
    max_abs_coordinate = float(ceil_coordinate)
    coordinates = linspace(-max_abs_coordinate, max_abs_coordinate, steps=steps, device=device)
    kernel: Tensor = Normal(loc=0.0, scale=std).log_prob(coordinates).exp()
    return kernel / kernel.sum()


def _conv1d(input_tensor: Tensor, kernel: Tensor, dim: int) -> Tensor:
    dim_size = input_tensor.size(dim)
    input_tensor = input_tensor.moveaxis(dim, -1)
    dim_excluded_shape = input_tensor.shape[:-1]
    input_tensor = input_tensor.reshape(-1, 1, dim_size)
    convolved = conv1d(  # pylint: disable=not-callable
        input_tensor, kernel[None, None], bias=None
    ).reshape(dim_excluded_shape + (-1,))
    return convolved.moveaxis(-1, dim)
