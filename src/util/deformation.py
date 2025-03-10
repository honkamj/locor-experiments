"""Deformation related utilities."""

from typing import Sequence

from composable_mapping import (
    CoordinateSystem,
    DataFormat,
    LinearInterpolator,
    NearestInterpolator,
    from_file,
    samplable_volume,
)
from nibabel import load as nib_load
from numpy import ndarray, zeros
from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import from_numpy, linspace, meshgrid, stack


def deform_image_from_path(
    path: str,
    displacement_field: ndarray,
    affine: ndarray,
    mask_path: str | None = None,
    interpolation_mode: str = "linear",
) -> tuple[ndarray, ndarray]:
    """Deform an image at path"""
    interpolator = {
        "linear": LinearInterpolator,
        "nearest": NearestInterpolator,
    }[interpolation_mode]
    torch_displacement_field = from_numpy(displacement_field).movedim(-1, 0)
    volume = from_file(
        path,
        mask_path=mask_path,
        sampler=interpolator(extrapolation_mode="zeros"),
        dtype=torch_displacement_field.dtype,
    )
    deformation = samplable_volume(
        torch_displacement_field[None],
        coordinate_system=CoordinateSystem.from_affine_matrix(
            torch_displacement_field.shape[1:],
            from_numpy(affine).to(dtype=torch_displacement_field.dtype),
        ),
        data_format=DataFormat.voxel_displacements(),
    )
    deformed_values, deformed_mask = (
        (volume @ deformation).sample_to(deformation).generate(generate_missing_mask=True)
    )
    deformed_values = deformed_values[0].movedim(0, -1).squeeze(-1)
    deformed_mask = deformed_mask[0].movedim(0, -1).squeeze(-1)
    return deformed_values.numpy(force=True), deformed_mask.numpy(force=True)


def generate_voxel_coordinate_grid(
    shape: Sequence[int], device: torch_device, dtype: torch_dtype | None = None
) -> Tensor:
    """Generate voxel coordinate grid

    Args:
        shape: Shape of the grid
        device: Device of the grid

    Returns: Tensor with shape (1, len(shape), dim_1, ..., dim_{len(shape)})
    """
    axes = [
        linspace(start=0, end=int(dim_size) - 1, steps=int(dim_size), device=device, dtype=dtype)
        for dim_size in shape
    ]
    coordinates = stack(meshgrid(axes, indexing="ij"), dim=0)
    return coordinates[None]


def obtain_zero_displacement_field(reference_image_path: str) -> tuple[ndarray, ndarray]:
    """Obtain zero displacement field in reference image coordinates"""
    reference_image = nib_load(reference_image_path)
    affine = reference_image.affine  # type: ignore
    n_dims = affine.shape[0] - 1
    shape = reference_image.dataobj.shape[:n_dims]  # type: ignore
    displacement_field = zeros(shape + (n_dims,), dtype="float32")
    return displacement_field, affine
