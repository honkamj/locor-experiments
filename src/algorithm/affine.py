"""Affine transformations."""

from torch import Tensor, cat, eye, matrix_exp, tril_indices, triu_indices, zeros


def generate_rotation_matrix(rotations: Tensor) -> Tensor:
    """Generate a rotation matrix."""
    if rotations.size(0) == 1:
        n_dims = 2
    elif rotations.size(0) == 3:
        n_dims = 3
    else:
        raise ValueError("Only 2D and 3D rotations are supported.")
    non_diagonal_indices = cat(
        (triu_indices(n_dims, n_dims, 1), tril_indices(n_dims, n_dims, -1)), dim=1
    )
    log_rotation_matrix = zeros(n_dims, n_dims, device=rotations.device, dtype=rotations.dtype)
    log_rotation_matrix[non_diagonal_indices[0], non_diagonal_indices[1]] = cat(
        (rotations, -rotations), dim=0
    )
    rotation_matrix = matrix_exp(log_rotation_matrix)
    embedded_rotation_matrix = zeros(
        n_dims + 1, n_dims + 1, device=rotations.device, dtype=rotations.dtype
    )
    embedded_rotation_matrix[:-1, :-1] = rotation_matrix
    embedded_rotation_matrix[-1, -1] = 1.0
    return embedded_rotation_matrix


def generate_translation_matrix(translations: Tensor) -> Tensor:
    """Generate a translation matrix."""
    n_dims = translations.size(0)
    translation_matrix = eye(n_dims + 1, device=translations.device, dtype=translations.dtype)
    translation_matrix[:-1, -1] = translations
    translation_matrix[-1, -1] = 1.0
    return translation_matrix
