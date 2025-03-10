"""Interface for evaluation of registration algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Mapping

from numpy import ndarray


class RegistrationMethod(ABC):
    """Interface for registration methods."""

    @property
    @abstractmethod
    def parameters(self) -> Mapping[str, Any]:
        """Get the parameters of the registration method."""

    @abstractmethod
    def register(
        self,
        reference_image_path: str,
        moving_image_path: str,
        reference_mask_path: str | None = None,
        moving_mask_path: str | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Register the moving image to the fixed image.

        Args:
            reference_image_path: The fixed image.
            moving_image_path: The moving image.
            reference_mask_path: The fixed mask.
            moving_mask_path: The moving mask.

        Returns:
            Displacement field and its affine matrix.
        """
