"""Corrfield implementation for evaluation"""

from os.path import join
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from nibabel import load as nib_load
from numpy import ndarray, ones

from data.util import save_nifti
from method.interface import RegistrationMethod


class Corrfield(RegistrationMethod):
    """Corrfield implementation for evaluation."""

    def __init__(
        self,
        corrfield_path: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self._parameters: Mapping[str, Any] = (
            self.default_parameters
            if parameters is None
            else dict(self.default_parameters) | dict(parameters)
        )
        self._corrfield_path = corrfield_path

    @property
    def default_parameters(self) -> Mapping[str, Any]:
        """Default parameters for the Corrfield registration method."""
        return {"alpha": 2.5, "beta": 150, "gamma": 5, "delta": 1, "last_stage_search_radius": 8}

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self._parameters

    def register(
        self,
        reference_image_path: str,
        moving_image_path: str,
        reference_mask_path: str | None = None,
        moving_mask_path: str | None = None,
    ) -> tuple[ndarray, ndarray]:
        with TemporaryDirectory() as output_directory:
            command = [
                join(self._corrfield_path, "corrfield.py"),
                "-F",
                reference_image_path,
                "-M",
                moving_image_path,
                "-a",
                str(self._parameters["alpha"]),
                "-b",
                str(self._parameters["beta"]),
                "-g",
                str(self._parameters["gamma"]),
                "-d",
                str(self._parameters["delta"]),
                "-L",
                f"{2 * self._parameters['last_stage_search_radius']}x"
                f"{2 * self._parameters['last_stage_search_radius']}x"
                f"{self._parameters['last_stage_search_radius']}",
                "-N",
                "6x6x3",
                "-Q",
                "2x2x1",
                "-R",
                "3x3x2",
                "-T",
                "rxnxn",
                "-O",
                join(output_directory, "output"),
            ]
            affine = nib_load(reference_image_path).affine  # type: ignore
            if reference_mask_path is None:
                mask = ones(
                    nib_load(reference_image_path).dataobj.shape, dtype="uint8"  # type: ignore
                )
                save_nifti(mask, join(output_directory, "reference_mask.nii.gz"), affine=affine)
                reference_mask_path = join(output_directory, "reference_mask.nii.gz")
            command.extend(
                [
                    "-m",
                    reference_mask_path,
                ]
            )
            print(f"Running command:\n{' '.join(command)}")
            run(
                command,
                check=True,
                capture_output=False,
            )
            displacement_field_path = join(output_directory, "output.nii.gz")
            displacement_field_image = nib_load(displacement_field_path)
            displacement_field = displacement_field_image.dataobj[...]  # type: ignore
        return displacement_field, affine
