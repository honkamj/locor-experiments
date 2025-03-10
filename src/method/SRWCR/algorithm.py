"""SRWCR baseline implementation"""

from os import environ
from os.path import join
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from nibabel import load as nib_load
from numpy import ndarray

from method.config_template import save_config_to
from method.interface import RegistrationMethod


class SRWCR(RegistrationMethod):
    """SRWCR baseline"""

    DEFAULT_PARAMS: dict[str, Any] = {
        "regularization_weight": 5.0e2,
        "n_last_stage_iterations": 40,
        "sliding_window_stride": 4,
        "n_bins": 32,
        "affine_learning_rate": 5e-3,
        "base_dense_learning_rate": 4e-2,
    }

    def __init__(
        self,
        parameters: Mapping[str, Any] | None = None,
        devices: Sequence[str] | None = None,
    ) -> None:
        self._devices = devices
        self._parameters = (
            self.DEFAULT_PARAMS if parameters is None else self.DEFAULT_PARAMS | dict(parameters)
        )

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
            save_config_to(
                "method/SRWCR/config_template.pyt",
                join(output_directory, "config.py"),
                replacements={
                    "{$REGULARIZATION_WEIGHT}": self._parameters["regularization_weight"],
                    "{$N_LAST_STAGE_ITERATIONS}": self._parameters["n_last_stage_iterations"],
                    "{$SLIDING_WINDOW_STRIDE}": self._parameters["sliding_window_stride"],
                    "{$N_BINS}": self._parameters["n_bins"],
                    "{$AFFINE_LEARNING_RATE}": self._parameters["affine_learning_rate"],
                    "{$BASE_DENSE_LEARNING_RATE}": self._parameters["base_dense_learning_rate"],
                },
            )
            command = [
                "python",
                "-m",
                "srwcr_baseline",
                reference_image_path,
                moving_image_path,
                "-d",
                join(output_directory, "deformation.nii"),
                "--config",
                join(output_directory, "config.py"),
                "--initialize-at-center",
            ]
            if self._devices is not None:
                for device in self._devices:
                    command.append("--device")
                    command.append(device)
            if reference_mask_path is not None:
                command.append("--mask-reference")
                command.append(reference_mask_path)
            if moving_mask_path is not None:
                command.append("--mask-moving")
                command.append(moving_mask_path)
            print(f"Running command:\n{' '.join(command)}")
            run(
                command,
                check=True,
                env=environ,
                capture_output=False,
            )
            displacement_field_path = join(output_directory, "deformation.nii")
            displacement_field_image = nib_load(displacement_field_path)
            displacement_field = displacement_field_image.dataobj[...]  # type: ignore
            affine = displacement_field_image.affine  # type: ignore
        return displacement_field, affine
