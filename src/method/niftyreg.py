"""NiftyReg implementation for evaluation"""

from os.path import join
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from composable_mapping import CoordinateSystem, DataFormat, samplable_volume
from nibabel import load as nib_load
from numpy import ndarray
from torch import from_numpy

from method.interface import RegistrationMethod


class _NiftyRegBase(RegistrationMethod):
    """Base NiftyReg implementation for evaluation."""

    def __init__(
        self,
        niftyreg_path: str,
        parameters: Mapping[str, Any] | None = None,
        n_threads: int | None = None,
    ) -> None:
        self._parameters: Mapping[str, Any] = (
            self.default_parameters
            if parameters is None
            else dict(self.default_parameters) | dict(parameters)
        )
        self._niftyreg_path = niftyreg_path
        self._n_threads = n_threads

    @property
    def default_parameters(self) -> Mapping[str, Any]:
        """Default parameters for the NiftyReg registration method."""
        return {
            "affine": True,
            "bending_energy_weight": 0.001,
            "first_order_penalty_weight": 0.01,
            "velocity_field": False,
        }

    @property
    def f3d_command_parameters(self) -> Sequence[str]:
        """Parameters for the f3d command."""
        parameters = [
            "-be",
            str(self._parameters["bending_energy_weight"]),
            "-le",
            str(self._parameters["first_order_penalty_weight"]),
        ]
        if self._parameters["velocity_field"]:
            parameters.append("-vel")
        return parameters

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
            if self._parameters["affine"]:
                affine_command = [
                    join(self._niftyreg_path, "reg_aladin"),
                    "-ref",
                    reference_image_path,
                    "-flo",
                    moving_image_path,
                    "-aff",
                    join(output_directory, "affine.txt"),
                    "-res",
                    join(output_directory, "registered.nii"),
                    "-omp",
                    str(self._n_threads),
                ]
                run(
                    affine_command,
                    check=True,
                    capture_output=False,
                )
            command = [
                join(self._niftyreg_path, "reg_f3d"),
                "-ref",
                reference_image_path,
                "-flo",
                moving_image_path,
                "-cpp",
                join(output_directory, "control_point_grid.nii"),
                "-res",
                join(output_directory, "registered.nii"),
                "-omp",
                str(self._n_threads),
            ] + list(self.f3d_command_parameters)
            if self._parameters["affine"]:
                command.append("-aff")
                command.append(join(output_directory, "affine.txt"))
            if reference_mask_path is not None:
                command.append("-rmask")
                command.append(reference_mask_path)
            if moving_mask_path is not None:
                command.append("-fmask")
                command.append(moving_mask_path)
            print(f"Running command:\n{' '.join(command)}")
            run(
                command,
                check=True,
                capture_output=False,
            )
            deformation_command = [
                join(self._niftyreg_path, "reg_transform"),
                "-ref",
                reference_image_path,
                "-def",
                join(output_directory, "control_point_grid.nii"),
                join(output_directory, "deformation.nii"),
            ]
            print(f"Running command:\n{' '.join(deformation_command)}")
            run(
                deformation_command,
                check=True,
                capture_output=False,
            )
            coordinate_mapping_path = join(output_directory, "deformation.nii")
            coordinate_mapping_image = nib_load(coordinate_mapping_path)
            coordinate_mapping_data = coordinate_mapping_image.dataobj[...][  # type: ignore
                ..., 0, :
            ].astype("float32")
            deformation = samplable_volume(
                from_numpy(coordinate_mapping_data).movedim(-1, 0)[None],
                coordinate_system=CoordinateSystem.from_affine_matrix(
                    spatial_shape=coordinate_mapping_data.shape[:-1],
                    affine_matrix=from_numpy(
                        coordinate_mapping_image.affine.astype("float32")  # type: ignore
                    ),
                ),
            )
            displacement_field = (
                deformation.sample(DataFormat.voxel_displacements())
                .generate_values()[0]
                .movedim(0, -1)
                .numpy(force=True)
            )
            affine = coordinate_mapping_image.affine  # type: ignore
        return displacement_field, affine


class NiftyRegNMI(_NiftyRegBase):
    """NiftyReg implementation using the NMI metric."""

    @property
    def f3d_command_parameters(self) -> Sequence[str]:
        return list(super().f3d_command_parameters) + ["--nmi"]


class NiftyRegMIND(_NiftyRegBase):
    """NiftyReg implementation using the MIND metric."""

    @property
    def default_parameters(self) -> Mapping[str, Any]:
        return dict(super().default_parameters) | {
            "mind_offset": 1,
        }

    @property
    def f3d_command_parameters(self) -> Sequence[str]:
        return list(super().f3d_command_parameters) + [
            "--mind",
            str(self._parameters["mind_offset"]),
        ]
