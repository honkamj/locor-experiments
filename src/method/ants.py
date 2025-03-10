"""ANTs implementation for evaluation"""

from os import environ
from os.path import join
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from composable_mapping import CoordinateSystem, DataFormat, samplable_volume
from nibabel import load as nib_load
from numpy import ndarray
from torch import from_numpy

from method.interface import RegistrationMethod


class ANTs(RegistrationMethod):
    """ANTs implementation for evaluation."""

    def __init__(
        self,
        ants_path: str,
        parameters: Mapping[str, Any] | None = None,
        n_threads: int | None = None,
    ) -> None:
        self._parameters: Mapping[str, Any] = (
            self.default_parameters
            if parameters is None
            else dict(self.default_parameters) | dict(parameters)
        )
        self._ants_path = ants_path
        self._n_threads = n_threads

    @property
    def default_parameters(self) -> Mapping[str, Any]:
        """Default parameters for the ANTs registration method."""
        return {
            "rigid_stepsize": 0.1,
            "affine_stepsize": 0.1,
            "syn_stepsize": 0.1,
            "windowing_quantile": 0.005,
            "update_field_variance": 3.0,
            "total_field_variance": 0.0,
            "rigid_affine_max_n_last_stage_iterations": 100,
            "syn_max_n_last_stage_iterations": 40,
        }

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
        env = (
            None
            if self._n_threads is None
            else environ | {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(self._n_threads)}
        )
        with TemporaryDirectory() as output_directory:
            n_dims = nib_load(reference_image_path).dataobj.ndim  # type: ignore
            rigid_affine_max_n_interations = (
                f"{10 * self._parameters['rigid_affine_max_n_last_stage_iterations']}x"
                f"{5 * self._parameters['rigid_affine_max_n_last_stage_iterations']}x"
                f"{int(round(2.5 * self._parameters['rigid_affine_max_n_last_stage_iterations']))}x"
                f"{self._parameters['rigid_affine_max_n_last_stage_iterations']}"
            )
            syn_max_n_interations = (
                f"{5 * self._parameters['syn_max_n_last_stage_iterations']}x"
                f"{int(round(3.5 * self._parameters['syn_max_n_last_stage_iterations']))}x"
                f"{int(round(2.5 * self._parameters['syn_max_n_last_stage_iterations']))}x"
                f"{self._parameters['syn_max_n_last_stage_iterations']}"
            )
            command = [
                join(self._ants_path, "antsRegistration"),
                "--dimensionality",
                str(n_dims),
                "--interpolation",
                "Linear",
                "--winsorize-image-intensities",
                f"[{self._parameters['windowing_quantile']},"
                f"{1 - self._parameters['windowing_quantile']}]",
                "--use-histogram-matching",
                "0",
                "--initial-moving-transform",
                f"[{reference_image_path},{moving_image_path},0]",
                "--transform",
                f"Rigid[{self._parameters['rigid_stepsize']}]",
                "--metric",
                f"MI[{reference_image_path},{moving_image_path},1,32,Regular,0.25]",
                "--convergence",
                f"[{rigid_affine_max_n_interations},1e-6,10]",
                "--shrink-factors",
                "8x4x2x1",
                "--smoothing-sigmas",
                "3x2x1x0vox",
                "--transform",
                f"Affine[{self._parameters['affine_stepsize']}]",
                "--metric",
                f"MI[{reference_image_path},{moving_image_path},1,32,Regular,0.25]",
                "--convergence",
                f"[{rigid_affine_max_n_interations},1e-6,10]",
                "--shrink-factors",
                "8x4x2x1",
                "--smoothing-sigmas",
                "3x2x1x0vox",
                "--transform",
                f"SyN[{self._parameters['syn_stepsize']},"
                f"{self._parameters['update_field_variance']},"
                f"{self._parameters['total_field_variance']}]",
                "--metric",
                f"MI[{reference_image_path},{moving_image_path},1,32]",
                "--convergence",
                f"[{syn_max_n_interations},1e-6,10]",
                "--shrink-factors",
                "8x4x2x1",
                "--smoothing-sigmas",
                "3x2x1x0vox",
                "--output",
                f"[{join(output_directory, 'transformation')},"
                f"{join(output_directory, 'warped.nii')}]",
                "--verbose",
                "--write-composite-transform",
                "1",
            ]
            if reference_mask_path is None:
                reference_mask_path = join(output_directory, "no_mask.nii")
            if moving_mask_path is None:
                moving_mask_path = join(output_directory, "no_mask.nii")
            command.extend(
                [
                    "--masks",
                    f"[{reference_mask_path},{moving_mask_path}]",
                ]
            )
            print(f"Running command:\n{' '.join(command)}")
            run(
                command,
                check=True,
                capture_output=False,
                env=env,
            )
            composition_command = [
                join(self._ants_path, "antsApplyTransforms"),
                "--dimensionality",
                str(n_dims),
                "--reference-image",
                reference_image_path,
                "--transform",
                join(output_directory, "transformationComposite.h5"),
                "--output",
                f"[{join(output_directory, 'deformation.nii')},1]",
            ]
            print(f"Running command:\n{' '.join(composition_command)}")
            run(
                composition_command,
                check=True,
                capture_output=False,
                env=env,
            )
            coordinate_mapping_path = join(output_directory, "deformation.nii")
            coordinate_mapping_image = nib_load(coordinate_mapping_path)
            coordinate_mapping_data = coordinate_mapping_image.dataobj[...][  # type: ignore
                ..., 0, :
            ].astype("float32")
            # Convert displacements from LPS (used by ITK) to RAS (used by NiBabel)
            coordinate_mapping_data[..., 0] = -coordinate_mapping_data[..., 0]
            coordinate_mapping_data[..., 1] = -coordinate_mapping_data[..., 1]
            deformation = samplable_volume(
                from_numpy(coordinate_mapping_data).movedim(-1, 0)[None],
                coordinate_system=CoordinateSystem.from_affine_matrix(
                    spatial_shape=coordinate_mapping_data.shape[:-1],
                    affine_matrix=from_numpy(
                        coordinate_mapping_image.affine.astype("float32")  # type: ignore
                    ),
                ),
                data_format=DataFormat.world_displacements(),
            )
            displacement_field = (
                deformation.sample(DataFormat.voxel_displacements())
                .generate_values()[0]
                .movedim(0, -1)
                .numpy(force=True)
            )
            affine = coordinate_mapping_image.affine  # type: ignore
        return displacement_field, affine
