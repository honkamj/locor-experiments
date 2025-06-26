"""Configuration template for SRWCR baseline."""

from srwcr_baseline.affine_transformation import AffineTransformationTypeDefinition
from srwcr_baseline.config_parameters import (
    AffineStageParameters,
    ConfigBuildingArguments,
    DenseStageParameters,
    ImageParameters,
    RegistrationParameters,
    RegularizationParameters,
)
from srwcr_baseline.regularization import BendingEnergy


def build_config(arguments: ConfigBuildingArguments) -> RegistrationParameters:
    """Build default configuration for the image registration tool."""
    feature_learning_rate = 1e-2
    similarity_sliding_window_stride = {$SLIDING_WINDOW_STRIDE}

    reference_voxel_spacing = arguments.reference.coordinate_system.grid_spacing_cpu()
    moving_voxel_spacing = arguments.moving.coordinate_system.grid_spacing_cpu()

    # Regularization base weight is scaled by the square of the image sampling
    # grid spacing for each stage to account for the larger voxel size causing
    # reduced relative bending energy weight.
    min_voxel_spacing_reference = reference_voxel_spacing.amin().item()

    regularization_base_weight = {$REGULARIZATION_WEIGHT}
    regularization_loss = BendingEnergy()

    similarity_n_bins = 32

    n_last_stage_iterations = {$N_LAST_STAGE_ITERATIONS}

    base_dense_learning_rate = {$BASE_DENSE_LEARNING_RATE}

    return RegistrationParameters(
        affine_stage_parameters=AffineStageParameters(
            n_iterations=2 * n_last_stage_iterations,
            feature_learning_rate=feature_learning_rate,
            transformation_learning_rate={$AFFINE_LEARNING_RATE},
            reference_image_parameters=ImageParameters(
                image_sampling_spacing=2 * reference_voxel_spacing,
                similarity_sliding_window_stride=similarity_sliding_window_stride,
                similarity_n_bins=similarity_n_bins,
            ),
            moving_image_parameters=ImageParameters(
                image_sampling_spacing=2 * moving_voxel_spacing,
                similarity_sliding_window_stride=similarity_sliding_window_stride,
                similarity_n_bins=similarity_n_bins,
            ),
            transformation_type=AffineTransformationTypeDefinition.full(),
        ),
        dense_stage_parameters=[
            DenseStageParameters(
                n_iterations=2 * n_last_stage_iterations,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=4 * base_dense_learning_rate,
                spline_grid_spacing=32 * reference_voxel_spacing,
                deformation_sampling_spacing=4 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (4 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=2 * n_last_stage_iterations,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=4 * base_dense_learning_rate,
                spline_grid_spacing=16 * reference_voxel_spacing,
                deformation_sampling_spacing=4 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=10,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=10,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (4 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=n_last_stage_iterations,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=2 * base_dense_learning_rate,
                spline_grid_spacing=8 * reference_voxel_spacing,
                deformation_sampling_spacing=2 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=10,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=10,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (2 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=n_last_stage_iterations,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=2 * base_dense_learning_rate,
                spline_grid_spacing=4 * reference_voxel_spacing,
                deformation_sampling_spacing=2 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=10,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=10,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (2 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=n_last_stage_iterations,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=base_dense_learning_rate,
                spline_grid_spacing=4 * reference_voxel_spacing,
                deformation_sampling_spacing=1 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=1 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=5,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=1 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_n_bins=similarity_n_bins,
                    sampling_coordinates_padding=5,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (1 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
        ],
    )
