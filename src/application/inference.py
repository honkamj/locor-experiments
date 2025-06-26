"""Inference for a registration algorithm."""

from json import dump
from os import makedirs
from os.path import basename, join
from traceback import print_exc

from nibabel import Nifti1Image
from nibabel import load as nib_load
from nibabel import save as nib_save
from numpy import array, mean
from tqdm import tqdm  # type: ignore

from data.interface import RegistrationDataset
from method.interface import RegistrationMethod
from util.deformation import deform_image_from_path, obtain_zero_displacement_field
from util.evaluation import compute_summary_statistics


def registration_inference(
    method: RegistrationMethod,
    dataset: RegistrationDataset,
    target_folder: str,
    save_displacement_field: bool = False,
    save_deformed_image: bool = False,
) -> float:
    """Perform inference and evaluation for a registration method.

    Returns:
        The average score of the registration method
    """
    makedirs(target_folder, exist_ok=True)
    scores: list[float] = []
    print("Starting inference...")
    for index, (
        case_name,
        reference_image_path,
        moving_image_path,
        reference_mask_path,
        moving_mask_path,
    ) in tqdm(enumerate(dataset)):
        makedirs(join(target_folder, case_name), exist_ok=True)
        try:
            displacement_field, affine = method.register(
                reference_image_path,
                moving_image_path,
                reference_mask_path,
                moving_mask_path,
            )
        except Exception:
            displacement_field, affine = obtain_zero_displacement_field(reference_image_path)
            print_exc()
        case_basename = basename(case_name)
        if save_displacement_field:
            nib_save(
                Nifti1Image(displacement_field, affine=affine),
                join(target_folder, case_name, f"{case_basename}-deformation.nii.gz"),
            )
        if save_deformed_image:
            deformed_moving, _deformed_moving_mask = deform_image_from_path(
                moving_image_path, displacement_field, affine
            )
            nib_save(
                Nifti1Image(deformed_moving, affine=affine),
                join(target_folder, case_name, f"{case_basename}-deformed.nii.gz"),
            )
        evaluation_results = dataset.evaluate(index, displacement_field)
        scores.append(dataset.metrics_to_single_score(evaluation_results))
        with open(
            join(target_folder, case_name, f"{case_basename}-evaluation.json"),
            mode="w",
            encoding="utf-8",
        ) as results_file:
            dump(evaluation_results, results_file, indent=4)
    score_summary = compute_summary_statistics(array(scores))
    mean_score = float(mean(scores))
    with open(join(target_folder, "results.json"), mode="w", encoding="utf-8") as results_file:
        dump(
            {"score": score_summary},
            results_file,
            indent=4,
        )
    print(f"Mean score: {mean_score}")
    return mean_score


def registration_evaluation(
    dataset: RegistrationDataset,
    target_folder: str,
) -> float:
    """Evaluate a registration algorithm based on existing displacement fields.

    Returns:
        The average score of the registration results.
    """
    makedirs(target_folder, exist_ok=True)
    score_sum = 0.0
    print("Starting evaluation...")
    for index, (
        case_name,
        _reference_image_path,
        _moving_image_path,
        _reference_mask_path,
        _moving_mask_path,
    ) in tqdm(enumerate(dataset)):
        case_basename = basename(case_name)
        displacement_field_path = join(
            target_folder, case_name, f"{case_basename}-deformation.nii.gz"
        )
        displacement_field = nib_load(displacement_field_path).dataobj[...]  # type: ignore
        evaluation_results = dataset.evaluate(index, displacement_field)
        score_sum += dataset.metrics_to_single_score(evaluation_results)
        with open(
            join(target_folder, case_name, f"{case_basename}-evaluation.json"),
            mode="w",
            encoding="utf-8",
        ) as results_file:
            dump(evaluation_results, results_file, indent=4)
    print(f"Average score: {score_sum / len(dataset)}")
    return score_sum / len(dataset)
