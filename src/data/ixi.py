"""IXI dataset."""

from argparse import ArgumentParser
from math import ceil, pi
from os import makedirs
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from ants import from_numpy as ants_from_numpy  # type: ignore
from ants import iMath
from composable_mapping import (
    Affine,
    CoordinateSystem,
    DataFormat,
    LinearInterpolator,
    NearestInterpolator,
    ScalingAndSquaring,
    samplable_volume,
)
from nibabel import load as nib_load
from nibabel.affines import voxel_sizes
from numpy import allclose, ndarray, quantile, sqrt
from scipy.ndimage import median_filter  # type: ignore
from torch import Tensor
from torch import device as torch_device
from torch import float32, from_numpy, manual_seed, rand, randn, tensor, uint8
from torch.cuda import is_available as is_cuda_available
from tqdm import tqdm  # type: ignore

from algorithm.affine import generate_rotation_matrix, generate_translation_matrix
from algorithm.gaussian_smoothing import gaussian_smoothing
from data.interface import RegistrationDataset, RegistrationDatasetInitializer
from util.deformation import deform_image_from_path
from util.evaluation import (
    compute_regularity_metrics,
    compute_similarity_metrics,
    compute_summary_statistics,
)

from .util import download, save_nifti, untar

SEED = 42

CASES = {
    "validation": ["IXI161-HH-2533", "IXI121-Guys-0772", "IXI127-HH-1451"],
    "test": [
        "IXI077-Guys-0752",
        "IXI519-HH-2240",
        "IXI237-Guys-1049",
        "IXI607-Guys-1097",
        "IXI347-IOP-0927",
        "IXI029-Guys-0829",
        "IXI508-HH-2268",
        "IXI392-Guys-1064",
        "IXI056-HH-1327",
        "IXI227-Guys-0813",
        "IXI517-IOP-1144",
        "IXI238-IOP-0883",
        "IXI551-Guys-1065",
        "IXI467-Guys-0983",
        "IXI225-Guys-0832",
        "IXI549-Guys-1046",
        "IXI508-HH-2268",
        "IXI021-Guys-0703",
        "IXI257-HH-1724",
        "IXI224-Guys-0823",
        "IXI637-HH-2785",
        "IXI216-HH-1635",
        "IXI265-Guys-0845",
        "IXI418-Guys-0956",
        "IXI184-Guys-0794",
        "IXI219-Guys-0894",
        "IXI411-Guys-0959",
        "IXI148-HH-1453",
        "IXI621-Guys-1100",
        "IXI475-IOP-1139",
        "IXI468-Guys-0985",
        "IXI554-Guys-1068",
        "IXI469-IOP-1136",
        "IXI638-HH-2786",
        "IXI330-Guys-0881",
        "IXI200-Guys-0812",
        "IXI208-Guys-0808",
        "IXI630-Guys-1108",
        "IXI633-HH-2689",
        "IXI334-HH-1907",
        "IXI495-Guys-1009",
        "IXI524-HH-2412",
        "IXI480-Guys-1033",
        "IXI325-Guys-0911",
        "IXI202-HH-1526",
        "IXI359-Guys-0918",
        "IXI320-Guys-0902",
        "IXI369-Guys-0924",
        "IXI177-Guys-0831",
        "IXI084-Guys-0741",
    ],
}

MIN_NOISE_STD = 700.0
MAX_NOISE_STD = 900.0

MIN_SMOOTHING_STD = 14.0
MAX_SMOOTHING_STD = 16.0
TRUNCATE_AT_N_STDS = 4.0

MIN_ROTATION = -pi / 8
MAX_ROTATION = pi / 8

MIN_TRANSLATION = -30.0
MAX_TRANSLATION = 30.0

DEVICE = torch_device("cuda") if is_cuda_available() else torch_device("cpu")


class IXIDatasetInitializer(RegistrationDatasetInitializer):
    """Initializer for IXI dataset."""

    def __init__(self) -> None:
        super().__init__("IXI")

    def _licence_agreement_question(self) -> str | None:
        return (
            "IXI dataset is not available in the data root. "
            "Do you want to download it? "
            "By downloading the data you agree to the terms of use and the licence at "
            "https://brain-development.org/ixi-dataset/. The data is released under "
            "CC BY-SA 3.0 license."
            "(yes/no) "
        )

    def _build(self, data_folder: str) -> None:
        manual_seed(SEED)
        with TemporaryDirectory() as temp_dir:
            download(
                "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
                join(temp_dir, "IXI-T2.tar"),
                description="Downloading IXI T2 images",
            )
            download(
                "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar",
                join(temp_dir, "IXI-PD.tar"),
                description="Downloading IXI PD images",
            )
            print("Extracting IXI dataset...")
            untar(join(temp_dir, "IXI-T2.tar"), remove_after=True)
            untar(join(temp_dir, "IXI-PD.tar"), remove_after=True)
            print("Processing validation cases...")
            for case in tqdm(CASES["validation"]):
                self._process_case(case, temp_dir, join(data_folder, "validation"))
            print("Processing test cases...")
            for case in tqdm(CASES["test"]):
                self._process_case(case, temp_dir, join(data_folder, "test"))

    @staticmethod
    def _generate_random_ddf(shape: Sequence[int], voxel_size: Sequence[float]) -> Tensor:
        noise_std = rand(1, device=DEVICE) * (MAX_NOISE_STD - MIN_NOISE_STD) + MIN_NOISE_STD
        smoothing_std = (
            (
                rand(1, device=DEVICE) * (MAX_SMOOTHING_STD - MIN_SMOOTHING_STD) + MIN_SMOOTHING_STD
            ).expand(len(shape))
            / tensor(voxel_size, device=DEVICE)
        ).tolist()
        paddings = [
            int(ceil(dim_smoothing_std * TRUNCATE_AT_N_STDS)) for dim_smoothing_std in smoothing_std
        ]
        random_noise = (
            randn(
                (
                    len(shape),
                    *(dim_size + 2 * padding for padding, dim_size in zip(paddings, shape)),
                ),
                device=DEVICE,
            )
            * noise_std
        )
        random_ddf = gaussian_smoothing(random_noise, smoothing_std, TRUNCATE_AT_N_STDS)
        return random_ddf

    @staticmethod
    def _generate_random_affine(voxel_size: Sequence[float]) -> Tensor:
        rotation = (
            rand(len(voxel_size), device=DEVICE) * (MAX_ROTATION - MIN_ROTATION) + MIN_ROTATION
        )
        translation = (
            rand(len(voxel_size), device=DEVICE) * (MAX_TRANSLATION - MIN_TRANSLATION)
            + MIN_TRANSLATION
        ) / tensor(voxel_size, device=DEVICE, dtype=float32)
        rotation_matrix = generate_rotation_matrix(rotation)
        translation_matrix = generate_translation_matrix(translation)
        return translation_matrix @ rotation_matrix

    @staticmethod
    def _generate_evaluation_mask(pd_image: ndarray, voxel_size: Sequence[float]) -> ndarray:
        filtered = pd_image / quantile(pd_image, 0.99)
        filtered = median_filter(filtered, size=3)
        filtered = (filtered > 0.05).astype(filtered.dtype)
        filtered_ants = ants_from_numpy(filtered, spacing=voxel_size)
        filtered_ants = iMath(filtered_ants, "ME", 1)
        filtered_ants = iMath(filtered_ants, "MD", 1)
        filtered_ants = iMath(filtered_ants, "GetLargestComponent")
        filtered_ants = iMath(filtered_ants, "MC", 8)
        filtered_ants = iMath(filtered_ants, "FillHoles").threshold_image(1, 2)
        return filtered_ants.numpy()

    @classmethod
    def _process_case(cls, case: str, source_dir: str, target_dir: str) -> None:
        makedirs(
            join(target_dir, case),
            exist_ok=True,
        )
        template_image = nib_load(join(source_dir, f"{case}-PD.nii.gz"))
        voxel_size: Sequence[float] = voxel_sizes(template_image.affine).tolist()  # type: ignore
        shape = template_image.header.get_data_shape()  # type: ignore

        comparison_image = nib_load(join(source_dir, f"{case}-T2.nii.gz"))

        template_data = template_image.dataobj[...].astype("float32")  # type: ignore
        evaluation_mask = from_numpy(cls._generate_evaluation_mask(template_data, voxel_size))

        assert allclose(template_image.affine, comparison_image.affine)  # type: ignore
        assert allclose(shape, comparison_image.header.get_data_shape())  # type: ignore

        random_ddf = cls._generate_random_ddf(shape, voxel_size)
        random_affine = cls._generate_random_affine(voxel_size)

        centered_coordinates = CoordinateSystem.centered(shape, voxel_size, device=DEVICE)

        deformation = (
            Affine.from_matrix(random_affine)
            @ samplable_volume(
                random_ddf[None],
                coordinate_system=centered_coordinates,
                data_format=DataFormat.voxel_displacements(),
                sampler=ScalingAndSquaring(),
            )
        ).resample()

        for modality in ["T2", "PD"]:
            source_path = join(source_dir, f"{case}-{modality}.nii.gz")
            image = nib_load(source_path)
            data = from_numpy(image.dataobj[...].astype("float32")).to(  # type: ignore
                device=DEVICE
            )
            deformed_volume = (
                samplable_volume(
                    data[None, None],
                    coordinate_system=centered_coordinates,
                    sampler=LinearInterpolator(extrapolation_mode="zeros"),
                )
                @ deformation
            ).sample()
            save_nifti(
                data,
                join(target_dir, case, f"{case}-{modality}.nii.gz"),
                affine=image.affine,  # type: ignore
            )
            save_nifti(
                deformed_volume.generate_values()[0, 0],
                join(target_dir, case, f"{case}-{modality}-deformed.nii.gz"),
                affine=image.affine,  # type: ignore
            )
            if modality == "T2":
                save_nifti(
                    deformed_volume.generate_mask(generate_missing_mask=True).to(dtype=uint8)[0, 0],
                    join(target_dir, case, f"{case}-mask-deformed.nii.gz"),
                    affine=image.affine,  # type: ignore
                )
        save_nifti(
            deformation.sample(DataFormat.voxel_displacements())
            .generate_values()[0]
            .movedim(0, -1),
            join(target_dir, case, f"{case}-deformation.nii.gz"),
            affine=image.affine,  # type: ignore
        )

        evaluation_mask_volume = samplable_volume(
            evaluation_mask.to(dtype=float32, device=DEVICE)[None, None],
            coordinate_system=centered_coordinates,
            sampler=NearestInterpolator(extrapolation_mode="zeros"),
        )
        deformed_evaluation_mask = (evaluation_mask_volume @ deformation).sample()
        save_nifti(
            evaluation_mask.to(dtype=uint8),
            join(target_dir, case, f"{case}-evaluation-mask.nii.gz"),
            affine=image.affine,  # type: ignore
        )
        save_nifti(
            (
                deformed_evaluation_mask.generate_values()
                * deformed_evaluation_mask.generate_mask(generate_missing_mask=True, cast_mask=True)
            ).to(dtype=uint8)[0, 0],
            join(target_dir, case, f"{case}-evaluation-mask-deformed.nii.gz"),
            affine=image.affine,  # type: ignore
        )

    def _load(self, data_folder: str, division: str) -> "RegistrationDataset":
        return IXIDataset(data_folder, division)


class IXIDataset(RegistrationDataset):
    """Access to IXI data."""

    def __init__(self, data_folder: str, division: str) -> None:
        self._data_folder = data_folder
        self._division = division

    def __len__(self) -> int:
        return len(CASES[self._division])

    def __getitem__(self, index: int) -> tuple[str, str, str, str | None, str | None]:
        case_name = CASES[self._division][index]
        return (
            case_name,
            join(self._data_folder, self._division, case_name, f"{case_name}-PD-deformed.nii.gz"),
            join(self._data_folder, self._division, case_name, f"{case_name}-T2.nii.gz"),
            join(self._data_folder, self._division, case_name, f"{case_name}-mask-deformed.nii.gz"),
            None,
        )

    def evaluate(self, index: int, displacement_field: ndarray) -> Mapping[str, Any]:
        (
            case_name,
            _reference_image_path,
            moving_image_path,
            _reference_mask_path,
            _moving_mask_path,
        ) = self[index]
        ground_truth_displacment_field: ndarray = nib_load(
            join(self._data_folder, self._division, case_name, f"{case_name}-deformation.nii.gz")
        ).dataobj[  # type: ignore
            ...
        ]
        evaluation_mask_image = nib_load(
            join(
                self._data_folder,
                self._division,
                case_name,
                f"{case_name}-evaluation-mask-deformed.nii.gz",
            )
        )
        affine = evaluation_mask_image.affine  # type: ignore
        evaluation_mask = evaluation_mask_image.dataobj[...]  # type: ignore

        displacement_field_diff_volume: ndarray = sqrt(
            ((displacement_field - ground_truth_displacment_field) ** 2).sum(axis=-1)
        )
        displacement_field_metrics = compute_summary_statistics(
            displacement_field_diff_volume, mask=evaluation_mask
        )

        ground_truth_deformed_moving = nib_load(
            join(self._data_folder, self._division, case_name, f"{case_name}-T2-deformed.nii.gz")
        ).dataobj[  # type: ignore
            ...
        ]
        deformed_moving, deformed_moving_mask = deform_image_from_path(
            moving_image_path,
            displacement_field,
            affine,
        )

        return {
            "displacement_field": displacement_field_metrics,
            "similarity": compute_similarity_metrics(
                ground_truth_deformed_moving,
                deformed_moving,
                evaluation_mask * deformed_moving_mask,
            ),
            "regularity": compute_regularity_metrics(displacement_field),
        }

    def metrics_to_single_score(self, metrics: Mapping[str, Any]) -> float:
        return metrics["displacement_field"]["mean"]


def _main():
    parser = ArgumentParser(description="Generate IXI dataset.")
    parser.add_argument(
        "data_root",
        type=str,
        help="Path to the root directory where the generated dataset will be stored.",
    )
    args = parser.parse_args()

    IXIDatasetInitializer().build(args.data_root, "null")


if __name__ == "__main__":
    _main()
