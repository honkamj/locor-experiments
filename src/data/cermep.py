"""CERMEP-IDB-MRXFDG dataset.

Mérida, Inés, et al. "CERMEP-IDB-MRXFDG: a database of 37 normal adult human
brain [18F] FDG PET, T1 and FLAIR MRI, and CT images available for research."
EJNMMI research 11.1 (2021)
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from ants import from_numpy as ants_from_numpy  # type: ignore
from ants import iMath
from nibabel import Nifti1Image
from nibabel import load as nib_load
from nibabel import save as nib_save
from nibabel.affines import voxel_sizes
from numpy import ndarray, quantile
from scipy.ndimage import median_filter  # type: ignore
from tqdm import tqdm  # type: ignore

from util.deformation import deform_image_from_path
from util.evaluation import compute_regularity_metrics, compute_similarity_metrics

from .interface import RegistrationDataset, RegistrationDatasetInitializer
from .util import save_nifti, untar

CASES = {
    "validation": ["sub-0004", "sub-0005", "sub-0008"],
    "test": [
        "sub-0001",
        "sub-0002",
        "sub-0003",
        "sub-0006",
        "sub-0007",
        "sub-0009",
        "sub-0010",
        "sub-0011",
        "sub-0012",
        "sub-0013",
        "sub-0014",
        "sub-0015",
        "sub-0016",
        "sub-0017",
        "sub-0018",
        "sub-0019",
        "sub-0020",
        "sub-0021",
        "sub-0022",
        "sub-0023",
        "sub-0024",
        "sub-0025",
        "sub-0026",
        "sub-0027",
        "sub-0028",
        "sub-0029",
        "sub-0030",
        "sub-0031",
        "sub-0032",
        "sub-0033",
        "sub-0034",
        "sub-0035",
        "sub-0036",
        "sub-0037",
    ],
}


class CERMEPDatasetInitializer(RegistrationDatasetInitializer):
    """Initializer for CERMEP-IDB-MRXFDG dataset."""

    def __init__(self) -> None:
        super().__init__("CERMEP")

    def _licence_agreement_question(self) -> str | None:
        return None

    def _build(self, data_folder: str) -> None:
        with TemporaryDirectory() as temp_dir:
            cermep_path = input(
                "Please provide the path to the CERMEP-iDB-MRXFDG database arhive "
                "(iDB-CERMEP-MRXFDG_*.tar.gz) containing CT and MR images. "
                "The database can be requested from the authors of the database "
                '(Mérida, Inés, et al. "CERMEP-IDB-MRXFDG: a database of 37 normal adult '
                "human brain [18F] FDG PET, T1 and FLAIR MRI, and CT images available for "
                'research." EJNMMI research 11.1 (2021)).'
                "\n\nPath: "
            )
            gt_dir = input(
                "\nPlease provide the path to the inference folder of the pseudo-CT images "
                "generated using the method at https://github.com/honkamj/non-aligned-i2i. "
                '(Honkamaa, Joel, et al. "Deformation equivariant cross-modality image '
                'synthesis with paired non-aligned training data." Medical Image Analysis '
                "90 (2023))."
                "\n\nPath: "
            )
            print("Extracting CERMEP dataset...")
            untar(cermep_path, target_dir=temp_dir, remove_after=False)
            source_dir = join(
                temp_dir,
                "home",
                "pool",
                "DM",
                "TEP",
                "CERMEP_MXFDG",
                "BASE",
                "DATABASE_SENT",
                "ALL",
            )
            print("Processing validation cases...")
            for case in tqdm(CASES["validation"]):
                self._process_case(case, source_dir, gt_dir, join(data_folder, "validation"))
            print("Processing test cases...")
            for case in tqdm(CASES["test"]):
                self._process_case(case, source_dir, gt_dir, join(data_folder, "test"))

    @classmethod
    def _process_case(cls, case: str, source_dir: str, gt_dir: str, target_dir: str) -> None:
        makedirs(join(target_dir, case), exist_ok=True)
        ct_image = nib_load(join(source_dir, case, "ct", f"{case}_ct.nii.gz"))
        t1_image = nib_load(join(source_dir, case, "anat", f"{case}_T1w.nii.gz"))
        pseudo_ct = nib_load(
            join(gt_dir, case, f"{case}_predicted.nii.gz")
        ).dataobj[  # type: ignore
            ...
        ]
        evaluation_mask = cls._generate_evaluation_mask(
            t1_image.dataobj[...], voxel_sizes(t1_image.affine).tolist()  # type: ignore
        )

        save_nifti(
            ct_image.dataobj[...],  # type: ignore
            join(target_dir, case, f"{case}_ct.nii.gz"),
            ct_image.affine,  # type: ignore
        )
        save_nifti(
            t1_image.dataobj[...],  # type: ignore
            join(target_dir, case, f"{case}_T1w.nii.gz"),
            t1_image.affine,  # type: ignore
        )
        save_nifti(
            pseudo_ct,
            join(target_dir, case, f"{case}_pseudo-ct.nii.gz"),
            t1_image.affine,  # type: ignore
        )
        save_nifti(
            evaluation_mask,
            join(target_dir, case, f"{case}_evaluation-mask.nii.gz"),
            t1_image.affine,  # type: ignore
        )

    @staticmethod
    def _generate_evaluation_mask(t1_image: ndarray, voxel_size: Sequence[float]) -> ndarray:
        filtered = t1_image / quantile(t1_image, 0.99)
        filtered = median_filter(filtered, size=3)
        filtered = (filtered > 0.1).astype(filtered.dtype)
        filtered_ants = ants_from_numpy(filtered, spacing=voxel_size)
        filtered_ants = iMath(filtered_ants, "ME", 1)
        filtered_ants = iMath(filtered_ants, "MD", 1)
        filtered_ants = iMath(filtered_ants, "GetLargestComponent")
        filtered_ants = iMath(filtered_ants, "MC", 8)
        filtered_ants = iMath(filtered_ants, "FillHoles").threshold_image(1, 2)
        filtered_ants = iMath(filtered_ants, "ME", 4)
        return filtered_ants.numpy()

    def _load(self, data_folder: str, division: str) -> "RegistrationDataset":
        return CERMEPDataset(data_folder, division)

    @staticmethod
    def _save(data: ndarray, path: str, affine: ndarray) -> None:
        nib_save(
            Nifti1Image(
                data,
                affine=affine,
            ),
            path,
        )


class CERMEPDataset(RegistrationDataset):
    """Access to CERMEP-IDB-MRXFDG data."""

    def __init__(self, data_folder: str, division: str) -> None:
        self._data_folder = data_folder
        self._division = division

    def __len__(self) -> int:
        return len(CASES[self._division])

    def __getitem__(self, index: int) -> tuple[str, str, str, str | None, str | None]:
        case_name = CASES[self._division][index]
        return (
            case_name,
            join(self._data_folder, self._division, case_name, f"{case_name}_T1w.nii.gz"),
            join(self._data_folder, self._division, case_name, f"{case_name}_ct.nii.gz"),
            None,
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
        ground_truth_deformed_moving_image = nib_load(
            join(self._data_folder, self._division, case_name, f"{case_name}_pseudo-ct.nii.gz")
        )
        ground_truth_deformed_moving = ground_truth_deformed_moving_image.dataobj[  # type: ignore
            ...
        ]
        deformed_moving, _deformed_moving_mask = deform_image_from_path(
            moving_image_path,
            displacement_field,
            ground_truth_deformed_moving_image.affine,  # type: ignore
        )
        evaluation_mask = nib_load(
            join(
                self._data_folder,
                self._division,
                case_name,
                f"{case_name}_evaluation-mask.nii.gz",
            )
        ).dataobj[  # type: ignore
            ...
        ]

        return {
            "similarity": compute_similarity_metrics(
                ground_truth_deformed_moving,
                deformed_moving,
                evaluation_mask,
            ),
            "regularity": compute_regularity_metrics(displacement_field),
        }

    def metrics_to_single_score(self, metrics: Mapping[str, Any]) -> float:
        return metrics["similarity"]["absolute"]["mean"]


def _main():
    parser = ArgumentParser(description="Generate CERMEP-IDB-MRXFDG dataset.")
    parser.add_argument(
        "data_root",
        type=str,
        help="Path to the root directory where the generated dataset will be stored.",
    )
    args = parser.parse_args()

    CERMEPDatasetInitializer().build(args.data_root, "null")


if __name__ == "__main__":
    _main()
