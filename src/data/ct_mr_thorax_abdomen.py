"""CT-MR Thorax-Abdomen dataset."""

from argparse import ArgumentParser
from os import makedirs
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from gdown import download as gdown_download  # type: ignore
from nibabel import load as nib_load
from numpy import ndarray
from scipy.ndimage import binary_erosion  # type: ignore
from tqdm import tqdm  # type: ignore

from data.util import download, save_nifti, unzip
from util.deformation import deform_image_from_path
from util.evaluation import compute_regularity_metrics, compute_tissue_overlap_metrics

from .interface import RegistrationDataset, RegistrationDatasetInitializer

CASES = {
    "validation": ["0012_tcia", "0014_tcia", "0016_tcia"],
    "test": [
        "0002_tcia",
        "0004_tcia",
        "0006_tcia",
        "0008_tcia",
        "0010_tcia",
    ],
}


FOREGROUND_MASK_THRESHOLD_EXCEPTIONS = {"0010_tcia": {"CT": -1023.0}}


# The number of iterations for the binary erosion operation for generating the
# foreground mask. Two images requires more iterations to get rid of background
# regions. Erosion is needed in the first place (in addition to thresholding)
# due to the images having been interpolated.
FOREGROUND_MASK_EROSION_ITERATION_EXCEPTIONS = {"0010_tcia": {"MR": 3}, "0014_tcia": {"MR": 4}}


LABEL_TO_NAME = {1: "liver", 2: "spleen", 3: "right_kidney", 4: "left_kidney"}


class CTMRThoraxAbdomenDatasetInitializer(RegistrationDatasetInitializer):
    """Initializer for CT-MR Thorax-Abdomen dataset."""

    def __init__(self, mask_type: str = "foreground_mask") -> None:
        super().__init__("CT-MR_Thorax-Abdomen")
        self._mask_type = mask_type

    def _licence_agreement_question(self) -> str | None:
        return (
            "CT-MR Thorax-Abdomen intra-patient registration dataset from Learn2Reg "
            "challenge is not available in the data root. "
            "Do you want to download it? "
            "By downloading the data you agree to the terms of use and the licence at "
            "https://learn2reg.grand-challenge.org/Learn2Reg2021/. "
            "(yes/no) "
        )

    def _build(self, data_folder: str) -> None:
        with TemporaryDirectory() as temp_dir:
            print("Downloading CT-MR Thorax Abdomen dataset...")
            download(
                "https://cloud.imi.uni-luebeck.de/s/DgGZFpTKBEn8PpS/download/L2R_Task1_MRCT_Train.zip",  # pylint: disable=line-too-long
                join(temp_dir, "train.zip"),
                description="Downloading CT-MR Thorax Abdomen images",
            )
            print("Downloading CT-MR Thorax Abdomen ROI masks...")
            gdown_download(
                id="1pW_UNe28_7gnZ4GynpV6XvLP-hLQrbyP",
                output=join(temp_dir, "roi_masks.zip"),
                quiet=False,
            )
            print("Extracting CT-MR Thorax Abdomen dataset...")
            unzip(join(temp_dir, "train.zip"), remove_after=True)
            unzip(join(temp_dir, "roi_masks.zip"), remove_after=True)
            print("Processing validation cases...")
            for case in tqdm(CASES["validation"]):
                self._process_case(case, temp_dir, join(data_folder, "validation"))
            print("Processing test cases...")
            for case in tqdm(CASES["test"]):
                self._process_case(case, temp_dir, join(data_folder, "test"))

    @classmethod
    def _process_case(cls, case: str, source_dir: str, target_dir: str) -> None:
        makedirs(join(target_dir, case), exist_ok=True)
        mr_image = nib_load(join(source_dir, "Train", f"img{case}_MR.nii.gz"))
        ct_image = nib_load(join(source_dir, "Train", f"img{case}_CT.nii.gz"))
        mr_roi_mask_image = nib_load(
            join(source_dir, "L2R_Task1_MRCT", "Train", f"mask{case}_MR.nii.gz")
        )
        ct_roi_mask_image = nib_load(
            join(source_dir, "L2R_Task1_MRCT", "Train", f"mask{case}_CT.nii.gz")
        )
        mr_seg_image = nib_load(join(source_dir, "Train", f"seg{case}_MR.nii.gz"))
        ct_seg_image = nib_load(join(source_dir, "Train", f"seg{case}_CT.nii.gz"))
        foreground_mask_mr = binary_erosion(
            mr_image.dataobj[...]  # type: ignore
            > FOREGROUND_MASK_THRESHOLD_EXCEPTIONS.get(case, {}).get("MR", 0.0),
            iterations=FOREGROUND_MASK_EROSION_ITERATION_EXCEPTIONS.get(case, {}).get("MR", 1),
        )
        foreground_mask_ct = binary_erosion(
            ct_image.dataobj[...]  # type: ignore
            > FOREGROUND_MASK_THRESHOLD_EXCEPTIONS.get(case, {}).get("CT", -1024.0),
            iterations=FOREGROUND_MASK_EROSION_ITERATION_EXCEPTIONS.get(case, {}).get("CT", 1),
        )
        joint_mask_mr = (mr_roi_mask_image.dataobj[...] > 0) & foreground_mask_mr  # type: ignore
        joint_mask_ct = (ct_roi_mask_image.dataobj[...] > 0) & foreground_mask_ct  # type: ignore

        save_nifti(
            mr_image.dataobj[...],  # type: ignore
            join(target_dir, case, f"img{case}_MR.nii.gz"),
            mr_image.affine,  # type: ignore
        )
        save_nifti(
            ct_image.dataobj[...],  # type: ignore
            join(target_dir, case, f"img{case}_CT.nii.gz"),
            ct_image.affine,  # type: ignore
        )
        save_nifti(
            mr_seg_image.dataobj[...],  # type: ignore
            join(target_dir, case, f"seg{case}_MR.nii.gz"),
            mr_seg_image.affine,  # type: ignore
        )
        save_nifti(
            ct_seg_image.dataobj[...],  # type: ignore
            join(target_dir, case, f"seg{case}_CT.nii.gz"),
            ct_seg_image.affine,  # type: ignore
        )
        save_nifti(
            joint_mask_mr.astype("uint8"),
            join(target_dir, case, f"roi_mask{case}_MR.nii.gz"),
            mr_roi_mask_image.affine,  # type: ignore
        )
        save_nifti(
            joint_mask_ct.astype("uint8"),
            join(target_dir, case, f"roi_mask{case}_CT.nii.gz"),
            ct_roi_mask_image.affine,  # type: ignore
        )
        save_nifti(
            foreground_mask_mr.astype("uint8"),
            join(target_dir, case, f"foreground_mask{case}_MR.nii.gz"),
            mr_image.affine,  # type: ignore
        )
        save_nifti(
            foreground_mask_ct.astype("uint8"),
            join(target_dir, case, f"foreground_mask{case}_CT.nii.gz"),
            ct_image.affine,  # type: ignore
        )

    def _load(self, data_folder: str, division: str) -> "RegistrationDataset":
        return CTMRThoraxAbdomenDataset(data_folder, division, mask_type=self._mask_type)

    @property
    def name(self) -> str:
        return f"{super().name}_{self._mask_type}"


class CTMRThoraxAbdomenDataset(RegistrationDataset):
    """Access CT-MR Thorax-Abdomen data."""

    def __init__(self, data_folder: str, division: str, mask_type: str = "foreground_mask") -> None:
        self._data_folder = data_folder
        self._division = division
        if mask_type not in ("roi_mask", "foreground_mask"):
            raise ValueError(f"Invalid mask type: {mask_type}")
        self._mask_type = mask_type

    def __len__(self) -> int:
        return len(CASES[self._division])

    def __getitem__(self, index: int) -> tuple[str, str, str, str | None, str | None]:
        case_name = CASES[self._division][index]
        return (
            case_name,
            join(self._data_folder, self._division, case_name, f"img{case_name}_MR.nii.gz"),
            join(self._data_folder, self._division, case_name, f"img{case_name}_CT.nii.gz"),
            join(
                self._data_folder,
                self._division,
                case_name,
                f"{self._mask_type}{case_name}_MR.nii.gz",
            ),
            join(
                self._data_folder,
                self._division,
                case_name,
                f"{self._mask_type}{case_name}_CT.nii.gz",
            ),
        )

    def evaluate(self, index: int, displacement_field: ndarray) -> Mapping[str, Any]:
        case_name = CASES[self._division][index]
        moving_segmentation_path = join(
            self._data_folder, self._division, case_name, f"seg{case_name}_CT.nii.gz"
        )
        reference_segmentation_image = nib_load(
            join(self._data_folder, self._division, case_name, f"seg{case_name}_MR.nii.gz")
        )
        reference_foreground_mask = nib_load(
            join(
                self._data_folder,
                self._division,
                case_name,
                f"foreground_mask{case_name}_MR.nii.gz",
            )
        ).dataobj[  # type: ignore
            ...
        ]
        deformed_moving_segmentation, _ = deform_image_from_path(
            moving_segmentation_path,
            displacement_field,
            reference_segmentation_image.affine,  # type: ignore
            interpolation_mode="nearest",
        )
        deformed_moving_segmentation = deformed_moving_segmentation.round().astype("int16")

        return {
            "tissue_overlap": compute_tissue_overlap_metrics(
                reference=reference_segmentation_image.dataobj[...],  # type: ignore
                registered=deformed_moving_segmentation,
                label_to_name=LABEL_TO_NAME,
                mask=reference_foreground_mask > 0,
            ),
            "regularity": compute_regularity_metrics(displacement_field),
        }

    def metrics_to_single_score(self, metrics: Mapping[str, Any]) -> float:
        return -metrics["tissue_overlap"]["dice"]["mean"]


def _main():
    parser = ArgumentParser(description="Generate CT-MR Thorax-Abdomen dataset.")
    parser.add_argument(
        "data_root",
        type=str,
        help="Path to the root directory where the generated dataset will be stored.",
    )
    args = parser.parse_args()

    CTMRThoraxAbdomenDatasetInitializer().build(args.data_root, "null")


if __name__ == "__main__":
    _main()
