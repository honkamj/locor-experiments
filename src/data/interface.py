"""Interface for data processing."""

from abc import ABC, abstractmethod
from datetime import datetime
from os import listdir, makedirs
from os.path import isdir, isfile, join
from typing import Any, Iterator, Mapping

from numpy import ndarray


class RegistrationDatasetInitializer(ABC):
    """Interface for initializing registration datasets."""

    def __init__(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name

    @abstractmethod
    def _build(self, data_folder: str) -> None:
        """Build the dataset."""

    @abstractmethod
    def _load(self, data_folder: str, division: str) -> "RegistrationDataset":
        """Load the dataset."""

    @abstractmethod
    def _licence_agreement_question(self) -> str | None:
        """Licence agreement question."""

    def build(self, data_root: str, division: str) -> "RegistrationDataset":
        """Build the registration dataset."""
        data_folder = join(data_root, self._dataset_name)
        if not self._is_built(data_folder):
            self._ensure_target_folder_empty(data_folder)
            self._create_data_folder(data_folder)
            license_agreement_question = self._licence_agreement_question()
            if license_agreement_question is not None:
                licece_agreement_question_answer = input(
                    self._licence_agreement_question()
                )
                if licece_agreement_question_answer.lower() != "yes":
                    raise RuntimeError(
                        "You must agree to the licence agreement to use the dataset."
                    )
            self._build(data_folder)
            self._write_timestamp(data_folder)
        return self._load(data_folder, division)

    @staticmethod
    def _is_built(data_folder: str) -> bool:
        return isfile(join(data_folder, "timestamp.txt"))

    @staticmethod
    def _ensure_target_folder_empty(data_folder: str) -> None:
        if isdir(data_folder):
            if len(listdir(data_folder)) != 0:
                raise RuntimeError(f"Target directory {data_folder} is not empty.")

    @staticmethod
    def _create_data_folder(data_folder: str) -> None:
        makedirs(data_folder, exist_ok=True)

    @staticmethod
    def _write_timestamp(data_folder: str) -> None:
        with open(
            join(data_folder, "timestamp.txt"), mode="w", encoding="utf-8"
        ) as timestamp_file:
            timestamp_file.write(str(datetime.now()))

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self._dataset_name


class RegistrationDataset(ABC):
    """Interface for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[str, str, str, str | None, str | None]:
        """Return name of the case, and the paths of the reference image, moving
        image, reference mask, and moving mask."""

    def __iter__(self) -> Iterator[tuple[str, str, str, str | None, str | None]]:
        for index in range(len(self)):
            yield self[index]

    @abstractmethod
    def evaluate(
        self,
        index: int,
        displacement_field: ndarray,
    ) -> Mapping[str, Any]:
        """Evaluate the registration result."""

    @abstractmethod
    def metrics_to_single_score(self, metrics: Mapping[str, Any]) -> float:
        """Convert metrics to a single score which can be used as a target for
        hyperparameter optimization."""
