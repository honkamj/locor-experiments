"""Interfaces for evaluation applications."""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, Mapping

from optuna import Trial
from optuna.samplers import BaseSampler, GPSampler, TPESampler

from data.interface import RegistrationDatasetInitializer
from method.interface import RegistrationMethod


class EvaluationApplicationDefinition(ABC):
    """Interface for defining an evaluation application"""

    @abstractmethod
    def build_hyperparameter_optimization_method(
        self, trial: Trial, args: Namespace
    ) -> RegistrationMethod:
        """Build registration method for the given hyperparameter optimization
        trial and arguments."""

    @abstractmethod
    def build_test_method(
        self, best_parameters: Mapping[str, Any], args: Namespace
    ) -> RegistrationMethod:
        """Build registration method for the given parameters and arguments."""

    def build_validation_sampler(
        self, args: Namespace, n_objectives: int  # pylint: disable=unused-argument
    ) -> BaseSampler:
        """Build a sampler for the given arguments."""
        if n_objectives > 1:
            return TPESampler()
        return GPSampler(deterministic_objective=self.deterministic_objective)

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        """Build the subparser for the given application mode."""

    def hyperparameter_optimization_study_optimize_kwargs(self) -> Mapping[str, Any]:
        """Provide optima.Study.optimize kwargs for executing the hyperparameter
        optimization study."""
        return {}

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Name of the registration method."""

    def method_path(self, args: Namespace) -> str:  # pylint: disable=unused-argument
        """Path to the registration method in the results folder."""
        return self.method_name

    @property
    def deterministic_objective(self) -> bool:
        """Whether the objective is deterministic."""
        return False


class EvaluationDataset(ABC):
    """Interface for defining an evaluation dataset."""

    @abstractmethod
    def build_dataset_initializer(self, args: Namespace) -> RegistrationDatasetInitializer:
        """Build the dataset initializer for the given arguments."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset."""

    def dataset_path(self, args: Namespace) -> str:  # pylint: disable=unused-argument
        """Path to the dataset in the results folder."""
        return self.dataset_name
