"""Base evaluation applications for different methods."""

from argparse import ArgumentParser, Namespace
from typing import Any, Mapping

from optuna import Trial

from application.interface import EvaluationApplicationDefinition
from method.ants import ANTs
from method.corrfield import Corrfield
from method.locor import Locor
from method.niftyreg import NiftyRegMIND, NiftyRegNMI
from method.SRWCR import SRWCR


class LocorApplication(EvaluationApplicationDefinition):
    """Locor evaluation application."""

    def _build_parameters(
        self, trial: Trial, args: Namespace  # pylint: disable=unused-argument
    ) -> Mapping[str, Any]:
        return {
            "regularization_weight": trial.suggest_float(
                "regularization_weight", 1e-1, 1e4, log=True
            ),
            "sliding_window_std": trial.suggest_float("sliding_window_std", 0.5, 6.0),
            "n_features": trial.suggest_int("n_features", 2, 5),
        }

    def build_hyperparameter_optimization_method(self, trial: Trial, args: Namespace) -> Locor:
        return Locor(
            parameters=self._build_parameters(trial, args),
            devices=args.device,
        )

    def build_test_method(self, best_parameters: Mapping[str, Any], args: Namespace) -> Locor:
        return Locor(parameters=best_parameters, devices=args.device)

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        super().build_subparser(mode, subparser)
        subparser.add_argument("--device", type=str, action="append")

    @property
    def method_name(self) -> str:
        return "locor"


class SRWCRApplication(EvaluationApplicationDefinition):
    """SRWCR evaluation application."""

    def _build_parameters(
        self, trial: Trial, args: Namespace  # pylint: disable=unused-argument
    ) -> Mapping[str, Any]:
        return {
            "regularization_weight": trial.suggest_float(
                "regularization_weight", 1e-1, 1e4, log=True
            ),
            "sliding_window_stride": trial.suggest_int("sliding_window_stride", 3, 5),
            "base_dense_learning_rate": trial.suggest_float(
                "base_dense_learning_rate", 1e-2, 5e-2, log=True
            ),
        }

    def build_hyperparameter_optimization_method(self, trial: Trial, args: Namespace) -> SRWCR:
        return SRWCR(
            parameters=self._build_parameters(trial, args),
            devices=args.device,
        )

    def build_test_method(self, best_parameters: Mapping[str, Any], args: Namespace) -> SRWCR:
        return SRWCR(parameters=best_parameters, devices=args.device)

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        super().build_subparser(mode, subparser)
        subparser.add_argument("--device", type=str, action="append")

    @property
    def method_name(self) -> str:
        return "SRWCR"


class NiftyRegNMIApplication(EvaluationApplicationDefinition):
    """NiftyReg NMI evaluation application."""

    def build_hyperparameter_optimization_method(
        self, trial: Trial, args: Namespace
    ) -> NiftyRegNMI:
        return NiftyRegNMI(
            parameters={
                "bending_energy_weight": trial.suggest_float(
                    "bending_energy_weight", 1e-4, 1e-2, log=True
                ),
                "first_order_penalty_weight": trial.suggest_float(
                    "first_order_penalty_weight", 1e-3, 1e-1, log=True
                ),
                "velocity_field": False,
            },
            niftyreg_path=args.niftyreg_path,
            n_threads=args.n_threads,
        )

    def build_test_method(self, best_parameters: Mapping[str, Any], args: Namespace) -> NiftyRegNMI:
        return NiftyRegNMI(
            parameters=best_parameters, niftyreg_path=args.niftyreg_path, n_threads=args.n_threads
        )

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        super().build_subparser(mode, subparser)
        subparser.add_argument("--niftyreg-path", type=str, required=False, default="")
        subparser.add_argument("--n-threads", type=int, required=False, default=None)

    @property
    def method_name(self) -> str:
        return "NiftyReg_NMI"

    @property
    def deterministic_objective(self) -> bool:
        return True


class NiftyRegMINDApplication(EvaluationApplicationDefinition):
    """NiftyReg MIND evaluation application."""

    def build_hyperparameter_optimization_method(
        self, trial: Trial, args: Namespace
    ) -> NiftyRegMIND:
        return NiftyRegMIND(
            parameters={
                "bending_energy_weight": trial.suggest_float(
                    "bending_energy_weight", 1e-4, 1e-2, log=True
                ),
                "first_order_penalty_weight": trial.suggest_float(
                    "first_order_penalty_weight", 1e-3, 1e-1, log=True
                ),
                "mind_offset": trial.suggest_int("mind_offset", 1, 2),
                "velocity_field": False,
            },
            niftyreg_path=args.niftyreg_path,
            n_threads=args.n_threads,
        )

    def build_test_method(
        self, best_parameters: Mapping[str, Any], args: Namespace
    ) -> NiftyRegMIND:
        return NiftyRegMIND(
            parameters=best_parameters, niftyreg_path=args.niftyreg_path, n_threads=args.n_threads
        )

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        super().build_subparser(mode, subparser)
        subparser.add_argument("--niftyreg-path", type=str, required=False, default="")
        subparser.add_argument("--n-threads", type=int, required=False, default=None)

    @property
    def method_name(self) -> str:
        return "NiftyReg_MIND"

    @property
    def deterministic_objective(self) -> bool:
        return True


class ANTsApplication(EvaluationApplicationDefinition):
    """ANTs evaluation applicationn."""

    def build_hyperparameter_optimization_method(self, trial: Trial, args: Namespace) -> ANTs:
        return ANTs(
            parameters={
                "rigid_stepsize": trial.suggest_float(
                    "rigid_stepsize",
                    0.1,
                    2.0,
                ),
                "affine_stepsize": trial.suggest_float(
                    "affine_stepsize",
                    0.1,
                    2.0,
                ),
                "syn_stepsize": trial.suggest_float(
                    "syn_stepsize",
                    0.1,
                    2.0,
                ),
                "windowing_quantile": trial.suggest_float(
                    "windowing_quantile",
                    0.0,
                    0.05,
                ),
                "update_field_variance": trial.suggest_float("update_field_variance", 0.5, 8.0),
                "total_field_variance": trial.suggest_float("total_field_variance", 0.0, 2.0),
                "rigid_affine_max_n_last_stage_iterations": trial.suggest_int(
                    "rigid_affine_max_n_last_stage_iterations", 50, 200
                ),
                "syn_max_n_last_stage_iterations": trial.suggest_int(
                    "syn_max_n_last_stage_iterations", 20, 80
                ),
            },
            ants_path=args.ants_path,
            n_threads=args.n_threads,
        )

    def build_test_method(self, best_parameters: Mapping[str, Any], args: Namespace) -> ANTs:
        return ANTs(parameters=best_parameters, ants_path=args.ants_path, n_threads=args.n_threads)

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        super().build_subparser(mode, subparser)
        subparser.add_argument("--ants-path", type=str, required=False, default="")
        subparser.add_argument("--n-threads", type=int, required=False, default=None)

    @property
    def method_name(self) -> str:
        return "ANTs"


class CorrfieldApplication(EvaluationApplicationDefinition):
    """Corrfield evaluation application."""

    def build_hyperparameter_optimization_method(self, trial: Trial, args: Namespace) -> Corrfield:
        return Corrfield(
            parameters={
                "alpha": trial.suggest_float("alpha", 1e-1, 1e2, log=True),
                "beta": trial.suggest_float("beta", 1e0, 1e3, log=True),
                "gamma": trial.suggest_float("gamma", 1, 10),
                "delta": trial.suggest_int("delta", 1, 2),
                "last_stage_search_radius": trial.suggest_int("last_stage_search_radius", 4, 16),
            },
            corrfield_path=args.corrfield_path,
        )

    def build_test_method(self, best_parameters: Mapping[str, Any], args: Namespace) -> Corrfield:
        return Corrfield(parameters=best_parameters, corrfield_path=args.corrfield_path)

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        super().build_subparser(mode, subparser)
        subparser.add_argument("--corrfield-path", type=str, required=False, default="")

    @property
    def method_name(self) -> str:
        return "corrfield"
