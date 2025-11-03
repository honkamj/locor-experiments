"""Base evaluation applications for different methods."""

from argparse import ArgumentParser, Namespace
from typing import Any, Mapping

from optuna import Trial

from application.interface import EvaluationApplicationDefinition
from method.ants import ANTs
from method.corrfield import Corrfield
from method.locor import Locor
from method.locor_ablation_study import LocorAblationStudy
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
            "affine_learning_rate": trial.suggest_float(
                "affine_learning_rate", 1e-3, 1e-1, log=True
            ),
            "base_dense_learning_rate": trial.suggest_float(
                "base_dense_learning_rate", 1e-3, 1e-1, log=True
            ),
            "feature_learning_rate": trial.suggest_float(
                "feature_learning_rate", 1e-3, 1e-1, log=True
            ),
            "sliding_window_std": trial.suggest_float("sliding_window_std", 0.35, 4.0),
            "n_features": trial.suggest_int("n_features", 1, 5),
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


class LocorAblationStudyApplication(EvaluationApplicationDefinition):
    """Locor ablation study evaluation application."""

    def _build_parameters(
        self, trial: Trial, args: Namespace  # pylint: disable=unused-argument
    ) -> Mapping[str, Any]:
        params = {
            "regularization_weight": trial.suggest_float(
                "regularization_weight", 1e-1, 1e4, log=True
            ),
            "affine_learning_rate": trial.suggest_float(
                "affine_learning_rate", 1e-3, 1e-1, log=True
            ),
            "base_dense_learning_rate": trial.suggest_float(
                "base_dense_learning_rate", 1e-3, 1e-1, log=True
            ),
        }
        if not args.use_mind_ssc and not args.use_mi:
            params = params | {
                "sliding_window_std": trial.suggest_float("sliding_window_std", 0.35, 4.0),
            }
            if args.n_local_correlation_ratio_bins is None:
                if args.do_not_use_learned_features:
                    if args.do_not_use_derivatives:
                        params = params | {"n_features": trial.suggest_int("n_features", 1, 5)}
                    else:
                        params = params | {"n_features": trial.suggest_int("n_features", 1, 3)}
                else:
                    params = params | {
                        "n_features": trial.suggest_int("n_features", 1, 5),
                        "feature_learning_rate": trial.suggest_float(
                            "feature_learning_rate", 1e-3, 1e-1, log=True
                        ),
                    }
        if args.use_mi:
            params = params | {
                "mi_bins": trial.suggest_int("mi_bins", 10, 24),
                "mi_quantile": trial.suggest_float("mi_quantile", 0.0, 0.05),
            }
        return params

    def build_hyperparameter_optimization_method(
        self, trial: Trial, args: Namespace
    ) -> LocorAblationStudy:
        return LocorAblationStudy(
            parameters=dict(self._build_parameters(trial, args))
            | {
                "n_local_correlation_ratio_bins": args.n_local_correlation_ratio_bins,
                "use_derivatives": not args.do_not_use_derivatives,
                "use_gaussian_window": not args.do_not_use_gaussian_window,
                "use_log_similarity": not args.do_not_use_log_similarity,
                "use_learned_features": not args.do_not_use_learned_features,
                "use_mind_ssc": args.use_mind_ssc,
                "use_mi": args.use_mi,
            },
            devices=args.device,
        )

    def build_test_method(
        self, best_parameters: Mapping[str, Any], args: Namespace
    ) -> LocorAblationStudy:
        return LocorAblationStudy(parameters=best_parameters, devices=args.device)

    def build_subparser(self, mode: str, subparser: ArgumentParser) -> None:
        super().build_subparser(mode, subparser)
        subparser.add_argument("--device", type=str, action="append")
        subparser.add_argument(
            "--n-local-correlation-ratio-bins", type=int, required=False, default=None
        )
        subparser.add_argument(
            "--do-not-use-derivatives", action="store_true", required=False, default=False
        )
        subparser.add_argument(
            "--do-not-use-gaussian-window", action="store_true", required=False, default=False
        )
        subparser.add_argument(
            "--do-not-use-log-similarity", action="store_true", required=False, default=False
        )
        subparser.add_argument(
            "--do-not-use-learned-features", action="store_true", required=False, default=False
        )
        subparser.add_argument("--use-mind-ssc", action="store_true", required=False, default=False)
        subparser.add_argument("--use-mi", action="store_true", required=False, default=False)

    @property
    def method_name(self) -> str:
        return "locor_ablation_study"

    def method_path(self, args: Namespace) -> str:
        ablations = []
        if args.n_local_correlation_ratio_bins is not None:
            ablations.append(f"local_correlation_ratio_{args.n_local_correlation_ratio_bins}")
        if args.do_not_use_derivatives:
            ablations.append("no_derivatives")
        if args.do_not_use_gaussian_window:
            ablations.append("no_gaussian_window")
        if args.do_not_use_log_similarity:
            ablations.append("no_log_similarity")
        if args.do_not_use_learned_features:
            ablations.append("no_learned_features")
        if args.use_mind_ssc:
            ablations.append("mind_ssc")
        if args.use_mi:
            ablations.append("mi")
        return f"{self.method_name}{'_' if ablations else ''}{'_'.join(ablations)}"


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
            "affine_learning_rate": trial.suggest_float(
                "affine_learning_rate", 1e-3, 1e-1, log=True
            ),
            "base_dense_learning_rate": trial.suggest_float(
                "base_dense_learning_rate", 1e-3, 1e-1, log=True
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
                "bending_energy_weight": 0.0001026207225458064,
                "first_order_penalty_weight": 0.002338142723107715,
                "velocity_field": False,
                "mind_offset": 1.0,
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
