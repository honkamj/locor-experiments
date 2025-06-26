"""Execute study for a registration method with a dataset."""

from argparse import ArgumentParser, Namespace
from json import dump, load
from os import makedirs
from os.path import isdir, join
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from optuna import Study, Trial, create_study, load_study
from optuna.study import StudyDirection
from optuna.trial import TrialState

from algorithm.gp import get_best_posterior_trial
from application.inference import registration_inference
from application.interface import EvaluationApplicationDefinition
from data.interface import RegistrationDatasetInitializer
from method.interface import RegistrationMethod


def run_evaluation(
    available_applications: Sequence[EvaluationApplicationDefinition],
    available_datasets: Sequence[RegistrationDatasetInitializer],
) -> None:
    """Run the evaluation application."""
    parser = ArgumentParser()
    parser.add_argument("--target-folder", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--n-trials",
        type=int,
        required=True,
    )
    parser.add_argument("--save-deformed-images", action="store_true")
    parser.add_argument("--save-displacement-fields", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[dataset.name for dataset in available_datasets],
        action="append",
    )

    application_subparsers = parser.add_subparsers(dest="application")

    application: EvaluationApplicationDefinition

    for application_index, application in enumerate(available_applications):
        application_parser = application_subparsers.add_parser(application.method_name)
        application_parser.set_defaults(application_index=application_index)
        mode_subparsers = application_parser.add_subparsers(dest="mode")
        for mode in ["optimize_hyperparameters", "test"]:
            mode_parser = mode_subparsers.add_parser(mode)
            if mode == "test":
                mode_parser.add_argument(
                    "--validation-method-path",
                    type=str,
                    required=False,
                )
                mode_parser.add_argument(
                    "--validation-dataset",
                    type=str,
                    choices=[dataset.name for dataset in available_datasets],
                    action="append",
                    required=False,
                )
                mode_parser.add_argument(
                    "--test-division",
                    type=str,
                    default="test",
                    required=False,
                )
                mode_parser.add_argument(
                    "--fit-gp-to-quantile",
                    type=float,
                    required=False,
                )
                mode_parser.add_argument(
                    "--skip-existing-trials",
                    action="store_true",
                )
            application.build_subparser(mode, mode_parser)
    arguments = parser.parse_args()

    application = available_applications[arguments.application_index]
    dataset_initializers = [
        dataset for dataset in available_datasets if dataset.name in arguments.dataset
    ]

    if arguments.mode == "optimize_hyperparameters":
        target_folder = join(
            arguments.target_folder,
            application.method_path(arguments),
            "+".join(dataset.name for dataset in dataset_initializers),
            "validation",
        )
        sampler = application.build_validation_sampler(
            arguments, n_objectives=len(dataset_initializers)
        )
        makedirs(target_folder, exist_ok=True)
        study = create_study(
            sampler=sampler,
            study_name=(
                f"{application.method_path(arguments)}_"
                f"{'+'.join(dataset.name for dataset in dataset_initializers)}"
            ),
            storage=f"sqlite:///{join(arguments.target_folder, 'studies.db')}",
            directions=["minimize"] * len(dataset_initializers),
            load_if_exists=True,
        )
        execute_hyperparameter_optimization_study(
            study=study,
            method_builder=lambda trial: application.build_hyperparameter_optimization_method(
                trial, arguments
            ),
            dataset_initializers=dataset_initializers,
            target_folder=target_folder,
            data_root=arguments.data_root,
            study_optimize_kwargs=application.hyperparameter_optimization_study_optimize_kwargs(),
            n_trials=arguments.n_trials,
            save_deformed_images=arguments.save_deformed_images,
            save_displacement_fields=arguments.save_displacement_fields,
        )
    elif arguments.mode == "test":
        if len(dataset_initializers) > 1:
            raise ValueError("Only one dataset can be used for testing.")
        validation_dataset_initializers = (
            dataset_initializers
            if arguments.validation_dataset is None
            else [
                dataset
                for dataset in available_datasets
                if dataset.name in arguments.validation_dataset
            ]
        )
        validation_dataset_names = [
            dataset.name for dataset in validation_dataset_initializers
        ]
        validation_method_path = (
            application.method_path(arguments)
            if arguments.validation_method_path is None
            else arguments.validation_method_path
        )
        validation_study = load_study(
            study_name=(
                f"{validation_method_path}_"
                f"{'+'.join(name for name in validation_dataset_names)}"
            ),
            storage=f"sqlite:///{join(arguments.target_folder, 'studies.db')}",
        )
        dataset_index = validation_dataset_names.index(dataset_initializers[0].name)
        if arguments.fit_gp_to_quantile is not None:
            best_trial_index = get_best_posterior_trial(
                study=validation_study,
                objective_index=dataset_index,
                fit_to_quantile=arguments.fit_gp_to_quantile,
                deterministic_objective=application.deterministic_objective,
            )
        else:
            trials = validation_study.get_trials(states=(TrialState.COMPLETE,))
            trial_indices = [trial.number for trial in trials]
            objective_sign = (
                -1.0
                if validation_study.directions[dataset_index] == StudyDirection.MINIMIZE
                else 1.0
            )
            objective_values = [
                objective_sign * cast(float, trial.values[dataset_index]) for trial in trials
            ]
            best_trial_index = trial_indices[int(np.argmax(objective_values))]
        validation_folder = join(
            arguments.target_folder,
            validation_method_path,
            "+".join(dataset.name for dataset in validation_dataset_initializers),
            "validation",
        )
        target_folder = join(
            arguments.target_folder,
            application.method_path(arguments),
            dataset_initializers[0].name,
            "test",
        )
        test_method, parameters = _build_test_method(
            best_trial_index=best_trial_index,
            application=application,
            validation_folder=validation_folder,
            args=arguments,
        )
        makedirs(target_folder, exist_ok=True)
        with open(
            join(target_folder, "parameters.json"), mode="w", encoding="utf-8"
        ) as method_parameters_file:
            saved_parameters = {
                "method_parameters": parameters,
                "validation_trial": best_trial_index,
            }
            dump(saved_parameters, method_parameters_file, indent=4)
        execute_test_study(
            test_method,
            dataset_initializer=dataset_initializers[0],
            division=arguments.test_division,
            target_folder=target_folder,
            data_root=arguments.data_root,
            n_trials=arguments.n_trials,
            skip_existing_trials=arguments.skip_existing_trials,
            save_deformed_images=arguments.save_deformed_images,
            save_displacement_fields=arguments.save_displacement_fields,
        )
    else:
        raise ValueError(f"Unknown mode: {arguments.mode}")


def _build_test_method(
    best_trial_index: int,
    application: EvaluationApplicationDefinition,
    validation_folder: str,
    args: Namespace,
) -> tuple[RegistrationMethod, Mapping[str, Any]]:
    with open(
        join(validation_folder, f"trial_{best_trial_index}", "method_parameters.json"),
        mode="r",
        encoding="utf-8",
    ) as best_parameters_file:
        best_parameters = load(best_parameters_file)
    method = application.build_test_method(best_parameters, args)
    return method, best_parameters


def execute_hyperparameter_optimization_study(
    study: Study,
    method_builder: Callable[[Trial], RegistrationMethod],
    dataset_initializers: Sequence[RegistrationDatasetInitializer],
    target_folder: str,
    data_root: str,
    n_trials: int,
    study_optimize_kwargs: Mapping[str, Any],
    save_deformed_images: bool = False,
    save_displacement_fields: bool = False,
) -> None:
    """Run a trial for a registration method with a dataset."""

    datasets = [
        dataset.build(data_root, "validation") for dataset in dataset_initializers
    ]
    dataset_names = [dataset.name for dataset in dataset_initializers]

    def _objective(trial: Trial) -> tuple[float, ...]:
        method = method_builder(trial)
        trial_folder = join(target_folder, f"trial_{trial.number}")
        makedirs(trial_folder, exist_ok=True)
        with open(
            join(trial_folder, "method_parameters.json"), mode="w", encoding="utf-8"
        ) as method_parameters_file:
            dump(method.parameters, method_parameters_file, indent=4)
        print(f"Current parameters: {method.parameters}")
        return tuple(
            registration_inference(
                method,
                dataset,
                join(trial_folder, name),
                save_displacement_field=save_displacement_fields,
                save_deformed_image=save_deformed_images,
            )
            for dataset, name in zip(datasets, dataset_names)
        )

    results_file_name: str | None = None
    try:
        n_earlier_trials = len(
            [
                trial
                for trial in study.get_trials()
                if trial.state == TrialState.COMPLETE
            ]
        )
        extended_study_optimize_kwargs: Any = {
            "n_trials": n_trials - n_earlier_trials,
        } | dict(study_optimize_kwargs)
        study.optimize(_objective, **extended_study_optimize_kwargs, catch=(Exception,))
        results_file_name = "results.json"
    except BaseException:
        results_file_name = f"results_after_trial_{study.get_trials()[-1].number}.json"
    finally:
        best_trials = []
        for trial in study.best_trials:
            with open(
                join(target_folder, f"trial_{trial.number}", "method_parameters.json"),
                mode="r",
                encoding="utf-8",
            ) as best_trial_parameters_file:
                best_trial_parameters = load(best_trial_parameters_file)
            best_trials.append(
                {
                    "number": trial.number,
                    "parameters": best_trial_parameters,
                    "values": trial.values,
                }
            )

        assert results_file_name is not None
        with open(
            join(target_folder, results_file_name),
            mode="w",
            encoding="utf-8",
        ) as results_file:
            dump(
                {"best_trials": best_trials},
                results_file,
                indent=4,
            )


def execute_test_study(
    method: RegistrationMethod,
    dataset_initializer: RegistrationDatasetInitializer,
    division: str,
    target_folder: str,
    data_root: str,
    n_trials: int,
    skip_existing_trials: bool = True,
    save_deformed_images: bool = False,
    save_displacement_fields: bool = False,
) -> None:
    """Run a test study for a registration method with a dataset."""
    dataset = dataset_initializer.build(data_root, division)

    for trial_number in range(n_trials):
        if isdir(join(target_folder, f"trial_{trial_number}")) and skip_existing_trials:
            continue
        registration_inference(
            method,
            dataset,
            join(target_folder, f"trial_{trial_number}"),
            save_displacement_field=save_displacement_fields,
            save_deformed_image=save_deformed_images,
        )
