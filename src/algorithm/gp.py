"""GP related algorithms"""

from functools import partial
from logging import warning
from typing import Callable, Sequence

import numpy as np
import optuna
import optunahub
import torch
from hebo.models.base_model import BaseModel  # type: ignore
from hebo.models.model_factory import get_model  # type: ignore
from hebo.optimizers.hebo import HEBO  # type: ignore
from sklearn.preprocessing import PowerTransformer  # type: ignore
from torch import Tensor


def _inverse_transform(
    values: Tensor, power_transformer: PowerTransformer, std: np.ndarray
) -> Tensor:
    return values.new_tensor(power_transformer.inverse_transform(values.numpy(force=True)) * std)


def _get_gp_model(hebo_sampler: HEBO) -> tuple[BaseModel, Callable[[Tensor], Tensor]]:
    xc, xe = hebo_sampler.space.transform(hebo_sampler.X)
    try:
        std = hebo_sampler.y.std()
        if hebo_sampler.y.min() <= 0:
            power_transformer = PowerTransformer(method="yeo-johnson")
            y: torch.Tensor = torch.FloatTensor(
                power_transformer.fit_transform(hebo_sampler.y / std)
            )
        else:
            power_transformer = PowerTransformer(method="box-cox")
            y = torch.FloatTensor(power_transformer.fit_transform(hebo_sampler.y / std))
            if y.std() < 0.5:
                power_transformer = PowerTransformer(method="yeo-johnson")
                y = torch.FloatTensor(power_transformer.fit_transform(hebo_sampler.y / y.std()))
        inverse_transform: Callable[[Tensor], Tensor] = partial(
            _inverse_transform, power_transformer=power_transformer, std=std
        )
        if y.std() < 0.5:
            raise RuntimeError("Power transformation failed")
        model = get_model(
            hebo_sampler.model_name,
            hebo_sampler.space.num_numeric,
            hebo_sampler.space.num_categorical,
            1,
            **hebo_sampler.model_config,
        )
        model.fit(xc, xe, y)
    except Exception:
        inverse_transform = lambda x: x  # pylint: disable=unnecessary-lambda-assignment
        y = torch.FloatTensor(hebo_sampler.y).clone()
        model = get_model(
            hebo_sampler.model_name,
            hebo_sampler.space.num_numeric,
            hebo_sampler.space.num_categorical,
            1,
            **hebo_sampler.model_config,
        )
        model.fit(xc, xe, y)
    return model, inverse_transform


def evaluate_posterior_predictive(
    study: optuna.Study,
    objective_index: int = 0,
) -> tuple[Tensor, Tensor, Callable[[Tensor], Tensor], Sequence[int]]:
    """Evaluate posterior predictive distribution at sample locations"""
    if len(study.directions) > 1:
        raise ValueError(
            "Posterior predictive evaluation is only supported for single-objective studies."
        )
    if objective_index != 0:
        raise ValueError(f"Invalid objective index {objective_index}.")
    optuna_search_space = optuna.search_space.IntersectionSearchSpace().calculate(study)
    optuna_hebo_module = optunahub.load_module("samplers/hebo")
    optuna_hebo_sampler = optuna_hebo_module.HEBOSampler(search_space=optuna_search_space)
    hebo_sampler: HEBO = optuna_hebo_sampler._hebo  # pylint: disable=protected-access
    trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    optuna_hebo_sampler._transform_to_dict_and_observe(  # pylint: disable=protected-access
        hebo_sampler,
        optuna_search_space,
        study,
        trials,
    )
    trial_indices = [trial.number for trial in trials]
    gp_model, inverse_transform = _get_gp_model(hebo_sampler)
    xc, xe = hebo_sampler.space.transform(hebo_sampler.X)
    posterior_predictive_mean, posterior_predictive_variance = gp_model.predict(xc, xe)
    return (
        posterior_predictive_mean,
        posterior_predictive_variance,
        inverse_transform,
        trial_indices,
    )


def get_best_posterior_trials(
    study: optuna.Study,
    objective_index: int = 0,
    n_best_trials: int = 1,
    repeats: int = 50,
) -> Sequence[int]:
    """Get the index of the best trial in terms of posterior mean

    Args:
        study (optuna.Study): The Optuna study to evaluate.
        objective_index (int, optional): The index of the objective to consider. Defaults to 0.
        repeats (int, optional): Number of repetitions for posterior predictive
            evaluation. Defaults to 50. GP fitting involves (very small)
            randomness, so we repeat the evaluation to get a more stable
            estimate of the posterior mean.
    """
    means = []
    non_inverse_transformed_means = []
    do_not_use_inverse_transform = False
    for _ in range(repeats):
        (
            posterior_predictive_mean,
            _posterior_predictive_variance,
            inverse_transform,
            trial_indices,
        ) = evaluate_posterior_predictive(
            study,
            objective_index=objective_index,
        )
        means.append(inverse_transform(posterior_predictive_mean))
        non_inverse_transformed_means.append(posterior_predictive_mean)
        if means[-1].isnan().any():
            do_not_use_inverse_transform = True
    if do_not_use_inverse_transform:
        means = non_inverse_transformed_means
        warning(
            "Warning: Inverse transformation of the GP output failed. "
            "Using non-inverse-transformed values to select the best trials. "
            "This is very unlikely to cause any issues."
        )
    posterior_predictive_mean = torch.stack(means, dim=0).mean(dim=0)
    best_trial_indices = torch.argsort(posterior_predictive_mean, dim=0, stable=True)
    return [trial_indices[int(best_trial_indices[order, 0])] for order in range(n_best_trials)]
