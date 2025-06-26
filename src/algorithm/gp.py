"""GP related algorithms"""

from typing import cast

import numpy as np
import optuna
from optuna._gp import acqf, gp, prior
from optuna._gp.gp import posterior
from optuna._gp.search_space import ScaleType, get_search_space_and_normalized_params
from optuna.samplers._gp import GPSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from torch import Tensor, from_numpy, tensor


def evaluate_posterior_predictive(
    study: optuna.Study,
    deterministic_objective: bool = False,
    objective_index: int = 0,
    fit_to_quantile: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate posterior predictive distribution at sample locations"""
    search_space = GPSampler().infer_relative_search_space(study, None)  # type:ignore
    states = (TrialState.COMPLETE,)
    trials = study.get_trials(states=states)

    (
        internal_search_space,
        normalized_params,
    ) = get_search_space_and_normalized_params(trials, search_space)

    _sign = -1.0 if study.directions[objective_index] == StudyDirection.MINIMIZE else 1.0

    score_vals = np.array([_sign * cast(float, trial.values[objective_index]) for trial in trials])
    threshold = np.quantile(score_vals, fit_to_quantile)

    included_trials = score_vals > threshold
    score_vals = score_vals[included_trials]
    normalized_params = normalized_params[included_trials]

    if np.any(~np.isfinite(score_vals)):
        finite_score_vals = score_vals[np.isfinite(score_vals)]
        best_finite_score = np.max(finite_score_vals, initial=0.0)
        worst_finite_score = np.min(finite_score_vals, initial=0.0)

        score_vals = np.clip(score_vals, worst_finite_score, best_finite_score)

    score_vals_mean = score_vals.mean()
    score_vals_std = max(1e-10, score_vals.std())
    standarized_score_vals = (score_vals - score_vals_mean) / score_vals_std

    kernel_params = gp.fit_kernel_params(
        X=normalized_params,
        Y=standarized_score_vals,
        is_categorical=(internal_search_space.scale_types == ScaleType.CATEGORICAL),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic_objective,
    )
    acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
        kernel_params=kernel_params,
        search_space=internal_search_space,
        X=normalized_params,
        Y=standarized_score_vals,
    )
    mean, var = posterior(
        acqf_params.kernel_params,
        from_numpy(acqf_params.X),
        from_numpy(acqf_params.search_space.scale_types == ScaleType.CATEGORICAL),
        from_numpy(acqf_params.cov_Y_Y_inv),
        from_numpy(acqf_params.cov_Y_Y_inv_Y),
        from_numpy(normalized_params),
    )
    pred_mean = (mean * score_vals_std + score_vals_mean) * _sign
    pred_var = var * score_vals_std**2
    return (
        pred_mean,
        pred_var,
        from_numpy(included_trials),
    )


def get_best_posterior_trial(
    study: optuna.Study,
    deterministic_objective: bool = False,
    objective_index: int = 0,
    fit_to_quantile: float = 0.0,
) -> int:
    """Get the index of the best trial in the posterior predictive distribution"""
    all_indices = tensor(
        [trial.number for trial in study.get_trials(states=(TrialState.COMPLETE,))]
    )
    _sign = -1.0 if study.directions[objective_index] == StudyDirection.MINIMIZE else 1.0
    pred_mean, _pred_var, included_trials = evaluate_posterior_predictive(
        study,
        deterministic_objective=deterministic_objective,
        objective_index=objective_index,
        fit_to_quantile=fit_to_quantile,
    )
    pred_mean = pred_mean * _sign
    best_index = all_indices[included_trials][pred_mean.argmax()]
    return int(best_index)
