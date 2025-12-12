"""
Module to train/eval/infer models with AutoGluon

WARNING: Ensure the model checkpoints are downloaded in the cache dir before running on a server without internet access.
$ huggingface-cli download Prior-Labs/TabPFN-v2-clf tabpfn-v2-classifier-finetuned-zk73skhh.ckpt --local-dir ~/.cache/tabpfn
$ huggingface-cli download jingang/TabICL-clf tabicl-classifier-v1.1-0506.ckpt
$ huggingface-cli download autogluon/mitra-classifier
"""
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import StratifiedGroupKFold

# Ref: https://huggingface.co/spaces/TabArena/leaderboard
# Ref: https://auto.gluon.ai/stable/api/autogluon.tabular.models.html
DEFAULT_HYPERPARAMS = {
    "RF": {}, # Random Forest
    "XT": {}, # Extra Trees
    "KNN": {}, # K-Nearest Neighbors
    "GBM": {}, # Light Gradient-Boosting Machine
    "CAT": {}, # Categorical Boosting
    "XGB": {}, # eXtreme Gradient Boosting
    "REALMLP": {}, # Real Multilayer Perceptron
    "TABM": {}, # Tabular DL model that makes Multiple predictions
    "LR": {
        "solver": "saga",
        "max_iter": 1000,
        "penalty": "l2",
        "tol": 1e-4,
    }, # Logistic Regression
    "ENS_WEIGHTED": {}, # Greedy Weighted Ensemble
    "SIMPLE_ENS_WEIGHTED": {}, # Simple Weighted Ensemble
}
# The models below are pre-trained transformer models that uses in-context learning
# It is only appropriate for small to medium datasets (<20k samples)
ICL_HYPERPARAMS = {
    "TABICL": {}, # Tabular In-Context Learning
    "TABPFNV2": {}, # Tabular Prior-Fitted Networks v2
    "MITRA": {'fine_tune': True}, # Mitra
}

###############################################################################
# Train
###############################################################################
def train_models(
    feats: pd.DataFrame, 
    targs: pd.DataFrame, 
    meta: pd.DataFrame, 
    **kwargs
) -> dict[str, TabularPredictor]:
    """Train models for all targets

    Args:
        feats: feature matrix, where each row is a data sample and each column is a feature
        targs: target matrix, where each row is a data sample and each column is a prediction target
        meta:  metadata associated with each sample. Must include a `mrn` column (medical record number).

    TODO: look into joblib for parallelization
    """
    data = feats.copy()
    if 'cv_folds' in meta:
        data['cv_folds'] = meta['cv_folds']
    else:
        data['mrn'] = meta['mrn']

    models = {}
    for target, label in targs.items():
        data[target] = label
        mask = label != -1
        models[target] = train_model(data[mask].copy(), target, **kwargs)
        data.pop(target)
    return models


def train_model(
    data: pd.DataFrame,
    target: str,
    eval_metric: str = "average_precision",
    presets: str = "medium_quality",
    calibrate: bool = False,
    refit_on_full_data: bool = False,
    time_limit: int = 10000, # seconds
    save_path: str = '.',
    resume: bool = False,
    extra_init_kwargs: dict | None = None,
    extra_fit_kwargs: dict | None = None,
) -> TabularPredictor:
    """
    Args:
        refit_on_full_data: If True, refit the model with the full training dataset at the end.
            Note the only difference between 'high' and 'best' preset is that 'high' refits on the full data, 'best'
            does not (as of 2024-12-17)
    """
    if extra_init_kwargs is None:
        extra_init_kwargs = {}
    if extra_fit_kwargs is None:
        extra_fit_kwargs = {'hyperparameters': DEFAULT_HYPERPARAMS}
        if len(data) < 30000:
            extra_fit_kwargs['hyperparameters'].update(ICL_HYPERPARAMS)

    quality = presets.replace("_quality", "")
    save_path = f"{save_path}/{target}"

    if "cv_folds" not in data:
        if 'mrn' not in data:
            raise ValueError("Please include mrn or cv_folds in the data")
        
        # create custom cross-validation folds based on mrn
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        kf_splits = kf.split(
            X=data,
            y=data[target],
            groups=data['mrn'],
        )
        cv_folds = np.zeros(len(data))
        for fold, (_, valid_idxs) in enumerate(kf_splits):
            cv_folds[valid_idxs] = fold
        data["cv_folds"] = cv_folds
        data.pop('mrn')

    # set up the training parameters
    init_kwargs = dict(
        log_to_file=True, 
        path=save_path, 
        eval_metric=eval_metric, 
        **extra_init_kwargs
    )
    fit_kwargs = dict(
        presets=presets,
        # feature_prune_kwargs={}, # mixed results with feature pruning
        # fit_weighted_ensemble=False,
        calibrate=calibrate,
        save_bag_folds=True,  # save the individual cross validation fold models
        time_limit=time_limit,
        refit_full=refit_on_full_data,  # refit the model on all of the data in the end
        set_best_to_refit_full=refit_on_full_data,
        **extra_fit_kwargs,
    )
    print(init_kwargs, fit_kwargs)

    if quality == "medium":
        # not using cross-validation, use the following as the tuning set
        mask = data.pop("cv_folds") == 0
        fit_kwargs["tuning_data"] = data[mask]
        data = data[~mask]
    else:
        # use our own cross validation folds
        init_kwargs["groups"] = "cv_folds"

    if resume:
        predictor = TabularPredictor.load(save_path).fit_extra(**extra_fit_kwargs)
    else:
        predictor = TabularPredictor(label=target, **init_kwargs).fit(data, **fit_kwargs)
    return predictor


###############################################################################
# Evaluate
###############################################################################
def evaluate(
    models: dict[str, TabularPredictor],
    feats: pd.DataFrame,
    targs: pd.DataFrame,
    return_full: bool = False,
) -> pd.DataFrame:
    """Evaluate performance for all targets and all model types

    Args:
        return_full: If True, return the full information about models (training times, inference times, stack levels, etc)
    """
    data = feats.copy()
    results = {}
    for target, label in targs.items():
        mask = label != -1
        if label[mask].nunique() == 1: 
            continue
        if target not in models:
            continue
        
        data[target] = label
        res = models[target].leaderboard(data[mask], extra_metrics=["roc_auc", "average_precision"])
        results[target] = res if return_full else res[["model", "roc_auc", "average_precision"]]
        data.pop(target)

    results = pd.concat(results, axis=1)
    return results


###############################################################################
# Predict
###############################################################################
def get_val_preds(models: dict[str, TabularPredictor]):
    """Get the predictions in the validation set for all targets and all model types"""
    res = {}
    for target, model in models.items():
        preds = {}
        for model_name in model.model_names():
            preds[model_name] = get_val_pred(model, model_name=model_name)
        res[target] = pd.DataFrame(preds).dropna().reset_index()
    res = pd.concat(res, axis=1)
    return res


def get_val_pred(model: TabularPredictor, model_name: str | None = None):
    """Get the predictions in the validation set"""
    if model._trainer.bagged_mode:  # used cross-validation
        return model.predict_proba_oof(model=model_name)[True]  # oof = out of fold
    else:
        x, y = model.load_data_internal(data="val")
        return model.predict_proba(x, model=model_name)[True]