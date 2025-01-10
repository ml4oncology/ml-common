"""
Module to evaluate system performance
"""
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, 
    confusion_matrix,
    precision_recall_curve, 
    precision_score,
    roc_auc_score,
    roc_curve, 
)
from tqdm import tqdm


def get_model_performance(
    df: pd.DataFrame, 
    label_col: str, 
    pred_col: str, 
    pred_thresh: float = 0.2,
    main_date_col: str = 'assessment_date', 
    event_date_col: str = 'event_date', 
) -> dict[str, str]:
    """Evaluate model performance across all metrics for a given threshold.
    """
    labels, pred_probs = df[label_col], df[pred_col]

    metrics = get_perf_at_operating_point(
        df, label_col, pred_col, pred_thresh, 
        main_date_col=main_date_col, event_date_col=event_date_col
    )
    # convert to string and round to the 3rd decimal place
    metrics = {metric: f'{val:.3f}' for metric, val in metrics.items()}

    ci = auc_confidence_interval(labels, pred_probs)
    [auroc_lower, auprc_lower], [auroc_upper, auprc_upper] = ci.to_numpy()
    metrics.update(auc_scores(labels, pred_probs))
    metrics['AUROC'] = f'{metrics["AUROC"]:.3f} ({auroc_lower:.3f}-{auroc_upper:.3f})'
    metrics['AUPRC'] = f'{metrics["AUPRC"]:.3f} ({auprc_lower:.3f}-{auprc_upper:.3f})'
    
    return metrics


def get_perf_at_operating_point(df, targ_col: str, pred_col: str, threshold: float, **kwargs):
    df['alarm'] = df[pred_col] >= threshold
    res = {'Threshold': threshold}
    res['Warning rate'] = df['alarm'].mean()
    res['First-alarm precision'] = first_alarm_precision(df, targ_col)
    res['Outcome-level recall'] = outcome_level_sensitivity(df, **kwargs)

    TN, FP, FN, TP = confusion_matrix(df[targ_col], df['alarm']).ravel()
    res['Accuracy'] = (TP + TN) / (TP + FP + FN + TN)
    res['Precision'] = TP / (TP + FP) # aka PPV
    res['Recall'] = TP / (TP + FN) # aka Sensitivity, Hit Rate, True Positive Rate
    res['Specificity'] = TN / (TN + FP) # True Negative Rate
    res['NPV'] = TN / (TN + FN) # Negative Predictive Value
    res['FPR'] = FP / (FP + TN) # False Positive Rate aka Fall Out
    res['FNR'] = FN / (TP + FN) # False Negative Rate
    res['FDR'] = FP / (TP + FP) # False Discovery Rate

    del df['alarm']
    return res


def outcome_level_sensitivity(
    df: pd.DataFrame, 
    lookahead_window: int = 90,
    main_date_col: str = 'assessment_date', 
    event_date_col: str = 'event_date', 
):
    """Get the proportion of true outcomes where at least one alarm preceded the event

    E.g. if ED visit happens on Jan 20, our lookahead window is 30 days, and assessments 
        happens on Jan 1 and Jan 14, then the outcome-level true positive is if 
        either Jan 1 or Jan 14 trigger a warning, and false negative if neither do
    """
    # ensure all event dates occur within X days of an assessment date
    mask = df[event_date_col].notna()
    diff = (df.loc[mask, event_date_col] - df.loc[mask, main_date_col]).dt.days
    assert all(diff.between(0, lookahead_window))

    result = df.groupby(['mrn', event_date_col])['alarm'].any()
    return sum(result) / len(result) # tp / (tp + fn)


def first_alarm_precision(df: pd.DataFrame, target_col: str):
    """Get the proportion of first alarms that were true
    """
    df = df.query('alarm').groupby('mrn').first()
    return precision_score(df[target_col], df['alarm'], zero_division=0)


def first_alarm_precision_outcome_level_recall_curve(df: pd.DataFrame, target_col: str, pred_col: str = 'pred'):
    first_alarm_ppv = [] # first-alarm positive predictive value (PPV) / first-alarm precision
    outcome_level_recall = [] # outcome-level sensitivity / outcome-level recall
    _, _, thresholds = precision_recall_curve(df[target_col], df[pred_col])

    for threshold in tqdm(thresholds):
        df['alarm'] = df[pred_col] >= threshold

        # Compute first-alarm precision
        score = first_alarm_precision(df, target_col=target_col)
        first_alarm_ppv.append(score)

        # Compute outcome-level sensitivity
        score = outcome_level_sensitivity(df)
        outcome_level_recall.append(score)

    del df['alarm']
    return first_alarm_ppv, outcome_level_recall, thresholds


def auc_scores(labels, preds) -> dict[str, float]:
    if len(set(labels)) == 1:
        return {"AUPRC": np.nan, "AUROC": np.nan}
    
    return {
        'AUROC': roc_auc_score(labels, preds),
        'AUPRC': average_precision_score(labels, preds)
    }


###############################################################################
# Confidence Interval
###############################################################################
def auc_confidence_interval(
    labels: pd.Series, 
    pred_probs: pd.Series, 
    n_bootstraps: int = 1000, 
    q: tuple[float] = (0.025, 0.975)
) -> pd.DataFrame:
    """Computes the area under the curve (AUC) confidence interval (CI) via bootstrapping

    Args:
        q: lower and upper bound percentiles. (0.05, 0.95) for 90% CI, (0.025, 0.975) for 95% CI.

    TODO: support parallelization
    """
    bootstrapped_auc_scores = []
    for i in range(n_bootstraps):
        sampled_preds = pred_probs.sample(n=len(pred_probs), replace=True, random_state=i)
        sampled_labels = labels.loc[sampled_preds.index]
        bootstrapped_auc_scores.append(auc_scores(sampled_labels, sampled_preds))
    return pd.DataFrame(bootstrapped_auc_scores).quantile(q=q)


###############################################################################
# Plots
###############################################################################
def plot_roc(ax, y_true, y_pred, label: Optional[str] = None, **kwargs):
    if label is None:
        label = f'AUROC={roc_auc_score(y_true, y_pred):.3f}'
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    sns.lineplot(y=tpr, x=fpr, label=label, ax=ax)
    ax.set(ylabel='Sensitivity', xlabel='1 - Specificity')
    _clean_plot(ax, **kwargs)
    

def plot_prc(ax, y_true, y_pred, label: Optional[str] = None, **kwargs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    if label is None:
        label = f'AUPRC={average_precision_score(y_true, y_pred):.3f}'
    sns.lineplot(y=precision, x=recall, label=label, ax=ax)
    ax.set(ylabel='Precision', xlabel='Recall')
    _clean_plot(ax, **kwargs)


def _clean_plot(ax, legend_loc: str = 'best', remove_legend_line: bool = False):
    leg = ax.legend(loc=legend_loc, frameon=False)
    if remove_legend_line: 
        leg.legend_handles[0].set_linewidth(0)
