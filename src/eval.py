"""
Module to evaluate system performance
"""
from typing import Optional

import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    roc_curve, 
)
from tqdm import tqdm


def get_perf_at_operating_point(df, targ_col: str, pred_col: str, threshold: float):
    df['alarm'] = df[pred_col] >= threshold
    res = {'Threshold': threshold}
    res['Warning rate'] = df['alarm'].mean()
    res['Precision'] = precision_score(df[targ_col], df['alarm'])
    res['Recall'] = recall_score(df[targ_col], df['alarm'])
    res['First-alarm precision'] = first_alarm_precision(df, targ_col)
    res['Outcome-level recall'] = outcome_level_sensitivity(df)
    del df['alarm']
    return res


def outcome_level_sensitivity(df: pd.DataFrame, lookahead_window: int = 90):
    """Get the proportion of true outcomes where at least one alarm preceded the event

    E.g. if ED visit happens on Jan 20, our lookahead window is 30 days, and assessments 
        happens on Jan 1 and Jan 14, then the outcome-level true positive is if 
        either Jan 1 or Jan 14 trigger a warning, and false negative if neither do
    """
    result = []
    for (mrn, event_date), group in df.groupby(['mrn', 'event_date']):

        # ensure assessment date and event date is within X days of each other
        diff = (group['event_date'] - group['assessment_date']).dt.days
        assert all(diff.between(0, lookahead_window))

        result.append(any(group['alarm']))

    return sum(result) / len(result) # tp / (tp + fn)


def first_alarm_precision(df: pd.DataFrame, target_col: str):
    """Get the proportion of first alarms that were true
    """
    first_alarms = df.query('alarm').groupby('mrn').first()
    return precision_score(df[target_col], first_alarms, zero_division=0)


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
