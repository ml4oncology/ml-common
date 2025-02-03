"""
Module to create summary tables
"""
import pandas as pd

###############################################################################
# Labels
###############################################################################
def get_label_distribution(Y: pd.DataFrame, metainfo: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Provide a summary table of label distribution across each data split

    Args:
        Y: A table of labels. Rows correspond to samples, columns correspond to different targets.
        metainfo: A table containing "split" and "mrn" (patiend id) corresponding to each sample

    TODO: Support class label and continuous label distribution
    """
    dists = {}
    for split, group in Y.groupby(metainfo["split"]):
        dists[split] = _get_binary_label_distribution(group, metainfo, **kwargs)
    dists['Total'] = _get_binary_label_distribution(Y, metainfo, **kwargs)
    return pd.DataFrame(dists).T


def _get_binary_label_distribution(df: pd.DataFrame, metainfo: pd.DataFrame, with_respect_to: str = "sessions"):
    mrn = metainfo.loc[df.index, "mrn"]
    N = {'sessions': len(df), 'patients': mrn.nunique()}[with_respect_to]
    count = {}
    for target, labels in df.items():
        if with_respect_to == 'sessions':
            N_pos = sum(labels == 1) # positive label
            N_neg = sum(labels == 0) # negative label
            N_exc = sum(labels == -1) # exclude

        elif with_respect_to == 'patients':
            N_pos = mrn[labels == 1].nunique()
            N_neg = N - N_pos
            N_exc = 0

        count[(target, 1)] = f'{N_pos} ({N_pos/N*100:.2f}%)'
        count[(target, 0)] = f'{N_neg} ({N_neg/N*100:.2f}%)'
        if N_exc > 0: 
            count[(target, -1)] = f'{N_exc} ({N_exc/N*100:.2f}%)'
    return count