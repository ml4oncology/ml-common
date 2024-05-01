"""
Module to filter features and samples
"""
from collections.abc import Sequence
from typing import Optional
import logging

import numpy as np
import pandas as pd

from .constants import DRUG_COLS
from .util import get_nmissing, get_excluded_numbers

from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

###############################################################################
# General Filters
###############################################################################
def drop_highly_missing_features(
    df: pd.DataFrame, 
    missing_thresh: float, 
    keep_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Drop features with high level of missingness
    
    Args:
        keep_cols: list of feature names to keep regardless of high missingness
    """
    nmissing = get_nmissing(df)
    mask = nmissing['Missing (%)'] > missing_thresh
    exclude_cols = nmissing.index[mask].drop(keep_cols, errors='ignore').tolist()
    msg = f'Dropping the following {len(exclude_cols)} features for missingness over {missing_thresh}%: {exclude_cols}'
    logger.info(msg)
    return df.drop(columns=exclude_cols)


def drop_samples_with_no_targets(df: pd.DataFrame, targ_cols: Sequence[str], missing_val=None) -> pd.DataFrame:
    if missing_val is None: 
        mask = df[targ_cols].isnull()
    else:
        mask = df[targ_cols] == missing_val
    mask = ~mask.all(axis=1)
    get_excluded_numbers(df, mask, context=' with no targets')
    df = df[mask]
    return df


###############################################################################
# Specialized Filters
###############################################################################
def drop_unused_drug_features(df: pd.DataFrame) -> pd.DataFrame:
    # use 0 as a placeholder for nans and inf
    assert not (df[DRUG_COLS] == 0).any().any() # ensure none of the drug feature value equals to 0
    df[DRUG_COLS] = df[DRUG_COLS].fillna(0).replace(np.inf, 0)

    # remove drugs given less than 10 times
    mask = (df[DRUG_COLS] != 0).sum() < 10
    exclude = mask.index[mask].tolist()
    logger.info(f'Removing the following features for drugs given less than 10 times: {exclude}')
    df = df.drop(columns=exclude)

    return df