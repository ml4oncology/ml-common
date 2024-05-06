"""
Module to engineer features
"""
from collections.abc import Sequence
from typing import Optional
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd

from .constants import LAB_COLS, LAB_CHANGE_COLS, SYMP_COLS, SYMP_CHANGE_COLS

logger = logging.getLogger(__name__)


def get_change_since_prev_session(df: pd.DataFrame) -> pd.DataFrame:
    """Get change since last session"""
    cols = LAB_COLS + SYMP_COLS + ['patient_ecog']
    change_cols = SYMP_CHANGE_COLS + LAB_CHANGE_COLS + ['patient_ecog_change']
    result = []
    for mrn, group in tqdm(df.groupby('mrn'), desc='Getting change since last session...'):
        change = group[cols] - group[cols].shift()
        result.append(change.reset_index().to_numpy())
    result = np.concatenate(result)

    result = pd.DataFrame(result, columns=['index']+change_cols).set_index('index')
    result.index = result.index.astype(int)
    df = pd.concat([df, result], axis=1)

    return df


def get_missingness_features(
    df: pd.DataFrame, exclude_keyword: str = "target"
) -> pd.DataFrame:
    cols_with_nan = df.columns[df.isnull().any()]
    cols_with_nan = cols_with_nan[~cols_with_nan.str.contains(exclude_keyword)]
    df[cols_with_nan + "_is_missing"] = df[cols_with_nan].isnull()
    return df


def collapse_rare_categories(
    df: pd.DataFrame, 
    catcols: Sequence[str], 
    N: Optional[int] = 6
) -> pd.DataFrame:
    """Collapse rare categories to 'other'
    
    Args:
        N: the minimum number of patients a category must be assigned to
    """
    for feature in catcols:
        other_mask = False
        drop_cols = []
        for col in df.columns[df.columns.str.startswith(feature)]:
            mask = df[col]
            if df.loc[mask, 'mrn'].nunique() < N:
                drop_cols.append(col)
                other_mask |= mask
        df = df.drop(columns=drop_cols)
        df[f'{feature}_other'] = other_mask
        msg = (f'Reassigning the following {len(drop_cols)} indicators with less than {N} patients as other: {drop_cols}')
        logger.info(msg)
    return df