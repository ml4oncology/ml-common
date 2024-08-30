"""
Module to preprocess text
"""
from collections import OrderedDict

import pandas as pd

###############################################################################
# Miscellaneous Functions
###############################################################################
def combine_text(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    text_col: str = 'text',
    add_prev_text: bool = False,
    no_same_lines: bool = True,
) -> pd.DataFrame:
    """Combine notes/reports that occured on the same day

    Also supports combining all previous text with the current text

    Args:
        add_prev_text: If True, combines all past text to the current text
        no_same_lines: If True, removes duplicate lines
    """
    if add_prev_text:
        raise NotImplementedError('Combining all past text not supported yet')

    # combine same day texts
    df = df.groupby(['mrn', date_col])[text_col].apply('\n'.join).reset_index()

    if no_same_lines:
        # remove duplicate lines (some may be same text with addendums)
        df[text_col] = remove_duplicate_lines(df[text_col])

    return df


def remove_duplicate_lines(text: pd.Series) -> pd.Series:
    lines = text.str.split('\n')
    def ordered_set(arr):
        return list(OrderedDict({x: None for x in arr}).keys()) 
    return lines.apply(lambda text: "\n".join(ordered_set(text)))


def get_text_size(x):
    return pd.concat([
        x.str.len().describe(),
        x.str.split().str.len().describe(),
    ], keys=['Length of text', 'Number of words'], axis=1)