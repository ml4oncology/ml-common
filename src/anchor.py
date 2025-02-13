"""
Module to anchor features or targets
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

def combine_feat_to_main_data(
    main: pd.DataFrame, 
    feat: pd.DataFrame, 
    main_date_col: str,
    feat_date_col: str, 
    time_window: tuple[int, int] = (-5,0),
    include_feat_date: bool = True
) -> pd.DataFrame:
    """Extract the closest features prior to the main date within a lookback window and combine them to the main dataset

    Both main and feat should have mrn and date columns
    """
    lower_limit, upper_limit = time_window

    # pd.merge_asof uses binary search, requires input to be sorted by the time
    main = main.sort_values(by=main_date_col)
    feat = feat.sort_values(by=feat_date_col)
    feat[feat_date_col] = feat[feat_date_col].astype(main[main_date_col].dtype) # ensure date types match

    # merge each feature individually
    for col in feat.columns:
        if col in ['mrn', feat_date_col]: continue

        data_to_merge = feat.loc[feat[col].notnull(), ['mrn', feat_date_col, col]]

        # merges the closest row prior to main date while matching on mrn
        main = pd.merge_asof(
            main, data_to_merge, left_on=main_date_col, right_on=feat_date_col, by='mrn', direction='backward', 
            allow_exact_matches=True, tolerance=pd.Timedelta(days=abs(lower_limit))
        )

        # if measured outside the time window, set to NaN
        mask = main[feat_date_col] > main[main_date_col] + pd.Timedelta(days=upper_limit)
        main.loc[mask, [feat_date_col, col]] = None

        if include_feat_date:
            # rename the date column
            main = main.rename(columns={feat_date_col: f'{col}_{feat_date_col}'})
        else:
            del main[feat_date_col]

    return main


def extractor(
    partition, 
    main_date_col: str,
    feat_date_col: str,
    keep: str = 'last', 
    time_window: tuple[int, int] = (-5,0),
) -> list:
    """Extract either the sum, max, first, or last forward filled measurements (lab tests, symptom scores, etc) 
    taken within the time window (centered on each main visit date)

    Args:
        main_date_col: The column name of the main visit date
        feat_date_col: The column name of the feature measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) each visit date
        keep: Which measurements taken within the time window to keep, either `sum`, `first`, `last`

    TODO: support extraction of number of measurements in the time window
    TODO: support extraction of number of days from main date in which the first/last measurement was collected
    """
    if keep not in ['first', 'last', 'max', 'sum']:
        raise ValueError('keep must be either first, last, max, or sum')
    
    main_df, feat_df = partition
    lower_limit, upper_limit = time_window
    keep_cols = feat_df.columns.drop(['mrn', feat_date_col])

    results = []
    for mrn, main_group in main_df.groupby('mrn'):
        feat_group = feat_df.query('mrn == @mrn')

        for idx, date in main_group[main_date_col].items():
            earliest_date = date + pd.Timedelta(days=lower_limit)
            latest_date = date + pd.Timedelta(days=upper_limit)

            mask = feat_group[feat_date_col].between(earliest_date, latest_date)
            if not mask.any(): 
                continue

            feats = feat_group.loc[mask, keep_cols]
            if keep == 'sum':
                result = feats.sum()
            elif keep == 'max':
                result = feats.max()
            elif keep == 'first':
                result = feats.iloc[0]
            elif keep == 'last':
                result = feats.ffill().iloc[-1]

            results.append([idx]+result.tolist())
    
    return results