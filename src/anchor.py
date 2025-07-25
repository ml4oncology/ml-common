"""
Module to anchor features or targets
"""
from functools import partial
from typing import Callable

import pandas as pd

from .util import split_and_parallelize


def combine_meas_to_main_data(
    main: pd.DataFrame, 
    meas: pd.DataFrame, 
    main_date_col: str, 
    meas_date_col: str, 
    parallelize: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Combine separate measurement data to the main dataset

    Both main and meas should have mrn and date columns
    """
    mask = main['mrn'].isin(meas['mrn'])
    worker = partial(measurement_stat_extractor, main_date_col=main_date_col, meas_date_col=meas_date_col, **kwargs)
    result = split_and_parallelize((main[mask], meas), worker) if parallelize else worker((main[mask], meas))
    if not result:
        return main
    result = pd.DataFrame(result).set_index('index')
    df = main.join(result)
    return df


def measurement_stat_extractor(
    partition, 
    main_date_col: str,
    meas_date_col: str,
    stats: list[str] | None = None, 
    stat_func: Callable | None = None,
    time_window: tuple[int, int] = (-5,0),
    include_meas_date: bool = False,
) -> list:
    """Extract either the first, last, sum, max, min, mean, or count of measurements (lab tests, symptom scores, etc) 
    taken within the time window (centered on each main date)

    Args:
        main_date_col: The column name of the main visit date
        meas_date_col: The column name of the measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) the main visit dates
        stat: What aggregate functions to use for the measurements taken within the time window. 
            Options are first, last, sum, max, min, avg, or count
        include_meas_date: If True, stores the date of first / last measurement
    """
    if stats is None:
        stats = ['max']
    main_df, meas_df = partition
    lower_limit, upper_limit = time_window

    results = []
    meas_groups = meas_df.groupby('mrn')
    for mrn, main_group in main_df.groupby('mrn'):
        meas_group = meas_groups.get_group(mrn)

        for main_idx, date in main_group[main_date_col].items():
            earliest_date = date + pd.Timedelta(days=lower_limit)
            latest_date = date + pd.Timedelta(days=upper_limit)

            mask = meas_group[meas_date_col].between(earliest_date, latest_date)
            if not mask.any(): 
                continue

            # if user provided their own stat function
            if stat_func is not None:
                data = stat_func(meas_group[mask])
            else:
                meas = meas_group[mask].drop(columns=['mrn'])
                data = _measurement_stat_extractor(meas, meas_date_col, stats, include_meas_date)
    
            data['index'] = main_idx
            results.append(data)
    
    return results


def _measurement_stat_extractor(
    meas: pd.DataFrame, 
    meas_date_col: str,
    stats: list[str] | None = None, 
    include_meas_date: bool = True
) -> dict:
    meas_dates = meas.pop(meas_date_col)
    data = {}
    if 'first' in stats:
        result = meas.iloc[0]
        result.index += '_FIRST'
        data.update(result.to_dict())
        if include_meas_date: 
            # TODO: support extraction of first measurement dates for each measurement column
            data[f'{meas_date_col}_FIRST'] = meas_dates.iloc[0]
    if 'last' in stats:
        result = meas.ffill().iloc[-1]
        result.index += '_LAST'
        data.update(result.to_dict())
        if include_meas_date: 
            # TODO: support extraction of last measurement dates for each measurement column
            data[f'{meas_date_col}_LAST'] = meas_dates.iloc[-1]
    if 'sum' in stats:
        result = meas.sum()
        result.index += '_SUM'
        data.update(result.to_dict())
    if 'max' in stats:
        idxs = meas.loc[:, ~meas.isnull().all()].idxmax()
        for col, idx in idxs.items():
            data[f'{col}_MAX'] = meas.loc[idx, col]
            if include_meas_date: 
                data[f'{col}_MAX_date'] = meas_dates[idx]
    if 'min' in stats:
        idxs = meas.loc[:, ~meas.isnull().all()].idxmin()
        for col, idx in idxs.items():
            data[f'{col}_MIN'] = meas.loc[idx, col]
            if include_meas_date:
                data[f'{col}_MIN_date'] = meas_dates[idx]
    if 'avg' in stats:
        result = meas.mean()
        result.index += '_AVG'
        data.update(result.to_dict())
    if 'count' in stats:
        result = meas.count()
        result.index += '_COUNT'
        data.update(result.to_dict())
    return data


# An alternate approach to retrieving first or last measurement within a time window
# More performant than measurement_stat_extractor
def merge_closest_measurements(
    main: pd.DataFrame, 
    meas: pd.DataFrame, 
    main_date_col: str,
    meas_date_col: str, 
    direction: str = 'backward',
    time_window: tuple[int, int] = (-5,0),
    merge_individually: bool = True,
    include_meas_date: bool = False
) -> pd.DataFrame:
    """Extract the closest measurements (lab tests, symptom scores, etc) prior to / after the main date 
    within a lookback / lookahead window and combine them to the main dataset

    Both main and meas should have mrn and date columns
    
    Args:
        main_date_col: The column name of the main visit date
        meas_date_col: The column name of the measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) the main visit dates
        direction: specifies whether to merge measurements before or after the main date. Either 'backward' or 'forward'
        merge_individually: If True, merges each measurement column separately
        include_meas_date: If True, include the date of the closest measurement that was merged
    """
    lower_limit, upper_limit = time_window
    if direction == 'backward':
        main['main_date'] = main[main_date_col] + pd.Timedelta(days=upper_limit)
    elif direction == 'forward':
        main['main_date'] = main[main_date_col] + pd.Timedelta(days=lower_limit)

    # pd.merge_asof uses binary search, requires input to be sorted by the time
    main = main.sort_values(by='main_date')
    meas = meas.sort_values(by=meas_date_col)
    meas[meas_date_col] = meas[meas_date_col].astype(main['main_date'].dtype) # ensure date types match

    merge_kwargs = dict(
        left_on='main_date', right_on=meas_date_col, by='mrn', direction=direction, allow_exact_matches=True,
        tolerance=pd.Timedelta(days=upper_limit - lower_limit)
    )

    if merge_individually:
        # merge each measurement column individually
        for col in meas.columns:
            if col in ['mrn', meas_date_col]: continue

            data_to_merge = meas.loc[meas[col].notnull(), ['mrn', meas_date_col, col]]

            # merges the closest row to main date while matching on mrn
            main = pd.merge_asof(main, data_to_merge, **merge_kwargs)
            
            if include_meas_date:
                # rename the date column
                main = main.rename(columns={meas_date_col: f'{col}_{meas_date_col}'})
            else:
                del main[meas_date_col]

    else:
        main = pd.merge_asof(main, meas, **merge_kwargs)

    del main['main_date']
    return main