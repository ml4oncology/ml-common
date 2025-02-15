"""
Module to anchor features or targets
"""
import pandas as pd


def merge_closest_measurements(
    main: pd.DataFrame, 
    meas: pd.DataFrame, 
    main_date_col: str,
    meas_date_col: str, 
    direction: str = 'backward',
    time_window: tuple[int, int] = (-5,0),
    include_meas_date: bool = True
) -> pd.DataFrame:
    """Extract the closest measurements (lab tests, symptom scores, etc) prior to / after the main date 
    within a lookback / lookahead window and combine them to the main dataset

    Both main and meas should have mrn and date columns
    
    Args:
        direction: specifies whether to merge measurements before or after the main date. Either 'backward' or 'forward'
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

    # merge each measurement column individually
    for col in meas.columns:
        if col in ['mrn', meas_date_col]: continue

        data_to_merge = meas.loc[meas[col].notnull(), ['mrn', meas_date_col, col]]

        # merges the closest row to main date while matching on mrn
        main = pd.merge_asof(
            main, data_to_merge, left_on='main_date', right_on=meas_date_col, by='mrn', direction=direction, 
            allow_exact_matches=True, tolerance=pd.Timedelta(days=upper_limit - lower_limit)
        )
        
        if include_meas_date:
            # rename the date column
            main = main.rename(columns={meas_date_col: f'{col}_{meas_date_col}'})
        else:
            del main[meas_date_col]

    del main['main_date']
    return main


def measurement_stat_extractor(
    partition, 
    main_date_col: str,
    meas_date_col: str,
    stats: list[str] | None = None, 
    time_window: tuple[int, int] = (-5,0),
    include_meas_date: bool = True
) -> list:
    """Extract either the sum, max, min, mean, or count of measurements (lab tests, symptom scores, etc) 
    taken within the time window (centered on each main date)

    Args:
        main_date_col: The column name of the main visit date
        meas_date_col: The column name of the measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) each visit date
        stat: What aggregate functions to use for the measurements taken within the time window. Options are sum, max, min, avg, or count
    """
    if stats is None:
        stats = ['max']
    main_df, meas_df = partition
    lower_limit, upper_limit = time_window
    meas_cols = meas_df.columns.drop(['mrn', meas_date_col])

    results = []
    meas_groups = meas_df.groupby('mrn')
    for mrn, main_group in main_df.groupby('mrn'):
        meas_group = meas_groups.get_group(mrn)
        meas_dates = meas_group[meas_date_col]

        for main_idx, date in main_group[main_date_col].items():
            earliest_date = date + pd.Timedelta(days=lower_limit)
            latest_date = date + pd.Timedelta(days=upper_limit)

            mask = meas_dates.between(earliest_date, latest_date)
            if not mask.any(): 
                continue

            meas = meas_group.loc[mask, meas_cols]
            data = {}
            if 'sum' in stats:
                result = meas.sum()
                result.index += '_sum'
                data.update(result.to_dict())
            if 'max' in stats:
                idxs = meas.loc[:, ~meas.isnull().all()].idxmax()
                for col, idx in idxs.items():
                    data[f'{col}_max'] = meas.loc[idx, col]
                    if include_meas_date: 
                        data[f'{col}_max_date'] = meas_dates[idx]
            if 'min' in stats:
                idxs = meas.loc[:, ~meas.isnull().all()].idxmin()
                for col, idx in idxs.items():
                    data[f'{col}_min'] = meas.loc[idx, col]
                    if include_meas_date:
                        data[f'{col}_min_date'] = meas_dates[idx]
            if 'avg' in stats:
                result = meas.mean()
                result.index += '_avg'
                data.update(result.to_dict())
            if 'count' in stats:
                result = meas.count()
                result.index += '_count'
                data.update(result.to_dict())

            data['index'] = main_idx
            results.append(data)
    
    return results