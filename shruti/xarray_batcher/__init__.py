## Initialisation for xarray batcher, import all helper functions

import os
import datetime
import numpy as np

import batch
import create_tfrecords
import load_zarr
import variables_config

import importlib

importlib.reload(batch)
importlib.reload(create_tfrecords)
importlib.reload(load_zarr)
importlib.reload(variables_config)

## get field and meta data
(
    all_fcst_fields,
    all_fcst_levels,
    accumulated_fields,
    nonnegative_fields,
) = variables_config.get_config()
TEMP_RES = variables_config.get_time_res()

## get paths
FCST_PATH, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH = variables_config.get_paths()


def daterange(start_date, end_date):

    """
    Generator to get date range for a given time period from start_date to end_date
    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def get_valid_dates(
    year,
    TEMP_RES,
    start_hour=0,
    end_hour=24,
):

    """
    Returns list of valid forecast start dates for which 'truth' data
    exists, given the other input parameters. If truth data is not available
    for certain days/hours, this will not be the full year. Dates are returned
    as a list of YYYYMMDD strings.

    Parameters:
        year (int): forecasts starting in this year
        start_hour (int): Lead time of first forecast desired
        end_hour (int): Lead time of last forecast desired
    """

    # sanity checks for our dataset
    assert year in (2018, 2019, 2020, 2021, 2022)
    assert start_hour >= 0
    assert start_hour % TEMP_RES == 0
    assert end_hour % TEMP_RES == 0
    assert end_hour > start_hour

    # Build "cache" of truth data dates/times that exist as well as forecasts
    valid_dates = []

    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(
        year + 1, 1, end_hour // TEMP_RES + 2
    )  # go a bit into following year

    for curdate in daterange(start_date, end_date):
        datestr = curdate.strftime("%Y%m%d")
        valid = True

        ##first check for forecast data
        ##hard coded for now need to change so that it can naturally infer file name pattern
        ## should be edited as according to your forecast data, e.g. maybe you want to check hourly files etc.
        test_field = all_fcst_fields[-1]
        test_file = f"gfs{datestr}_t00z_f030_f054_{test_field.replace(' ','-')}_{all_fcst_levels[test_field]}.zarr"

        if not os.path.exists(os.path.join(FCST_PATH, test_file)):
            valid = False
            continue

        ## then check for truth data at the desired lead time
        datestr_true = (curdate + datetime.timedelta(hours=TEMP_RES)).strftime("%Y%m%d")
        for hr in np.arange(start_hour, end_hour, TEMP_RES):

            fname = f"{datestr_true}_{hr:02}"

            if not os.path.exists(os.path.join(TRUTH_PATH, f"{fname}.nc")):
                valid = False
                break

        if valid:
            valid_dates.append(curdate)

    return valid_dates
