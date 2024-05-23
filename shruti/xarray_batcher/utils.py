## Utils needed in loading zarr and batching
import os
import datetime
import numpy as np
import pickle

## Put all forecast fields, their levels (can be None also), and specify categories of accumulated and nonnegative fields

all_fcst_fields = [
    "Convective available potential energy",
    "Convective precipitation (water)",  #    "Medium cloud cover",
    "Surface pressure",
    "Upward short-wave radiation flux",
    "Downward short-wave radiation flux",
    "2 metre temperature",
    "Cloud water",
    "Precipitable water",
    "Ice water mixing ratio",
    "Cloud mixing ratio",
    "Rain mixing ratio",
    "Total Precipitation",
    "U component of wind",
    "V component of wind",
]

all_fcst_levels = {
    "Convective available potential energy": "surface",
    "Convective precipitation (water)": "surface",
    "Medium cloud cover": "middleCloudLayer",
    "Surface pressure": "surface",
    "Upward short-wave radiation flux": "surface",
    "Downward short-wave radiation flux": "surface",
    "2 metre temperature": "heightAboveGround",
    "Cloud water": "atmosphereSingleLayer",
    "Precipitable water": "atmosphereSingleLayer",
    "Ice water mixing ratio": "isobaricInhPa",
    "Cloud mixing ratio": "isobaricInhPa",
    "Rain mixing ratio": "isobaricInhPa",
    "Total Precipitation": "surface",
    "U component of wind": "isobaricInhPa",
    "V component of wind": "isobaricInhPa",
}


accumulated_fields = ["Convective precipitation (water)", "ssr", "Total Precipitation"]
nonnegative_fields = [
    "Convective available potential energy",
    "Convective precipitation (water)",
    "Medium cloud cover",
    "Surface pressure",
    "Upward short-wave radiation flux",
    "Downward short-wave radiation flux",
    "2 metre temperature",
    "Cloud water",
    "Precipitable water",
    "Ice water mixing ratio",
    "Cloud mixing ratio",
    "Rain mixing ratio",
    "Total Precipitation",
]


## Put other user-specification i.e., lon-lat box, spatial and temporal resolution (in hours)
TIME_RES = 24
LONLATBOX = [-14, 19, 25.25, 54.75]
FCST_SPAT_RES = 0.25

## Put all directories here

TRUTH_PATH = (
    "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/ICPAC_region/24h/"
)
FCST_PATH = "/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/netcdf/"
CONSTANTS_PATH = (
    "/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/cGAN/constants-regICPAC/"
)

TFRECORDS_PATH = "/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/cGAN/tfrecords/"


def get_config():
    return all_fcst_fields, all_fcst_levels, accumulated_fields, nonnegative_fields


def get_metadata():
    """
    Returns time resolution (in hours), lonlat box (bottom, left, top, right) and the forecast's spatial resolution
    """

    return TIME_RES, LONLATBOX, FCST_SPAT_RES


def get_paths():
    return FCST_PATH, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH


import pickle


def load_fcst_norm(year=2021):
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")

    with open(fcstnorm_path, "rb") as f:
        return pickle.load(f)


def daterange(start_date, end_date):
    """
    Generator to get date range for a given time period from start_date to end_date
    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def get_valid_dates(
    year,
    TIME_RES=TIME_RES,
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
    assert start_hour % TIME_RES == 0
    assert end_hour % TIME_RES == 0
    assert end_hour > start_hour

    # Build "cache" of truth data dates/times that exist as well as forecasts
    valid_dates = []

    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(
        year + 1, 1, end_hour // TIME_RES + 2
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
        datestr_true = (curdate + datetime.timedelta(hours=TIME_RES)).strftime("%Y%m%d")
        for hr in np.arange(start_hour, end_hour, TIME_RES):
            fname = f"{datestr_true}_06"  # {hr:02}

            if not os.path.exists(os.path.join(TRUTH_PATH, f"{fname}.nc")):
                valid = False
                break

        if valid:
            valid_dates.append(curdate)

    return valid_dates
