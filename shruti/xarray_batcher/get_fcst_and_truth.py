import utils
import time
import numpy as np
import xarray as xr

import load_zarr as lz
import importlib

importlib.reload(lz)
importlib.reload(utils)


def get_var(variable, dates, years, return_store=False):
    """
    Input
    -----

    var: str
         Variable name to extract

    dates: ndarray, dtype=datetime64[ns]
           valid dates to keep

    year: list
          years for which to extract variables for

    generator: Boolean, default=False


    Output
    ------

    xr.DataArray (time,lon,lat)
    """

    if return_store:
        return lz.load_all_single_field_forecasts(
            variable, years, return_store=return_store
        )

    else:
        da = lz.load_all_single_field_forecasts(
            variable, years, return_store=return_store
        )
        dates = [date for date in dates if date in da.time.values]
        da = da.sel({"time": dates})

        da = lz.streamline_and_normalise_zarr(variable, da)

        return da, dates


def get_all(years, generator=False):
    """
    Input
    -----
    year: list
          years for which to extract variables for

    Output
    ------

    xr.Dataset of variables (variable time,lon,lat)
    xr.Dataset of truth values alongside a valid mask
    """

    if not isinstance(years, list):
        years = [years]

    (
        all_fcst_fields,
        all_fcst_levels,
        accumulated_fields,
        nonnegative_fields,
    ) = utils.get_config()

    dates_all = []

    print("Getting all variables for years: ", years)
    start_time = time.time()

    ds_vars = []

    for year in years:
        dates = utils.get_valid_dates(year)
        dates = [date.strftime("%Y-%m-%d") for date in dates]
        dates_all += dates

    dates_all = np.array(dates_all, dtype="datetime64[ns]")

    if generator:
        zarr_store = []

        for var in all_fcst_fields:
            zarr_store.append(get_var(var, dates_all, years, return_store=True))

        print(
            "Extracted all %i variables in ----" % len(all_fcst_fields),
            time.time() - start_time,
            "s---- as zarr store list now working on truth",
        )
        start_time = time.time()

        ds_truth_and_mask = lz.load_truth_and_mask(np.unique(dates_all))
        ds_constants = lz.load_hires_constants(batch_size=len(dates_all))
        ds_constants["time"] = dates_all

        print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s----",
        )

        return (
            zarr_store,
            ds_truth_and_mask.rename({"latitude": "lat", "longitude": "lon"}),
            ds_constants,
        )

    else:
        dates_final = []
        for var in all_fcst_fields:
            da, dates_modified = get_var(var, dates_all, years)
            ds_vars.append(da.rename(var).to_dataset())

            dates_final += dates_modified

        print(
            "Extracted all %i variables in ----" % len(all_fcst_fields),
            time.time() - start_time,
            "s---- now consolidating into single Dataset",
        )
        start_time = time.time()

        ds_vars = xr.concat(ds_vars, dim="time")

        print(
            "Finished consolidation in ----",
            time.time() - start_time,
            "s---- now working on truth",
        )
        start_time = time.time()

        ds_truth_and_mask = lz.load_truth_and_mask(np.unique(dates_final))
        ds_constants = lz.load_hires_constants(batch_size=len(np.unique(dates_final)))
        ds_constants["time"] = np.unique(dates_final)

        print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s----",
        )

        return (
            ds_vars,
            ds_truth_and_mask.rename({"latitude": "lat", "longitude": "lon"}),
            ds_constants,
        )
