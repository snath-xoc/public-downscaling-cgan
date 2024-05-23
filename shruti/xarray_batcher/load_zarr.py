## Load in all zarr, thank god for kerchunk
import glob
import os

import numpy as np
import xarray as xr
import datetime

from kerchunk.zarr import ZarrToZarr
from kerchunk.combine import MultiZarrToZarr
import xesmf

import utils
import normalise as nm

import importlib

importlib.reload(utils)
importlib.reload(nm)

(
    all_fcst_fields,
    all_fcst_levels,
    accumulated_fields,
    nonnegative_fields,
) = utils.get_config()

FCST_PATH, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH = utils.get_paths()

time_res, lonlatbox, fcst_spat_res = utils.get_metadata()


def daterange(start_date, end_date):
    """
    Generator to get date range for a given time period from start_date to end_date
    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def get_lonlat():
    """
    Function to get longitudes and latitudes of forecast and truth data

    Input
    ------

    lonlatbox: list of int with length 4
               bottom, left, top, right corners of lon-lat box
    fcst_spat_res: float
               spatial resolution of forecasts

    Output
    ------

    centres of forecast (lon_reg, lat_reg), and truth (lon__reg_TRUTH, lat_reg_TRUTH), and their box
    edges: (lon_reg_b, lat_reg_b) and (lon__reg_TRUTH, lat_reg_TRUTH) for forecasts and truth resp.

    """
    assert len(lonlatbox) == 4

    lat_reg_b = np.arange(lonlatbox[0], lonlatbox[2], fcst_spat_res) - fcst_spat_res / 2
    lat_reg = 0.5 * (lat_reg_b[1:] + lat_reg_b[:-1])

    lon_reg_b = np.arange(lonlatbox[1], lonlatbox[3], fcst_spat_res) - fcst_spat_res / 2
    lon_reg = 0.5 * (lon_reg_b[1:] + lon_reg_b[:-1])

    data_path = glob.glob(TRUTH_PATH + "*.nc")

    ds = xr.open_mfdataset(data_path[0])
    # print(ds)
    ##infer spatial resolution of truth, we assume a standard lon lat grid!

    lat_reg_TRUTH = ds.latitude.values
    lon_reg_TRUTH = ds.longitude.values

    TRUTH_RES = np.abs(lat_reg_TRUTH[1] - lat_reg_TRUTH[0])

    lat_reg_TRUTH_b = np.append(
        (lat_reg_TRUTH - TRUTH_RES / 2), lat_reg_TRUTH[-1] + TRUTH_RES / 2
    )
    lon_reg_TRUTH_b = np.append(
        (lon_reg_TRUTH - TRUTH_RES / 2), lon_reg_TRUTH[-1] + TRUTH_RES / 2
    )

    return (
        lon_reg,
        lat_reg,
        lon_reg_b,
        lat_reg_b,
        lon_reg_TRUTH,
        lat_reg_TRUTH,
        lon_reg_TRUTH_b,
        lat_reg_TRUTH_b,
    )


def regridding(type="conservative"):
    """
    Perform regridding using xesmf

    Input
    -----

    lonlatbox: list of int with length 4
               bottom, left, top, right corners of lon-lat box
    fcst_spat_res: float
               spatial resolution of forecasts
    TRUTH_PATH: str
               path to truth data so we can infer grid type
    type: str
              type of regridding to be done, default is conservative

    Output
    ------

    xesmf regridder object to go from forecast to truth grids
    """

    (
        lon_reg,
        lat_reg,
        lon_reg_b,
        lat_reg_b,
        lon_reg_TRUTH,
        lat_reg_TRUTH,
        lon_reg_TRUTH_b,
        lat_reg_TRUTH_b,
    ) = get_lonlat()

    grid_in = {"lon": lon_reg, "lat": lat_reg, "lon_b": lon_reg_b, "lat_b": lat_reg_b}

    # output grid has a larger coverage and finer resolution
    grid_out = {
        "lon": lon_reg_TRUTH,
        "lat": lat_reg_TRUTH,
        "lon_b": lon_reg_TRUTH_b,
        "lat_b": lat_reg_TRUTH_b,
    }

    regridder = xesmf.Regridder(grid_in, grid_out, type)

    return regridder


def load_zarr_store(ds_path):
    z2z = [ZarrToZarr(ds).translate() for ds in ds_path]

    ## somehow the grib.idx files are not all identical so need to first extract similar ones into xarray then concat
    mode_length = np.array([len(z.keys()) for z in z2z]).flatten()
    modals, counts = np.unique(mode_length, return_counts=True)
    index = np.argmax(counts)

    return [z for z in z2z if len(z.keys()) == modals[index]]


def load_da_from_zarr_store(z2zs, field, from_idx=False):
    (
        lon_reg,
        lat_reg,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_lonlat()

    if len(z2zs) == 0:
        print(field)

        return

    if from_idx:
        z2zs = load_zarr_store(z2zs)
    try:
        mzz = MultiZarrToZarr(
            z2zs,
            concat_dims=["time"],
            identical_dims=["step", "latitude", "longitude", all_fcst_levels[field]],
        )

        ref = mzz.translate()

        backend_kwargs = {
            "consolidated": False,
            "storage_options": {
                "fo": ref,
            },
        }

        ds = xr.open_dataset(
            "reference://", engine="zarr", backend_kwargs=backend_kwargs
        ).sel({"latitude": lat_reg, "longitude": lon_reg})
    except:
        print("Not sure what happened with", field, z2zs)

        return

    if all_fcst_levels[field] == "isobaricInhPa":
        ds = ds.sel({all_fcst_levels[field]: 200})

    short_names = list(ds.variables.keys())

    short_name = [
        short_name
        for short_name in short_names
        if short_name
        not in [
            "latitude",
            "longitude",
            "step",
            all_fcst_levels[field],
            "surface",
            "time",
            "valid_time",
        ]
    ][0]

    return ds[short_name].drop(all_fcst_levels[field])


def load_all_single_field_forecasts(
    field, yearsordates, regrid=True, return_store=False
):
    """

    Load all the forecast data for a given field at a given year that are available within the forecast directory

    Inputs
    ------

    field: str
           Name of field to get forecast output for

    year: int
          Integer of year for which to extract forecast for

    FCST_PATH: str
          Directory containing forecasts (assumes all forecasts are in there and no subdirectories,
          to be potentially modified to year subdirectories

    Outputs
    -------

    xr.Dataset (time, lat, lon,)


    """

    ds_path = []

    for yearordate in yearsordates:
        ds_path += glob.glob(
            FCST_PATH
            + f"gfs{str(yearordate).replace('-','')}*_t00z_f030_f054_{field.replace(' ','-')}_{all_fcst_levels[field]}.zarr"
        )

    if return_store:
        return ds_path

    else:
        z2zs = load_zarr_store(ds_path)

        return load_da_from_zarr_store(z2zs, field)


import warnings


def streamline_and_normalise_zarr(field, da, regrid=True, norm=True, log_prec=True):
    """
    Streamlines zarr file by calculating daily (or ensemble) mean and std

    Input
    -----

    df: xr.Dataset (time, step, lon, lat,)

    regrid: Boolean

    norm: Boolean

    log_prec: Boolean

    Outputs
    -------

    xr.Dataset with mean calculated over time steps, and depending on norm and regrid specified,
    normalised and regridded
    """

    n_steps = len(da.step.values)
    steps_1 = da.step.values[: n_steps // 2]
    steps_2 = da.step.values[n_steps // 2 :]

    da_to_concat = [
        da.sel({"step": steps_1})
        .mean("step", skipna=True)
        .expand_dims(dim={"i_x": [0]}, axis=0),
        da.sel({"step": steps_2})
        .mean("step", skipna=True)
        .expand_dims(dim={"i_x": [2]}, axis=0),
    ]

    da_to_concat.append(
        da.sel({"step": steps_1})
        .std("step", skipna=True)
        .expand_dims(dim={"i_x": [1]}, axis=0)
    )
    da_to_concat.append(
        da.sel({"step": steps_2})
        .std("step", skipna=True)
        .expand_dims(dim={"i_x": [3]}, axis=0)
    )

    da = xr.concat(da_to_concat, dim="i_x")

    if field in nonnegative_fields:
        da = nm.nonnegative(da)

    da = nm.convert_units(da, field, log_prec)

    if norm:
        da = nm.get_norm(da, field, log_prec)
    warnings.filterwarnings("ignore", category=UserWarning)
    if regrid:
        regridder = regridding()
        da = regridder(da)

    return da.where(np.isfinite(da), 0)


def load_truth_and_mask(dates, time_idx=1, log_precip=True):
    """
    Returns a single (truth, mask) item of data.
    Parameters:
        date: forecast start date
        time_idx: forecast 'valid time' array index
        log_precip: whether to apply log10(1+x) transformation
    """
    ds_to_concat = []

    for date in dates:
        date = str(date).split("T")[0].replace("-", "")

        # convert date and time_idx to get the correct truth file
        fcst_date = datetime.datetime.strptime(date, "%Y%m%d")
        valid_dt = fcst_date + datetime.timedelta(
            hours=int(time_idx) * time_res
        )  # needs to change for 12Z forecasts

        fname = valid_dt.strftime("%Y%m%d")

        data_path = glob.glob(TRUTH_PATH + f"{fname}_*.nc")

        # ds = xr.concat([xr.open_dataset(dataset).expand_dims(dim={'time':i}, axis=0)
        # for i,dataset in enumerate(data_path)],dim='time').mean('time')
        ds = xr.open_dataset(data_path[0])

        if log_precip:
            ds["precipitation"] = nm.logprec(ds["precipitation"])

        # mask: False for valid truth data, True for invalid truth data
        # (compatible with the NumPy masked array functionality)
        # if all data is valid:
        mask = ~np.isfinite(ds["precipitation"])
        ds["mask"] = mask

        ds_to_concat.append(ds)

    return xr.concat(ds_to_concat, dim="time")


def load_hires_constants(batch_size=1):
    """

    Get elevation and land sea mask on IMERG resolution

    """

    oro_path = CONSTANTS_PATH + "elev.nc"

    lsm_path = CONSTANTS_PATH + "lsm.nc"

    ds = xr.open_mfdataset([lsm_path, oro_path])

    # LSM is already 0:1
    ds["elevation"] = ds["elevation"] / 10000.0

    return ds.expand_dims(dim={"time": batch_size})
