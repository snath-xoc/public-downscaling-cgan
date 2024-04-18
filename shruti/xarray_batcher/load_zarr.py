## Load in all zarr, thank god for kerchunk
import glob
import numpy as np
import xarray as xr
import datetime

from kerchunk.zarr import ZarrToZarr
from kerchunk.combine import MultiZarrToZarr
import xesmf

import variables_config

(
    all_fcst_fields,
    all_fcst_levels,
    accumulated_fields,
    nonnegative_fields,
) = variables_config.get_config()


def daterange(start_date, end_date):

    """
    Generator to get date range for a given time period from start_date to end_date
    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def get_lonlat(lonlatbox, fcst_spat_res, TRUTH_PATH):

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


def regridding(lonlatbox, fcst_spat_res, TRUTH_PATH, type="conservative"):

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
    ) = get_lonlat(lonlatbox, fcst_spat_res)

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


def load_all_single_field_forecasts_year(
    field, year, FCST_PATH, lonlatbox, fcst_spat_res, regrid=True
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
    ds_path = glob.glob(
        FCST_PATH
        + f"gfs{str(year)}*_t00z_f030_f054_{field.replace(' ','-')}_{all_fcst_levels[field]}.zarr"
    )
    (
        lon_reg,
        lat_reg,
        lon_reg_b,
        lat_reg_b,
        lon_reg_TRUTH,
        lat_reg_TRUTH,
        lon_reg_TRUTH_b,
        lat_reg_TRUTH_b,
    ) = get_lonlat(lonlatbox, fcst_spat_res)

    z2z = [ZarrToZarr(ds).translate() for ds in ds_path]

    ## somehow the grib.idx files are not all identical so need to first extract similar ones into xarray then concat
    mode_length = np.array([len(z.keys()) for z in z2z]).flatten()

    ds_to_concat = []

    for mode in np.unique(mode_length):

        z2zs = [z for z in z2z if len(z.keys()) == mode]

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

        ds_to_concat.append(
            xr.open_dataset(
                "reference://", engine="zarr", backend_kwargs=backend_kwargs
            )
            .sel({"latitude": lat_reg, "longitude": lon_reg})
            .squeeze(dim=all_fcst_levels[field], drop=True)
        )

    ## Issue, this may take very long, what shall I do, either throw away the few bad files or reformat them
    ds_to_concat = xr.concat(ds_to_concat, "time")

    return ds_to_concat
