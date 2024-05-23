## Normalisation functions, note to self you can apply universal functions from numpy as scipy that operate element wise on xarray
## not too many function comments as I feel like they are self-explanatory

import numpy as np
import utils

import importlib

importlib.reload(utils)


## Unfortunately need to have this look up table, not sure what a work around is
precip_fields = ["Convective precipitation (water)", "Total Precipitation"]

(
    _,
    _,
    accumulated_fields,
    nonnegative_fields,
) = utils.get_config()

## Normalisation to apply !!! make sure a field doesn't appear twice!!!
standard_scaling = ["Surface pressure", "2 metre temperature"]
maximum_scaling = [
    "Convective available potential energy",
    "Upward short-wave radiation flux",
    "Downward short-wave radiation flux",
    "Cloud water",
    "Precipitable water",
    "Ice water mixing ratio",
    "Cloud mixing ratio",
    "Rain mixing ratio",
]
absminimum_maximum_scaling = ["U component of wind", "V component of wind"]


## get some standard stuff from utils
fcst_norm = utils.load_fcst_norm(year=2021)
time_res, lonlatbox, fcst_spat_res = utils.get_metadata()


def logprec(data):
    return np.log10(1.0 + data)


def nonnegative(data):
    return np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues


def m_to_mm_per_hour(data, time_res):
    data *= 1000
    return data / time_res  # convert to mm/hr


def to_per_second(data, time_res):
    # for all other accumulated fields [just ssr for us]
    return data / (
        TIME_RES * 3600
    )  # convert from a 6-hr difference to a per-second rate


def centre_at_mean(data, field):
    # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
    return data - fcst_norm[field]["mean"]


def change_to_unit_std(data, field):
    return data / fcst_norm[field]["std"]


def max_scaling(data, field):
    return (data - fcst_norm[field]["min"]) / (
        fcst_norm[field]["min"] - fcst_norm[field]["max"]
    )


def absmin_max_scaling(data, field):
    return data / max(-fcst_norm[field]["min"], fcst_norm[field]["max"])


def convert_units(data, field, log_prec):
    if field in precip_fields:
        data = m_to_mm_per_hour(data, time_res)

        if log_prec:
            return logprec(data)

        else:
            return data

    elif field in accumulated_fields:
        data = to_per_second(data, time_res)

        return data

    else:
        return data


def get_norm(data, field, log_prec):
    if field in precip_fields and log_prec:
        return data

    if field in standard_scaling:
        data.loc[{"i_x": [0, 2]}] = centre_at_mean(data.sel({"i_x": [0, 2]}), field)

        return change_to_unit_std(data, field)

    if field in maximum_scaling:
        return max_scaling(data, field)

    if field in absminimum_maximum_scaling:
        return absmin_max_scaling(data, field)

    else:
        return data
