## Temporary storage of all user definable variables

## Put all forecast fields, their levels (can be None also), and specify categories of accumulated and nonnegative fields

all_fcst_fields = [
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
CONSTANTS_PATH = "/network/home/n/nath/cGAN/constants-regICPAC/"

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
