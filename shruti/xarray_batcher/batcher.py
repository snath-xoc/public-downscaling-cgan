from typing import Any, Callable, Optional, Tuple
import numpy as np
import xarray as xr
import xbatcher
import tensorflow as tf

import load_zarr as lz
import utils

import importlib

importlib.reload(lz)
importlib.reload(utils)


def batch_from_zarr_store(
    zarr_store, df_truth_and_mask, df_constants, batch_size=[2, 128, 128], full=False
):
    if not isinstance(batch_size, list):
        batch_size = [batch_size]

    zarr_len_all = np.array([len(zarr_store_var) for zarr_store_var in zarr_store])
    all_fcst_fields, _, _, _ = utils.get_config()

    for idx_zarr in range(8, zarr_len_all.max() - batch_size[0]):
        X = xr.concat(
            [
                lz.streamline_and_normalise_zarr(
                    field,
                    lz.load_da_from_zarr_store(
                        zarr_store_var[idx_zarr : idx_zarr + batch_size[0] + 1],
                        field,
                        from_idx=True,
                    ),
                )
                .rename(field)
                .to_dataset()
                for field, zarr_store_var in zip(all_fcst_fields, zarr_store)
            ],
            dim="time",
        )

        X = X.isel({"time": np.arange(batch_size[0])})

        df_truth_and_mask_batch = df_truth_and_mask.sel(
            {"time": X.time.values + np.timedelta64(1, "D") + np.timedelta64(6, "h")}
        )
        df_constants_batch = df_constants.isel({"time": np.arange(batch_size[0])})

        variables = [var for var in X.data_vars]
        constants = [constant for constant in df_constants_batch.data_vars]

        if full:
            yield (
                {
                    "lo_res_inputs": np.moveaxis(
                        np.vstack(
                            (
                                [
                                    X[variable]
                                    .fillna(0)
                                    .values.reshape(
                                        -1, len(X.lat.values), len(X.lon.values)
                                    )
                                    for variable in variables
                                ]
                            )
                        ),
                        0,
                        -1,
                    ),
                    "hi_res_inputs": np.moveaxis(
                        np.vstack(
                            (
                                [
                                    df_constants_batch[constant]
                                    .fillna(0)
                                    .values.reshape(
                                        -1, len(X.lat.values), len(X.lon.values)
                                    )
                                    for constant in constants
                                ]
                            )
                        ),
                        0,
                        -1,
                    ),
                },
                {
                    "output": np.squeeze(df_truth_and_mask_batch.precipitation.values),
                    "mask": np.squeeze(df_truth_and_mask_batch.mask.values),
                },
            )

        else:
            yield (
                xbatcher.BatchGenerator(
                    X.fillna(0),
                    {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                    input_overlap={"lat": batch_size[1] - 1, "lon": batch_size[1] - 1},
                ),
                xbatcher.BatchGenerator(
                    df_constants_batch.fillna(0),
                    {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                    input_overlap={"lat": batch_size[1] - 1, "lon": batch_size[1] - 1},
                ),
                xbatcher.BatchGenerator(
                    df_truth_and_mask_batch,
                    {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                    input_overlap={"lat": batch_size[1] - 1, "lon": batch_size[1] - 1},
                ),
            )


class CustomTFDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_generator,
        constant_generator,
        y_generator,
    ) -> None:
        """
        Keras Dataset adapter for Xbatcher

        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        transform : callable, optional
            A function/transform that takes in an array and returns a transformed version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """
        self.X_generator = X_generator
        self.constant_generator = constant_generator
        self.y_generator = y_generator

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        variables = [var for var in self.X_generator[0].data_vars]
        constants = [constant for constant in self.constant_generator[0].data_vars]

        X_batch = tf.convert_to_tensor(
            np.moveaxis(
                np.vstack(
                    ([self.X_generator[idx][variable].values for variable in variables])
                ),
                0,
                -1,
            )
        )
        constant_batch = tf.convert_to_tensor(
            np.moveaxis(
                np.stack(
                    (
                        [
                            self.constant_generator[idx][constant].values
                            for constant in constants
                        ]
                    )
                ),
                0,
                -1,
            )
        )
        y_batch = tf.convert_to_tensor(
            self.y_generator[idx].precipitation.fillna(0).values
        )
        mask_batch = tf.convert_to_tensor(self.y_generator[idx].mask.values)

        return (
            {"lo_res_inputs": X_batch, "hi_res_inputs": constant_batch},
            {"output": y_batch, "mask": mask_batch},
        )


def zarr_store_loader(df_vars, df_truth, df_constants):
    loader_time = batch_from_zarr_store(df_vars, df_truth, df_constants)

    for X_gen, constants_gen, y_gen in loader_time:
        loader = CustomTFDataset(X_gen, constants_gen, y_gen)

        for batch in loader:
            yield batch
