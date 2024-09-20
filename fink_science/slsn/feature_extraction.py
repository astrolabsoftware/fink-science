# Copyright 2022-2024 Fink Software
# Author: Etienne Russeil
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from line_profiler import profile
import warnings
import pandas as pd
import numpy as np
import fink_science.slsn.kernel as k
import fink_science.slsn.basic_functions as base
from light_curve.light_curve_py import RainbowFit
from light_curve.light_curve_py import warnings as rainbow_warnings
from pandas.testing import assert_frame_equal  # noqa: F401


@profile
def transform_data(data):
    """Apply transformations for each filters on a flux dataset
            - Shift cmidPointTai so that the max flux point is at 0
            - Normalize by dividing flux and flux err by the
            maximum flux of the k.NORMALIZING_BAND (kernel option)
            - Add a column with maxflux before normalization

    Split the results into multiple dataframes each containing only one passband.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of alerts.

    Returns
    -------
    all_transformed : list
        List of DataFrame. Each df is a transformed version of data
        that only contains observations from one passband and valid objects.
    valid: np.array
        Boolean array describing if each object is valid.
        Objects are valid if all required band have at least k.MINIMUM_POINTS observations.

    Examples
    --------
    """

    peak = data["cpsFlux"].apply(base.get_max)
    valid = np.array([True] * len(data))
    valid = valid & (data["cmidPointTai"].apply(lambda x: len(x) >= k.MINIMUM_POINTS))

    if not valid.any():
        return data, valid

    data["peak"] = peak
    data["cmidPointTai"] = data.apply(base.translate, axis=1)
    data[["cpsFlux", "cpsFluxErr"]] = data.apply(base.normalize, axis=1)
    data["snr"] = data[["cpsFlux", "cpsFluxErr"]].apply(
        lambda pdf: pdf["cpsFlux"] / pdf["cpsFluxErr"], axis=1
    )

    return data, valid


@profile
def parametrise(transformed, metadata, target_col=""):
    """Extract parameters .

    Parameters are :
            - "peak" : maximum flux before normalization for filter k.NORMALIZING_BAND
            - "ra" : right ascension
            - "decl" : declination

            Optional metadata:
                - "hostgal_snsep" : distance to host galaxy
                - "hostgal_zphot" : redshift of the host galaxy
                - "hostgal_zphot_err" : error on the redshift of the host galaxy

            For each filter:
                - 'std' : standard deviation of the flux for each filter
                - 'mean_snr' : mean signal over noise ratio for each filter
                - 'nb_points' : number of points for each filter

    Parameters
    ----------
    all_transformed : list
        List of transformed DataFrame using "transform_data" function.
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset.
        Default is ''

    Returns
    -------
    df_parameters : pd.DataFrame
        DataFrame of parameters.
        Also adds columns of cmidPointTai, cpsFlux and cpsFluxErr that
        will be used to compute color later on.

    Example
    -------
    """

    ids = transformed["diaObjectId"]

    df_parameters = pd.DataFrame(data={"object_id": ids})
    df_parameters["peak"] = transformed["peak"]

    warnings.filterwarnings("ignore", category=rainbow_warnings.ExperimentalWarning)
    rainbow_features = transformed.apply(apply_rainbow, axis=1)

    for idx, name in enumerate(
        [
            "reference_time",
            "rise_time",
            "amplitude",
            "Tmin",
            "Tmax",
            "t_color",
            "fit_error",
        ]
    ):
        df_parameters[name] = [i[idx] for i in rainbow_features]

    for idx, band in enumerate(k.PASSBANDS):
        masks = transformed["cfilterName"].apply(lambda x: x == band)

        single_band_flux = pd.Series(
            [k[masks.iloc[idx2]] for idx2, k in enumerate(transformed["cpsFlux"])]
        )
        std = single_band_flux.apply(base.compute_std)

        single_band_snr = pd.Series(
            [k[masks.iloc[idx2]] for idx2, k in enumerate(transformed["snr"])]
        )
        mean_snr = single_band_snr.apply(base.compute_mean)

        df_parameters[f"std_{band}"] = list(std)
        df_parameters[f"mean_snr_{band}"] = list(mean_snr)

    df_parameters["ra"] = transformed["ra"]
    df_parameters["decl"] = transformed["decl"]
    df_parameters["nb_points"] = transformed["cpsFlux"].apply(lambda x: len(x))

    if metadata:
        df_parameters["hostgal_snsep"] = transformed["hostgal_snsep"]
        df_parameters["hostgal_zphot"] = transformed["hostgal_zphot"]
        df_parameters["hostgal_zphot_err"] = transformed["hostgal_zphot_err"]

    if target_col != "":
        targets = transformed[target_col]
        df_parameters[target_col] = targets

    return df_parameters


def apply_rainbow(pds):
    band_wave_aa = {"u": 3751, "g": 4742, "r": 6173, "i": 7502, "z": 8679, "Y": 9711}

    fitter = RainbowFit.from_angstrom(
        band_wave_aa, with_baseline=False, temperature="sigmoid", bolometric="linexp"
    )  # ,

    try:
        result = fitter._eval(
            t=pds["cmidPointTai"],
            m=pds["cpsFlux"],
            sigma=pds["cpsFluxErr"],
            band=pds["cfilterName"],
        )
        return result

    except RuntimeError:
        return np.array([-9] * 7)


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
