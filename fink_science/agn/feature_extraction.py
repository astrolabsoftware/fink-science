# Copyright 2022 Fink Software
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

import pandas as pd
import fink_science.agn.models as mod
from pandas.testing import assert_frame_equal  # noqa: F401
import fink_science.agn.kernel as k
import numpy as np
import pickle  # noqa: F401
from scipy.optimize import curve_fit
import warnings
import fink_science.agn.unit_examples as uex  # noqa: F401


def map_fid(ps):
    """Convert LSST filters to corresponding int value
    From u, g, r, i, z, Y to 0, 1, 2, 3, 4, 5

    Parameters
    ----------
    ps: pd.Series
        Must contain columns 'cfid'

    Returns
    -------
    pd.Series
        Serie with 'cfid' converted to ints

    Examples
    --------
    >>> np.array_equal(map_fid(['u', 'g', 'r', 'i', 'z', 'Y']), np.array([0, 1, 2, 3, 4, 5]))
    True
    """

    band_dict = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5}
    return np.array(list(map(band_dict.get, ps)))


def remove_nan(ps):
    """
    funtion that remove nan values from list contains in columns

    Paramters
    ---------
    ps : pd.Series
        each rows of the dataframe. each entries must be numeric list

    Return
    ------
    list_without_nan : list
        list of the same size as x, each entries is the original list from the
        current rows without the nan values and the associated values from the other columns.

    Examples
    --------
    >>> serie_example = pd.Series(data = {'cmagpsf':[0,1,np.nan],'cjd':[3, 4, np.nan],'cra':[95, -4, np.nan],'cdec':[58, 43, np.nan],'anything':[1,5,8]})
    >>> remove_nan(serie_example)
    [array([0, 1]), array([3, 4]), array([95, -4]), array([58, 43]), array([1, 5])]
    """

    mask = np.equal(ps["cmagpsf"], ps["cmagpsf"])

    return [np.array(_col)[mask].astype(type(_col[0])) for _col in ps]


def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
    """Conversion from magnitude to Fluxcal from SNANA manual
    Parameters
    ----------
    magpsf: float
        PSF-fit magnitude from ZTF.
    sigmapsf: float
        Error on PSF-fit magnitude from ZTF.

    Returns
    ----------
    fluxcal: float
        Flux cal as used by SNANA
    fluxcal_err: float
        Absolute error on fluxcal (the derivative has a minus sign)

    Examples
    --------
    >>> flux = mag2fluxcal_snana(18.9, 0.07)
    >>> round(flux[0], 5), round(flux[1], 5)
    (2754.2287, 177.5718)
    >>> mag2fluxcal_snana(None, None)
    (None, None)
    """

    if magpsf is None:
        return None, None

    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10**10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err


def compute_hostgal_dist(df):
    """Compute the distance to host galaxy column
        using simple Pythagoras computation.

    Parameters
    ----------
    df: pd.DataFrame
       ELASTiCC alert data.
       Must contain "hostgal_ra","hostgal_dec", "ra", "dec" columns.

    Returns
    -------
    np.array
        Distance from object to host galaxy.
        Returns -9 if galaxy position is unspecified

    Examples
    --------
    >>> df1 = pd.Series(data = {"hostgal_ra":90,"hostgal_dec":46, "ra":89, "dec":46.5})
    >>> round(compute_hostgal_dist(df1))
    1118
    >>> df2 = pd.Series(data = {"hostgal_ra":90,"hostgal_dec":-999, "ra":89, "dec":46.5})
    >>> compute_hostgal_dist(df2)
    -9
    """

    if (df["hostgal_ra"] == -999) | (df["hostgal_dec"] == -999):
        hostgal_dist = -9
    else:
        hostgal_dist = (
            np.sqrt((df["ra"] - df["hostgal_ra"]) ** 2 + (df["dec"] - df["hostgal_dec"]) ** 2) * 1e3
        )

    return hostgal_dist


def convert_full_dataset(clean: pd.DataFrame):
    """
    Convert all mag and mag err to flux and flux err

    Paramters
    ---------
    clean : pd.DataFrame
        Dataframe of alerts with nan removed

    Return
    ------
    pd.DataFrame
         DataFrame with column cmagpsf replaced with cflux
         and column csigmagpsf replaced with csigflux

    Example
    -------
    >>> example = pd.DataFrame(data = {'cmagpsf':[20, 10.5],'csigmapsf':[1, 1.5], 'anything':['toto', 86]})
    >>> converted = convert_full_dataset(example)
    >>> converted.round(1).equals(pd.DataFrame(data = {'cflux':[1000.0, 6309573.4],'csigflux':[921.0, 8717000.9], 'anything':['toto', 86]}))
    True

    """

    flux = clean[["cmagpsf", "csigmapsf"]].apply(
        lambda x: mag2fluxcal_snana(x[0], x[1]), axis=1
    )
    clean[["cmagpsf", "csigmapsf"]] = pd.DataFrame(flux.to_list())
    clean = clean.rename(columns={"cmagpsf": "cflux", "csigmapsf": "csigflux"})

    return clean


def format_data(df, source):
    """Transform filter names to ints and
    add distance to host galaxy column.

    Parameters
    ----------
    df: pd.DataFrame
       Alerts data with columns :
           For ELASTICC:
            "objectId": diaObjectId,
            "cjd": cmidPoinTai,
            "cflux": cpsFlux,
            "csigflux": cpsFluxErr,
            "cfid": cfilterName,
            "ra": ra,
            "dec": decl,
            "hostgal_zphot": hostgal_zphot,
            "hostgal_zphot_err": hostgal_zphot_err,
            "hostgal_ra": hostgal_ra,
            "hostgal_dec": hostgal_dec

           For ZTF:
            "objectId": objectId,
            "cjd": cjd,
            "cmagpsf": cmagpsf,
            "csigmapsf": csigmapsf,
            "cfid": cfid,
            "ra": ra,
            "dec": deccfid 	cra 	cdec

    source: string
        Origin of the data.
        Currently accepts 'ZTF' or 'ELASTICC'.

    Returns
    -------
    pd.DataFrame
        Formated dataset

    Examples
    --------
    >>> df = pd.DataFrame({"cfid":[['u', 'g', 'r', 'i', 'z', 'Y']], "hostgal_ra":[90], "hostgal_dec":[46], "ra":[89], "dec":[46.5]})
    >>> expect = pd.DataFrame({'cfid': {0: [0, 1, 2, 3, 4, 5]}, 'ra': {0: 89}, 'dec': {0: 46.5}, 'hostgal_dist': {0: 1118.033988749895}})
    >>> assert_frame_equal(format_data(df, 'ELASTICC'), expect)
    """

    if source == 'ELASTICC':
        # Compute distance from host
        df["hostgal_dist"] = df.apply(compute_hostgal_dist, axis=1)
        df = df.drop(columns={"hostgal_ra", "hostgal_dec"})

        # Transform band str to int
        df["cfid"] = df["cfid"].apply(map_fid)

    if source == 'ZTF':
        df[["cfid", "cjd", "cmagpsf", "csigmapsf"]] = df[
            ["cfid", "cjd", "cmagpsf", "csigmapsf"]
        ].apply(remove_nan, axis=1, result_type="expand")
        df = convert_full_dataset(df)

    return df


def keep_filter(ps, band):
    """
    Funtion that removes points from other bands than the one specified

    Parameters
    ---------
    ps: pd.Series
        each rows of the dataframe. each entries must be numeric list.
    band: int
        Integer associated with the filter to keep.

    Return
    ------
    list
        list of the same size as x, each entries is the original list from the
        current rows with only the wanted filter and the associated values from the other columns.

    Example
    -------
    >>> example = pd.Series(data = {'cfid':np.array([1, 1, 2, 1, 2]), 'anything':np.array([-2, 86.9, 58.1, 24, 42])})
    >>> filtered = keep_filter(example, 2)
    >>> (np.array_equal(filtered[0], np.array([2, 2]))) & (np.array_equal(filtered[1], np.array([58.1, 42])))
    True
    >>> example2 = pd.Series(data = {'cfid':np.array([2, 2]), 'anything':np.array([24, 42])})
    >>> filtered2 = keep_filter(example2, 1)
    >>> (np.array_equal(filtered2[0], np.array([]))) & (np.array_equal(filtered2[1], np.array([])))
    True

    """

    mask = ps["cfid"] == band

    return [np.array(_col)[mask].astype(type(_col[0])) for _col in ps]


def translate(ps):
    """Translate a cjd list by substracting maxflux point

    Parameters
    ----------
    ps: pd.Series
        Must contain ['cjd', 'cflux']

    Returns
    -------
    np.array
        Translated array. Returns empty array if input was empty

    Example
    -------
    >>> example = pd.Series(data = {'cjd':np.array([1,2,3]), 'cflux':np.array([-2, 42, 23]), 'anything':np.array(['toto', 82, -8])})
    >>> np.array_equal(translate(example), np.array([-1,  0,  1]))
    True
    >>> example2 = pd.Series(data = {'cjd':np.array([]), 'cflux':np.array([]), 'anything':np.array(['toto', 82, -8])})
    >>> np.array_equal(translate(example2), np.array([]))
    True

    """

    if len(ps["cjd"]) == 0:
        return []

    else:
        return ps["cjd"] - ps["cjd"][np.argmax(ps["cflux"])]


def normalize(ps):
    """Normalize by dividing by a data frame of maximum

    Parameters
    ----------
    ps: pd.Series
        Must contain 'cflux', 'csigflux' and 'peak'

    Returns
    -------
    pd.Series
        Dataframe with columns 'cflux' and 'csigflux' normalized

    Example
    -------
    >>> example = pd.Series(data = {'cflux':np.array([17, 35.7, -3]), 'csigflux':np.array([0.7, 1, 0]), 'peak':35.7})
    >>> out = normalize(example)
    >>> np.array_equal(np.round(out[0], 3), np.array([ 0.476,  1.   , -0.084]))
    True
    >>> np.array_equal(np.round(out[1], 3), np.array([0.02 , 0.028, 0.   ]))
    True
    >>> example2 = pd.Series(data = {'cflux':np.array([]), 'csigflux':np.array([]), 'peak':-1})
    >>> out2 = normalize(example2)
    >>> (np.array_equal(out2[0], np.array([]))) & (np.array_equal(out2[1], np.array([])))
    True
    """

    if len(ps["cflux"]) == 0:
        return ps[["cflux", "csigflux"]]

    else:
        ps["cflux"] = ps["cflux"] / ps["peak"]
        ps["csigflux"] = ps["csigflux"] / ps["peak"]
        return ps[["cflux", "csigflux"]]


def get_max(x, absolute=False):

    """Returns maximum of an array. Returns -1 if array is empty

    Parameters
    ----------
    x: np.array

    absolute: bool
        If true returns absolute maximum.
        Default is False.

    Returns
    -------
    float
        Maximum of the array or -1 if array is empty.

    Example
    -------
    >>> get_max(np.array([1, 78, -6])) == 78
    True
    >>> get_max(np.array([])) == -1
    True
    >>> get_max(np.array([1, 8, -62]), absolute=True) == -62
    True
    """

    if len(x) == 0:
        return -1

    elif absolute:
        return max(x, key=abs)

    else:
        return x.max()


def get_min(x, absolute=False):

    """Returns minimum of an array. Returns -1 if array is empty

    Parameters
    ----------
    x: np.array

    absolute: bool
        If true returns absolute minimum.
        Default is False.

    Returns
    -------
    float
        Minimum of the array or -1 if array is empty.

    Example
    -------
    >>> get_min(np.array([1, 78, -6])) == -6
    True
    >>> get_min(np.array([])) == -1
    True
    >>> get_min(np.array([1, 8, -62]), absolute=True) == 1
    True
    """

    if len(x) == 0:
        return -1

    elif absolute:
        return min(x, key=abs)

    else:
        return x.min()


def transform_data(formated, minimum_points, source):
    """Apply transformations for each filters on a flux formated dataset
            - Shift cjd so that the max flux point is at 0
            - Normalize by dividing flux and flux err by the maximum flux
            - Add a column with maxflux before normalization

    Split the results into multiple dataframes each containing only one passband.


    Parameters
    ----------
    formated : pd.DataFrame
        Dataframe of alerts formated using "format_data" function.
    minimum_points: minimum number of points for a passband to be considered valid
        The classifier requires at least two consecutive valid passbands.
    source: string
        Origin of the data.
        Currently accepts 'ZTF' or 'ELASTICC'.

    Returns
    -------
    all_transformed : list
        List of DataFrame. Each df is a transformed version of formated
        that only contains observations from one passband and valid objects.
    valid: np.array
        Boolean array describing if each object is valid.
        Objects are valid if they have at least two consecutive passbands
        containing at least k.MINIMUM_POINTS observations.

    Examples
    --------
    >>> df = uex.formated_unit
    >>> expect1_transformed = uex.expect1_transformed
    >>> expect2_transformed = uex.expect2_transformed
    >>> assert_frame_equal(transform_data(df, 4, 'ELASTICC')[0][0], expect1_transformed)
    >>> assert_frame_equal(transform_data(df, 4, 'ELASTICC')[0][2], expect2_transformed)
    >>> transform_data(df, 4, 'ELASTICC')[1][0]
    True
    """

    all_transformed = []

    if source == 'ELASTICC':
        passbands = [0, 1, 2, 3, 4, 5]

    elif source == 'ZTF':
        passbands = [1, 2]

    for band in passbands:
        transformed = formated.copy()
        transformed[["cfid", "cjd", "cflux", "csigflux"]] = transformed[
            ["cfid", "cjd", "cflux", "csigflux"]
        ].apply(keep_filter, args=(band,), axis=1, result_type="expand")

        all_transformed.append(transformed)

    condition = []
    for pair in range(len(passbands) - 1):
        condition.append((all_transformed[pair]['cjd'].apply(len) >= minimum_points) & (all_transformed[pair + 1]['cjd'].apply(len) >= minimum_points))

    valid = np.array([False] * len(formated))
    for cond in condition:
        valid = valid | cond

    all_transformed = [x[valid].copy() for x in all_transformed]

    if not valid.any():
        return all_transformed, valid

    for df in all_transformed:

        df["cjd"] = df.apply(translate, axis=1)

        df["peak"] = df["cflux"].apply(get_max)
        df[["cflux", "csigflux"]] = df.apply(normalize, axis=1)
        df["snr"] = df[["cflux", "csigflux"]].apply(
            lambda pdf: pdf["cflux"] / pdf["csigflux"], axis=1
        )

    return all_transformed, valid


def parametric_bump(ps, band):

    """Fit the lightcurves using the bump function.
    Extract the minimized parameters of the fit.

    Parameters
    ----------
    ps: pd.Series
        Alerts that have been transformed using 'transform_data' function.
        Lightcurves's max flux must be centered on 40.
        p4 guess is set to the minimum flux value.
    band: int
        Integer associated with the filter to fit.

    Returns
    -------
    list
        List of best fitting parameter values [p1, p2, p3, p4]
        Returns [0.225, -2.5, 0.038, -1] if the fit didn't converge.

    Examples
    --------
    >>> ps = pd.Series({"cjd_1":np.array([-20, 20, 40, 45, 50, 60]),\
                "cflux_1":np.array([0, .1, 1, .7, .4, .1]),\
                "csigflux_1":np.array([.01, .01, .01, .01, .01, .01])})
    >>> res = parametric_bump(ps, 1)
    >>> expected = np.array([  0.57463348, -10.94381434,   0.05843842,   0.05655547])
    >>> np.array_equal(np.around(res, 4), np.around(expected, 4))
    True
    """

    try:
        fit = curve_fit(mod.bump, ps[f"cjd_{band}"], ps[f"cflux_{band}"], sigma=ps[f"csigflux_{band}"],
                        p0=[0.225, -2.5, 0.038, get_min(ps[f"cflux_{band}"])], maxfev=k.MAXFEV)

    except RuntimeError:
        fit = [[0.225, -2.5, 0.038, -1]]

    return fit[0]


def compute_color(ps):
    """Compute the color of an alert by computing blue-red
    Proceed by virtually filling missing points of each band using the bump fit

    Parameters
    ----------
    ps: pd.Series
        Dataframe of alerts as outputed by the parametrise function.
        Lightcurves's max flux must be centered on 40

    Returns
    -------
    np.array
        Array of color blue-red at each point

    Examples
    --------
    >>> ps = pd.Series({"cjd_blue":np.array([0, 10, 20, 30, 40]),\
                "cjd_red":np.array([35, 40, 45, 50]),\
                "cflux_blue":np.array([0, .1, .4, .7, 1]),\
                "cflux_red":np.array([.7, 1, .3, 0]),\
                "bump_blue":np.array([0.5, -10,   0.06,   0.07]),\
                "bump_red":np.array([0.225, -2.5, 0.038, 10]),\
                "peak_blue":80914,\
                "peak_red":81780})
    >>> res = compute_color(ps)
    >>> expected = np.array([-820197.15423085, -822210.11468053, -823501.83859425,\
           -822870.37878502, -808280.17878222,  -24123.45023986,\
            -54721.61867977,  -12014.06603495,    6161.4019725 ])
    >>> np.array_equal(np.around(res, 4), np.around(expected, 4))
    True
    """

    # Compute fitted values at cjd from the other band
    add_from_1 = mod.bump(ps["cjd_red"], *ps["bump_blue"])
    add_from_0 = mod.bump(ps["cjd_blue"], *ps["bump_red"])

    # Add to the flux list : maintain the same order : cjd from 0 then cjd from 1
    new_cflux_0 = np.append(ps["cflux_blue"], add_from_1)
    new_cflux_1 = np.append(add_from_0, ps["cflux_red"])

    unnorm_cflux_0 = new_cflux_0 * ps["peak_blue"]
    unnorm_cflux_1 = new_cflux_1 * ps["peak_red"]

    return unnorm_cflux_0 - unnorm_cflux_1


def compute_std(x):
    """Compute standard deviation of an array.
    Return -1 if the array is empty.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        Standard deviation of the array.

    Examples
    --------
    >>> compute_std(np.array([0, 1, 2, 3, 4])) == 1.4142135623730951
    True
    >>> compute_std(np.array([])) == -1
    True
    """

    if len(x) == 0:
        return -1
    else:
        return np.std(x)


def compute_mean(x):
    """Compute mean of an array.
    Return -1 if the array is empty.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        Mean of the array.

    Examples
    --------
    >>> compute_mean(np.array([0, 1, 2, 3, 4])) == 2
    True
    >>> compute_mean(np.array([])) == -1
    True
    """

    if len(x) == 0:
        return -1
    else:
        return np.mean(x)


def parametrise(all_transformed, source, target_col=""):
    """Extract parameters from a list of dataset outputed
       by the transform_data function.

    Parameters are :
            - "ra" : right ascension
            - "dec" : declination

            For each filter:
                - 'std' : standard deviation of the flux for each filter
                - 'peak' : maximum before normalization for each filter
                - 'mean_snr' : mean signal over noise ratio for each filter
                - 'nb_points' : number of points for each filter

    Additionnaly if source is ELASTICC:
            - "hostgal_dist" : distance to host galaxy
            - "hostgal_zphot" : redshift of the host galaxy
            - "hostgal_zphot_err" : error on the redshift of the host galaxy

    Parameters
    ----------
    all_transformed : list
        List of transformed DataFrame using "transform_data" function.
    source: string
        Origin of the data.
        Currently accepts 'ZTF' or 'ELASTICC'.
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset.
        Default is ''

    Returns
    -------
    df_parameters : pd.DataFrame
        DataFrame of parameters.
        Also adds columns of cjd, cflux and csigflux that
        will be used to compute color later on.

    Example
    -------
    >>> df = uex.all_transformed_unit
    >>> expect0 = uex.expect_all_features0
    >>> expect2 = uex.expect_all_features2
    >>> parametrised = parametrise(df, 'ELASTICC')
    >>> assert_frame_equal(parametrised[0], expect0)
    >>> assert_frame_equal(parametrised[2], expect2)
    >>> exec("for i in df: i['target']='AGN'")
    >>> expect0.insert(10, "target", 'AGN')
    >>> assert_frame_equal(parametrise(df, 'ELASTICC', 'target')[0], expect0)
    """

    if source == 'ELASTICC':
        passbands = [0, 1, 2, 3, 4, 5]

    elif source == 'ZTF':
        passbands = [1, 2]

    all_features = []

    for idx, band in enumerate(passbands):

        transformed = all_transformed[idx]
        nb_points = transformed["cflux"].apply(lambda x: len(x))
        peak = transformed["peak"]
        std = transformed["cflux"].apply(compute_std)
        mean_snr = transformed["snr"].apply(compute_mean)
        ids = transformed["objectId"]
        ra = transformed["ra"]
        dec = transformed["dec"]

        if source == 'ELASTICC':
            hostgal_dist = transformed["hostgal_dist"]
            hostgal_zphot = transformed["hostgal_zphot"]
            hostgal_zphot_err = transformed["hostgal_zphot_err"]

            df_parameters = pd.DataFrame(
                data={
                    "object_id": ids,
                    "ra": ra,
                    "dec": dec,
                    "hostgal_dist": hostgal_dist,
                    "hostgal_zphot": hostgal_zphot,
                    "hostgal_zphot_err": hostgal_zphot_err,
                    f"std_{band}": std,
                    f"peak_{band}": peak,
                    f"mean_snr_{band}": mean_snr,
                    f"nb_points_{band}": nb_points,
                }
            )

        elif source == 'ZTF':
            df_parameters = pd.DataFrame(
                data={
                    "object_id": ids,
                    "ra": ra,
                    "dec": dec,
                    f"std_{band}": std,
                    f"peak_{band}": peak,
                    f"mean_snr_{band}": mean_snr,
                    f"nb_points_{band}": nb_points,
                }
            )

        if target_col != "":
            targets = transformed[target_col]
            df_parameters[target_col] = targets

        # The bump function is built to fit transient centered on 40
        transformed["cjd"] = transformed["cjd"].apply(lambda x: np.array(x) + 40)

        df_parameters[f"cflux_{band}"] = transformed["cflux"]
        df_parameters[f"csigflux_{band}"] = transformed["csigflux"]
        df_parameters[f"cjd_{band}"] = transformed["cjd"]

        all_features.append(df_parameters)

    return all_features


def merge_features(all_features, minimum_points, source, target_col=""):
    """Merge feature tables of all filters.
    Additionnaly compute color parameters :
                                - 'max_color' : absolute maximum of the color
                                - 'std_color' : standard deviation of the color

    It requires k.MINIMUM_POINTS points in two consecutive passbands.
    We compute only one color, the first that satistfies the requirement,
    checked in the following order : g-r, r-i, z-Y, u-g, i-z.


    Parameters
    ----------
    all_features: DataFrame
        Parameter dataframe, output of the "parametrise" function.
    minimum_points: int
        Minimum number of point in a filter to be considered valid.
    source: string
        Origin of the data.
        Currently accepts 'ZTF' or 'ELASTICC'.
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset.
        Default is ''.

    Returns
    -------
    if source == 'ELASTICC':
        ordered_features : pd.DataFrame
            Final features dataset with ordered columns :
            ["object_id","ra","dec","hostgal_dist","hostgal_zphot",
            "hostgal_zphot_err","std_0","std_1","std_2","std_3",
            "std_4","std_5","peak_0","peak_1","peak_2","peak_3",
            "peak_4","peak_5","mean_snr_0","mean_snr_1","mean_snr_2",
            "mean_snr_3","mean_snr_4","mean_snr_5","nb_points_0",
            "nb_points_1","nb_points_2","nb_points_3","nb_points_4",
            "nb_points_5","std_color", "max_color"]

    elif source == 'ZTF':
        ordered_features : pd.DataFrame
                Final features dataset with ordered columns :
                ["object_id","ra","dec","std_1","std_2",
                "peak_1","peak_2","mean_snr_1","mean_snr_2",
                "nb_points_1","nb_points_2","std_color", "max_color"]


    Example
    -------
    >>> df = uex.all_features_unit
    >>> expected = uex.features_unit
    >>> assert_frame_equal(merge_features(df, 4, 'ELASTICC'), expected)
    >>> exec("for i in df: i['target']='AGN'")
    >>> expected['target'] = 'AGN'
    >>> assert_frame_equal(merge_features(df, 4, 'ELASTICC', target_col='target'), expected)
    """

    warnings.filterwarnings('ignore', '.*Covariance of the parameters could not be estimated.*')

    if source == 'ELASTICC':
        passbands = [0, 1, 2, 3, 4, 5]
        pairs = [[1, 2], [2, 3], [4, 5], [0, 1], [3, 4]]

    elif source == 'ZTF':
        passbands = [1, 2]
        pairs = [[1, 2]]

    features = all_features[0]

    # Avoid having twice the same column

    for band in range(1, len(passbands)):
        all_features[band] = all_features[band].drop(
            columns={
                "object_id",
                "ra",
                "dec",
                "hostgal_dist",
                "hostgal_zphot",
                "hostgal_zphot_err",
                target_col,
            }, errors='ignore')

        features = features.join(all_features[band])

    if source == 'ELASTICC':
        ordered_features = features[
            [
                "object_id",
                "ra",
                "dec",
                "hostgal_dist",
                "hostgal_zphot",
                "hostgal_zphot_err",
                "std_0",
                "std_1",
                "std_2",
                "std_3",
                "std_4",
                "std_5",
                "peak_0",
                "peak_1",
                "peak_2",
                "peak_3",
                "peak_4",
                "peak_5",
                "mean_snr_0",
                "mean_snr_1",
                "mean_snr_2",
                "mean_snr_3",
                "mean_snr_4",
                "mean_snr_5",
                "nb_points_0",
                "nb_points_1",
                "nb_points_2",
                "nb_points_3",
                "nb_points_4",
                "nb_points_5",
            ]
        ].copy()

    elif source == 'ZTF':
        ordered_features = features[
            [
                "object_id",
                "ra",
                "dec",
                "std_1",
                "std_2",
                "peak_1",
                "peak_2",
                "mean_snr_1",
                "mean_snr_2",
                "nb_points_1",
                "nb_points_2"
            ]
        ].copy()

    features['color_not_computed'] = True
    features[['bump_blue', 'blue_band', 'cjd_blue', 'cflux_blue', 'peak_blue',
              'bump_red', 'cjd_red', 'cflux_red', 'peak_red']] = None

    for pair in pairs:
        mask_color_compute = (features[f'cjd_{pair[0]}'].apply(len) >= k.MINIMUM_POINTS) & (features[f'cjd_{pair[1]}'].apply(len) >= k.MINIMUM_POINTS) & features['color_not_computed']

        for colo_idx, colo in enumerate(['blue', 'red']):
            features.loc[mask_color_compute, f'bump_{colo}'] = features[mask_color_compute].apply(parametric_bump, axis=1, args=(pair[colo_idx],))
            for colname in ['cjd', 'cflux', 'peak']:
                features.loc[mask_color_compute, f'{colname}_{colo}'] = features.loc[mask_color_compute, f'{colname}_{pair[colo_idx]}']

        features.loc[mask_color_compute, 'blue_band'] = pair[0]
        features.loc[mask_color_compute, 'color_not_computed'] = False

    # Add color features
    features['color'] = features.apply(compute_color, axis=1)

    ordered_features["std_color"] = features['color'].apply(compute_std)
    ordered_features["max_color"] = features['color'].apply(get_max, args=(True,))

    # Make sure that no value is above 2**32 (scipy uses float32)
    max_size = 2**30
    float_cols = ordered_features.columns[ordered_features.columns != 'object_id']

    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] > max_size, max_size)
    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] < -max_size, -max_size)

    if target_col != "":
        targets = features[target_col]
        ordered_features[target_col] = targets

    return ordered_features


def get_probabilities(clf, features, valid):
    """Returns probabilty of being an AGN predicted by the classifier.

    Parameters
    ----------
    clf: RandomForestClassifier
        Binary AGN vs non AGN classifier.
    features: pd.DataFrame
        Features extracted from the objects.
        Outputed by merge_features.
    valid: np.array
        Bool array, indicates if both passband respect the minimum number of points.

    Returns
    -------
    final_proba : np.array
        ordered probabilities of being an AGN.
        Proba = 0 if the object is not valid.

    Examples
    --------
    >>> res = get_probabilities(uex.clf_unit, uex.features_unit, [True])
    >>> len(res)
    1
    >>> res[0]!=-1
    True
    """

    final_proba = np.array([0.0] * len(valid)).astype(np.float64)

    if len(features) > 0:
        features = features.replace(np.inf, 0.0).replace(np.nan, 0.0)
        agn_or_not = clf.predict_proba(features.iloc[:, 1:])
        index_to_replace = features.iloc[:, 1:].index
        final_proba[index_to_replace.values] = agn_or_not[:, 1]

    return final_proba


if __name__ == "__main__":

    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
