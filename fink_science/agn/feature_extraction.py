import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
import fink_science.agn.models as mod
from pandas.testing import assert_frame_equal  # noqa: F401
import fink_science.agn.kernel as k  # noqa: F401
import numpy as np
import pickle  # noqa: F401


def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
    """Conversion from magnitude to Fluxcal from SNANA manual
    Parameterspip install --upgrade pip
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


def keep_filter(ps, band):
    """
    funtion that removes points from other bands than the one specified

    Parameters
    ---------
    ps : pd.Series
        each rows of the dataframe. each entries must be numeric list

    Return
    ------
    list_with_oneband : list
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


def clean_data(pdf: pd.DataFrame):
    """
    Remove all nan values from 'cmagpsf' along with the corresponding values
    inside "cfid", "cjd", 'csigmapsf', 'cra', 'cdec'.
    Keeps only one value for cra and cdec

    Paramters
    ---------
    pdf : pd.DataFrame
        Dataframe of alerts from Fink

    Return
    ------
    pdf_without_nan : pd.DataFrame
         DataFrame with nan and corresponding measurement removed.
         Replaced cols 'cra' and 'cdec' to 'ra' and 'dec'

    Examples
    --------
    >>> example = pd.DataFrame(data = {"cfid":[[1, 2, 2]], "cjd":[[20, np.nan, 10.5]],\
                     'cmagpsf':[[20.,np.nan, 10.5]],'csigmapsf':[[1.,np.nan, 1.6]],\
                     'cra':[[-7.,np.nan, -7]],'cdec':[[18.2, np.nan, 18.2]],\
                     'anything':'toto'})

    >>> expected = pd.DataFrame(data = {"cfid":[[1, 2]], "cjd":[[20, 10]],\
                         'cmagpsf':[[20., 10.5]],'csigmapsf':[[1., 1.6]],\
                         'ra':-7., 'dec':18.2,\
                         'anything':'toto'})

    >>> pd.testing.assert_frame_equal(expected, clean_data(example))
    """

    pdf = pdf.reset_index(drop=True)

    # Remove NaNs
    pdf[["cfid", "cjd", "cmagpsf", "csigmapsf", "cra", "cdec"]] = pdf[
        ["cfid", "cjd", "cmagpsf", "csigmapsf", "cra", "cdec"]
    ].apply(remove_nan, axis=1, result_type="expand")

    pdf['cra'] = pdf['cra'].apply(lambda x: x[0])
    pdf['cdec'] = pdf['cdec'].apply(lambda x: x[0])
    pdf = pdf.rename(columns={'cra': 'ra', 'cdec': 'dec'})

    return pdf


def convert_full_dataset(clean: pd.DataFrame):
    """
    Convert all mag and mag err to flux and flux err

    Paramters
    ---------
    clean : pd.DataFrame
        Dataframe of alerts from Fink with nan removed

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
    maxi: np.array
        array of all maximum values. -1 if alert has no points

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
        If true returns absolute maximum
        Default is False

    Returns
    -------
    float
        Maximum of the array or -1 if array is empty

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
        If true returns absolute minimum
        Default is False

    Returns
    -------
    float
        Minimum of the array or -1 if array is empty

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


def transform_data(converted):

    """Apply transformations for each band on a flux converted dataset
            - Shift cjd so that the max flux point is at 0
            - Normalize by dividing flux and flux err by the maximum flux
            - Add a column with maxflux before normalization


    Parameters
    ----------
    converted : pd.DataFrame
        Dataframe of alerts from Fink with nan removed and converted to flux

    Returns
    -------
    transformed_1 : pd.DataFrame
        Transformed DataFrame that only contains passband g


    transformed_2 : pd.DataFrame
        Transformed DataFrame that only contains passband r

        Examples
    --------
    >>> example = pd.DataFrame(data = {"cfid": [np.array([1, 2, 2]), np.array([2])],\
                                    "cjd" : [np.array([3, 3.5, 5]), np.array([7])],\
                                    "cflux" : [np.array([10, 20, 25]), np.array([10.5])],\
                                    "csigflux" : [np.array([1, 2, 2.5]), np.array([-10.5])],\
                                    "anything" : [np.array([0.1, 2]), np.array(['toto'])]})

    >>> expect_2 = pd.DataFrame(data = {"cfid": [np.array([2, 2]), np.array([2])],\
                                    "cjd" : [np.array([-1.5, 0]), np.array([0])],\
                                    "cflux" : [np.array([0.8, 1]), np.array([1])],\
                                    "csigflux" : [np.array([0.08, 0.1]), np.array([-1])],\
                                    "anything" : [np.array([0.1, 2]), np.array(['toto'])],\
                                    "peak" : [25, 10.5],\
                                    "snr" : [np.array([10, 10]), np.array([-1])]})

    >>> expect_1 = pd.DataFrame(data = {"cfid": [np.array([1]), np.array([])],\
                                    "cjd" : [np.array([0.]), np.array([])],\
                                    "cflux" : [np.array([1.]), np.array([])],\
                                    "csigflux" : [np.array([0.1]), np.array([])],\
                                    "anything" : [np.array([0.1, 2]), np.array(['toto'])],\
                                    "peak" : [10, -1],\
                                    "snr" : [np.array([10.]), np.array([])]})

    >>> pd.testing.assert_frame_equal((transform_data(example)[0]).round(2), expect_1)
    >>> pd.testing.assert_frame_equal((transform_data(example)[1]).round(2), expect_2)

    """

    # Create a dataframe with only measurement from band 1
    transformed_1 = converted.copy()

    transformed_1[["cfid", "cjd", "cflux", "csigflux"]] = transformed_1[
        ["cfid", "cjd", "cflux", "csigflux"]
    ].apply(keep_filter, args=(1,), axis=1, result_type="expand")

    # Create a dataframe with only measurement from band 2
    transformed_2 = converted.copy()
    transformed_2[["cfid", "cjd", "cflux", "csigflux"]] = transformed_2[
        ["cfid", "cjd", "cflux", "csigflux"]
    ].apply(keep_filter, args=(2,), axis=1, result_type="expand")

    for df in [transformed_1, transformed_2]:

        df["cjd"] = df.apply(translate, axis=1)

        df["peak"] = df["cflux"].apply(get_max)
        df[["cflux", "csigflux"]] = df.apply(normalize, axis=1)
        df["snr"] = df[["cflux", "csigflux"]].apply(
            lambda pdf: pdf["cflux"] / pdf["csigflux"], axis=1
        )

    return transformed_1, transformed_2


def parametric_bump(ps):

    """Fit the lightcurves using the bump function. Extract the parameters
    Parameters
    ----------
    ps : pd.Series
        pd Series of alerts from Fink with nan removed and converted to flux.
        Flux must be normalised
        Lightcurves's max flux must be centered on 40.
        p4 guess is set to the minimum flux value
    Returns
    -------
    parameters : list
        List of best fitting parameter values [p1, p2, p3, p4]

    Examples
    --------

    >>> example = pd.Series(data = {"cjd" : np.array([-1, 0]),\
                            "cflux" : np.array([20, 300]),\
                            "csigflux" : np.array([1, 3]),\
                            "anything" : np.array(['toto'])})

    >>> np.array_equal(np.round(parametric_bump(example), 3), np.array([ 13.465, 119.56 , -14.035,  47.9  ]))
    True

    >>> example2 = pd.Series(data = {"cjd" : np.array([]),\
                            "cflux" : np.array([]),\
                            "csigflux" : np.array([]),\
                            "anything" : np.array(['toto'])})

    >>> np.array_equal(np.round(parametric_bump(example2), 3), np.array([ 0.225, -2.5  ,  0.038, -1.   ]))
    True
    """

    parameters_dict = {"p1": 0.225, "p2": -2.5, "p3": 0.038, "p4": get_min(ps["cflux"])}

    least_squares = LeastSquares(ps["cjd"], ps["cflux"], ps["csigflux"], mod.bump)
    fit = Minuit(least_squares, **parameters_dict)

    fit.migrad()

    parameters = []
    for fit_v in range(len(fit.values)):
        parameters.append(fit.values[fit_v])

    return parameters


def compute_color(ps):

    """Compute the color of an alert by computing g-r (before normalisation)
    Proceed by virtually filling missing points of each band using the bump fit

    Parameters
    ----------
    ps : pd.Series
        Dataframe of alerts from Fink with nan removed and converted to flux.
        Flux must be normalised with normalization factor inside column ['peak']
        Lightcurves's max flux must be centered on 40

    Returns
    -------
    np.array
        Array of color g-r at each point

    Examples
    --------

    >>> example = pd.Series(data = {'cjd_1':np.array([1, 2]), 'cjd_2':np.array([3]),\
                            "cflux_1":np.array([0.3, 1]), "cflux_2":np.array([1]),\
                            'peak_1':np.array([500]), 'peak_2':np.array([321]),\
                            'bump_1':np.array([-10.596,  10.999,   0.067,  47.9  ]), 'bump_2':np.array([ 0.225, -2.5  ,  0.038,  0.   ])})

    >>> np.array_equal(np.round(compute_color(example), 1), np.array([  138.7,   486.5, 23629. ]))
    True

    """

    # Compute fitted values at cjd from the other band
    add_from_2 = mod.bump(ps["cjd_2"], *ps["bump_1"])
    add_from_1 = mod.bump(ps["cjd_1"], *ps["bump_2"])

    # Add to the flux list : maintain the same order : cjd from 1 then cjd from 2
    new_cflux_1 = np.append(ps["cflux_1"], add_from_2)
    new_cflux_2 = np.append(add_from_1, ps["cflux_2"])

    unnorm_cflux_1 = new_cflux_1 * ps["peak_1"]
    unnorm_cflux_2 = new_cflux_2 * ps["peak_2"]

    # Return g-r
    return unnorm_cflux_1 - unnorm_cflux_2


def parametrise(transformed, minimum_points, band, target_col=""):

    """Extract parameters from a transformed dataset. Construct a new DataFrame
    Parameters are :  - 'nb_points' : number of points
                      - 'std' : standard deviation of the flux
                      - 'peak' : maximum before normalization
                      - 'mean_snr' : mean signal over noise ratio
                      - 'valid' : is the number of point above the minimum (boolean)

    Also compute a fit using the bump function on each lightcurve. Parameters
    of the fit will later be used to compute color parameters

    Parameters
    ----------
    transformed : pd.DataFrame
        Transformed DataFrame that only contains a single passband
    minimum_points : int
        Minimum number of points in that passband to be considered valid
    band : int
        Passband of the dataframe
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset
        Default is ''

    Returns
    -------
    df_parameters : pd.DataFrame
        DataFrame of parameters.
        Contains additionnal columns which will be used to compute color parameters

    Example
    -------
    >>> example = pd.DataFrame(data = {"objectId": [np.array(5485), np.array(2000)],\
                                    "ra" : [np.array(90), np.array(-42)],\
                                    "dec" : [np.array(70), np.array(-2)],\
                                    "cjd" : [np.array([-1.5, 0]), np.array([0])],\
                                    "cflux" : [np.array([0.8, 1]), np.array([1])],\
                                    "csigflux" : [np.array([0.08, 0.1]), np.array([-1])],\
                                    "peak" : [25, 10.5],\
                                    "snr" : [np.array([10, 10]), np.array([-1])],\
                                    "target" : ['AGN', 'other']})

    >>> param = parametrise(example, 2, 1, target_col='target')
    >>> valid = param['valid_1']
    >>> (valid[0] == True) & (valid[1] == False)
    True

    >>> bump = param['bump_1']
    >>> (len(bump[0]) & len(bump[0])) == 4
    True

    >>> (param.keys() == ['object_id', 'ra', 'dec', 'std_1', 'peak_1', 'mean_snr_1', 'nb_points_1', 'valid_1',\
           'target', 'bump_1', 'cflux_1', 'cjd_1']).sum()==12
    True

    """

    nb_points = transformed["cflux"].apply(lambda x: len(x))
    peak = transformed["peak"]
    std = transformed["cflux"].apply(np.std)
    mean_snr = transformed["snr"].apply(np.mean)
    ids = transformed["objectId"]
    ra = transformed['ra']
    dec = transformed['dec']

    valid = nb_points >= minimum_points

    df_parameters = pd.DataFrame(
        data={
            "object_id": ids,
            "ra": ra,
            "dec": dec,
            f"std_{band}": std,
            f"peak_{band}": peak,
            f"mean_snr_{band}": mean_snr,
            f"nb_points_{band}": nb_points,
            f"valid_{band}": valid,
        }
    )

    if target_col != "":
        targets = transformed[target_col]
        df_parameters["target"] = targets

    # Compute missing values for the color
    # The bump function is built to fit transient centered on 40
    transformed["cjd"] = transformed["cjd"].apply(lambda x: np.array(x) + 40)
    bump_parameters = transformed.apply(parametric_bump, axis=1)
    df_parameters[f"bump_{band}"] = bump_parameters

    df_parameters[f"cflux_{band}"] = transformed["cflux"]
    df_parameters[f"cjd_{band}"] = transformed["cjd"]

    return df_parameters


def merge_features(features_1, features_2, target_col=""):

    """Merge feature tables of band g and r.
    Also merge valid columns into one.
    Compute color parameters : - 'max_color' : absolute maximum of the color
                               - 'std_color' : standard deviation of the color

    Parameters
    ----------
    features_1: pd.DataFrame
        features of band g
    features_2: pd.DataFrame
        features of band r
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset
        Default is ''

    Returns
    -------
    ordered_features : pd.DataFrame
        Final features dataset with ordered columns :
        ['object_id', 'std_1', 'std_2', 'peak_1', 'peak_2', 'mean_snr_1',
        'mean_snr_2', 'nb_points_1', 'nb_points_2', 'std_color', 'max_color']

    valid : np.array
        Boolean array, indicates if both passband respect the minimum number of points

    Examples
    --------
    >>> band1 = pd.DataFrame(data = {'object_id':42, 'valid_1':True,\
                             'ra':90, 'dec':90,\
                             'std_1':4.1, 'peak_1':2563,\
                             'mean_snr_1':0.8, 'nb_points_1':4,\
                             'bump_1':[np.array([0.225, -2.5, 0.038, 0.])],\
                             'cjd_1':[np.array([1, 20, 40, 45])],\
                             'cflux_1':[np.array([0.1, 0.5, 1, 0.7])]})

    >>> band2 = pd.DataFrame(data = {'object_id':42, 'valid_2':False,\
                             'ra':90, 'dec':90,\
                             'std_2':3.1, 'peak_2':263,\
                             'mean_snr_2':0.2, 'nb_points_2':2,\
                             'bump_2':[np.array([0.285, -2.3, 0.048, 0.2])],\
                             'cjd_2':[np.array([13, 40])],\
                             'cflux_2':[np.array([0.3, 1])]})

    >>> result = merge_features(band1, band2)
    >>> len(result)
    2
    >>> result[1][0]
    False
    >>> expected = pd.DataFrame(data = {'object_id':42, 'ra':90, 'dec':90,\
                                 'std_1':4.1, 'std_2':3.1, 'peak_1':2563, 'peak_2':263,\
                                 'mean_snr_1':0.8, 'mean_snr_2':0.2,\
                                 'nb_points_1':4, 'nb_points_2':2, 'std_color':747.15, 'max_color':2271.83}, index=[0])

    >>> pd.testing.assert_frame_equal(expected, result[0].round(2))

    >>> band1_target = pd.DataFrame(data = {'object_id':42, 'valid_1':True,\
                             'ra':90, 'dec':90,\
                             'std_1':4.1, 'peak_1':2563,\
                             'mean_snr_1':0.8, 'nb_points_1':4,\
                             'bump_1':[np.array([0.225, -2.5, 0.038, 0.])],\
                             'cjd_1':[np.array([1, 20, 40, 45])],\
                             'cflux_1':[np.array([0.1, 0.5, 1, 0.7])], \
                             'target':['AGN']})

    >>> band2_target = pd.DataFrame(data = {'object_id':42, 'valid_2':False,\
                             'ra':90, 'dec':90,\
                             'std_2':3.1, 'peak_2':263,\
                             'mean_snr_2':0.2, 'nb_points_2':2,\
                             'bump_2':[np.array([0.285, -2.3, 0.048, 0.2])],\
                             'cjd_2':[np.array([13, 40])],\
                             'cflux_2':[np.array([0.3, 1])],\
                             'target':['AGN']})

    >>> result_target = merge_features(band1_target, band2_target, target_col="target")
    """

    # Avoid having twice the same column
    if target_col == "":
        features_2 = features_2.drop(columns={"object_id", "ra", "dec"})
    else:
        features_2 = features_2.drop(columns={"object_id", "ra", "dec", target_col})

    features = features_1.join(features_2)
    valid = features["valid_1"] & features["valid_2"]
    features = features.drop(columns=["valid_1", "valid_2"])

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

    # Add color features
    color = features.apply(compute_color, axis=1)
    ordered_features["std_color"] = color.apply(np.std)
    ordered_features["max_color"] = color.apply(get_max, args=(True,))

    if target_col != "":
        ordered_features[target_col] = features[target_col]

    return ordered_features, valid


def get_probabilities(clf, features, valid):

    """Returns probabilty of being an AGN predicted by the classfier

    Parameters
    ----------
    clf: RandomForestClassifier
        Binary AGN vs non AGN classifier

    features: pd.DataFrame
        Features extracted from the objects.
        Outputed by merge_features

    valid: np.array
        Bool array, indicates if both passband respect the minimum number of points

    Returns
    -------
    final_proba : np.array
        ordered probabilities of being an AGN

    Examples
    --------
    >>> example = pd.DataFrame(data = {'object_id':[42, 24, 60], 'std_1':[4.1, 0.8, 0.07],\
                                 'ra':[50,300,200], 'dec':[-5.2, -10, 30],\
                                 'std_2':[0.7, 0.3, 0.07], 'peak_1':[2563, 10000, 1500], 'peak_2':[263, 10000, 1500],\
                                 'mean_snr_1':[0.8, 3, 6], 'mean_snr_2':[0.2, 4, 6],\
                                 'nb_points_1':[4,18, 5], 'nb_points_2':[2,12, 5],\
                                 'std_color':[74.15, 3, 0], 'max_color':[2271.83, 500, 0]})

    >>> valid = np.array([False, True, True])
    >>> clf = pickle.load(open(k.CLASSIFIER, "rb"))
    >>> proba = get_probabilities(clf, example, valid)
    >>> len(proba)
    3
    >>> proba[2]!=-1
    True
    >>> proba[1]!=-1
    True
    >>> proba[0]==-1
    True

    >>> valid2 = np.array([False, False, False])
    >>> proba = get_probabilities(clf, example, valid2)
    >>> len(proba)
    3
    >>> proba[2]==-1
    True
    >>> proba[1]==-1
    True
    >>> proba[0]==-1
    True
    """

    final_proba = np.array([-1] * len(features["object_id"])).astype(np.float64)

    valid_alerts = features.loc[valid]

    if len(valid_alerts) > 0:
        agn_or_not = clf.predict_proba(valid_alerts.iloc[:, 1:])
        index_to_replace = valid_alerts.iloc[:, 1:].index
        final_proba[index_to_replace.values] = agn_or_not[:, 1]

    return final_proba


if __name__ == "__main__":

    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
