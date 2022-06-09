import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
import models as mod
import numpy as np


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
    """

    if magpsf is None:
        return None, None

    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10**10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err


def remove_nan(x):
    """
    funtion that remove nan values from list contains in columns

    Paramters
    ---------
    x : pd.Series
        each rows of the dataframe. each entries must be numeric list

    Return
    ------
    list_without_nan : list
        list of the same size as x, each entries is the original list from the
        current rows without the nan values and the associated values from the other columns.
    """

    mask = np.equal(x["cmagpsf"], x["cmagpsf"])

    return [np.array(_col)[mask].astype(type(_col[0])) for _col in x]


def keep_filter(x, band):
    """
    funtion that removes points fron other bands than the one specified

    Parameters
    ---------
    x : pd.Series
        each rows of the dataframe. each entries must be numeric list

    Return
    ------
    list_with_oneband : list
        list of the same size as x, each entries is the original list from the
        current rows with only the wanted filter and the associated values from the other columns.
    """

    mask = x["cfid"] == band

    return [np.array(_col)[mask].astype(type(_col[0])) for _col in x]


def clean_data(pdf: pd.DataFrame):
    """
    Remove all nan values from 'cmagpsf' along with the corresponding values
    inside "cfid", "cjd", 'csigmapsf'.

    Paramters
    ---------
    pdf : pd.DataFrame
        Dataframe of alerts from Fink

    Return
    ------
    pdf_without_nan : pd.DataFrame
         DataFrame with nan and corresponding measurement removed
    """

    pdf = pdf.reset_index(drop=True)

    # Remove NaNs
    pdf[["cfid", "cjd", "cmagpsf", "csigmapsf"]] = pdf[
        ["cfid", "cjd", "cmagpsf", "csigmapsf"]
    ].apply(remove_nan, axis=1, result_type="expand")

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
    """

    flux = clean[["cmagpsf", "csigmapsf"]].apply(
        lambda x: mag2fluxcal_snana(x[0], x[1]), axis=1
    )
    clean[["cmagpsf", "csigmapsf"]] = pd.DataFrame(flux.to_list())
    clean = clean.rename(columns={"cmagpsf": "cflux", "csigmapsf": "csigflux"})

    return clean


def translate(pdf):

    """Translate a cjd list by substracting maxflux point

    Parameters
    ----------
    pdf: pd.DataFrame
        Must contain ['cjd', 'cflux']

    Returns
    -------
    np.array
        Translated array. Returns empty array if input was empty
    """

    if len(pdf["cjd"]) == 0:
        return []

    else:
        return pdf["cjd"] - pdf["cjd"][np.argmax(pdf["cflux"])]


def normalize(x, maxdf):

    """Normalize by dividing by a data frame of maximum

    Parameters
    ----------
    x: np.array
        Values to be divided
    max_value: float
        maximum value used for the normalisation

    Returns
    -------
    np.array
        Normalized array. Returns empty array if input was empty
    """

    max_value = maxdf[x.index]

    if len(x) == 0:
        return []

    else:
        return x / max_value


def get_max(x):

    """Returns maximum of an array. Returns -1 if array is empty

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        Maximum of the array or -1 if array is empty
    """

    if len(x) == 0:
        return -1

    else:
        return x.max()


def compute_snr(pdf):

    """Compute signal to noise ratio

    Parameters
    ----------
    pdf: pd.DataFrame
        Dataframe of alerts from Fink with nan removed and converted to flux
        Must contain columns ['cflux', 'csigflux']

    Returns
    -------
    pd.Series
        Signal to noise ratio
    """
    return pdf["cflux"] / pdf["csigflux"]


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

        maxdf = df["cflux"].apply(get_max)

        df[["cflux"]] = df[["cflux"]].apply(normalize, args=(maxdf,))
        df[["csigflux"]] = df[["csigflux"]].apply(normalize, args=(maxdf,))
        df["snr"] = df[["cflux", "csigflux"]].apply(compute_snr, axis=1)

        df["peak"] = maxdf

    return transformed_1, transformed_2


def parametric_bump(pdf):

    """Fit the lightcurves using the bump function. Extract the parameters
    Parameters
    ----------
    pdf : pd.DataFrame
        Dataframe of alerts from Fink with nan removed and converted to flux.
        Flux must be normalised
        Lightcurves's max flux must be centered on 40
    Returns
    -------
    parameters : list
        List of best fitting parameter values [p1, p2, p3, p4]
    """

    parameters_dict = {"p1": 0.225, "p2": -2.5, "p3": 0.038, "p4": 0}

    least_squares = LeastSquares(pdf["cjd"], pdf["cflux"], pdf["csigflux"], mod.bump)
    fit = Minuit(least_squares, **parameters_dict)

    fit.migrad()

    parameters = []
    for fit_v in range(len(fit.values)):
        parameters.append(fit.values[fit_v])

    return parameters


def compute_color(pdf):

    """Compute the color of an alert by computing g-r (before normalisation)
    Proceed by virtually filling missing points of each band using the bump fit

    Parameters
    ----------
    pdf : pd.DataFrame
        Dataframe of alerts from Fink with nan removed and converted to flux.
        Flux must be normalised with normalization factor inside column ['peak']
        Lightcurves's max flux must be centered on 40

    Returns
    -------
    np.array
        Array of color g-r at each point

    """

    # Compute fitted values at cjd from the other band
    add_from_2 = mod.bump(pdf["cjd_2"], *pdf["bump_1"])
    add_from_1 = mod.bump(pdf["cjd_1"], *pdf["bump_2"])

    # Add to the flux list : maintain the same order : cjd from 1 then cjd from 2
    new_cflux_1 = np.append(pdf["cflux_1"], add_from_2)
    new_cflux_2 = np.append(add_from_1, pdf["cflux_2"])

    unnorm_cflux_1 = new_cflux_1 * pdf["peak_1"]
    unnorm_cflux_2 = new_cflux_2 * pdf["peak_2"]

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
    """

    nb_points = transformed["cflux"].apply(lambda x: len(x))
    peak = transformed["peak"]
    std = transformed["cflux"].apply(np.std)
    mean_snr = transformed["snr"].apply(np.mean)
    ids = transformed["objectId"]

    valid = nb_points >= minimum_points

    df_parameters = pd.DataFrame(
        data={
            "object_id": ids,
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
    """

    # Avoid having twice the same column
    if target_col == "":
        features_2 = features_2.drop(columns={"object_id"})
    else:
        features_2 = features_2.drop(columns={"object_id", target_col})

    features = features_1.join(features_2)
    valid = features["valid_1"] & features["valid_2"]
    features = features.drop(columns=["valid_1", "valid_2"])

    ordered_features = features[
        [
            "object_id",
            "std_1",
            "std_2",
            "peak_1",
            "peak_2",
            "mean_snr_1",
            "mean_snr_2",
            "nb_points_1",
            "nb_points_2",
        ]
    ].copy()

    # Add color features
    color = features.apply(compute_color, axis=1)
    ordered_features["std_color"] = color.apply(np.std)
    ordered_features["max_color"] = color.apply(
        lambda x: max(np.min(x), np.max(x), key=abs)
    )

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

    """

    final_proba = np.array([-1] * len(features["object_id"])).astype(np.float64)

    agn_or_not = clf.predict_proba(features.loc[valid].iloc[:, 1:])

    index_to_replace = features.loc[valid].iloc[:, 1:].index

    final_proba[index_to_replace.values] = agn_or_not[:, 0]

    return final_proba
