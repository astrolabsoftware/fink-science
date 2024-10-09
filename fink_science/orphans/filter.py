from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import os
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing

from filter_utils import compute_duration_between_first_and_peak, compute_rate, compute_color, fit_light_curve

from fink_filters.tester import spark_unit_tests


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def orphan_grb(ctimemjd, cabmags, cabmagserr, cfilts):
    """ Filter to extract orphan GRB candidates.

    The filter is based on different features calculated from the light curve:
    - Duration between the first detection and the peak
    - Increase and decrease rates
    - Colour between different filter pairs
    - 4 parameters and chi2 from fitting the light curve

    A Boosting Decision Tree classifier was trained to discriminate orphans using these features.

    Parameters
    ----------
    ctimemjd: pandas.Series of list of float
        Concatenated time in MJD for the object
    cabmags: pandas.Series of list of float, or Nan
        Concatenated magnitude for the object
    cabmagserr: pandas.Series of list of float, or Nan
        Concatenated errors on the magnitude for the object
    cfilts: pandas.Series of list of int
        Concatenated filters for the object

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """


    # keep only non-NaN values and sort points by time
    times, mags, err, filts = clean_and_sort_light_curve(ctimemjd, cabmags, cabmagserr, cfilts)


    # duration between the first detection and the peak
    duration = np.array(
        [
            compute_duration_between_first_and_peak(t, m) for t, m in zip(times.values, mags.values)
        ]
    )

    # increase rate and decrease rates (average, in the 1/3 and the 3/3 parts of the light curve)
    rates = np.array(
        [
            compute_rate(t, m, f) for t, m, f in zip(times.values, mags.values, filts.values)
        ]
    )

    # colours for the pairs (u, g), (g, r), (r, i), (i, z), (z, Y)
    colour = np.array(
        [
            compute_color(t, m, f) for t, m, f in zip(times.values, mags.values, filts.values)
        ]
    )

    # parameters of the fit A, B, C, D and chi2
    fit_parameters = np.array(
        [
            fit_light_curve(t, m, e, f) for t, m, e, f in zip(times.values, mags.values, err.values, filts.values)
        ]
    )

    # gather all the features in one DataFrame and normalise them
    features = {
        'duration': duration,
        'increase_rate': rates[:, 0],
        'decrease_rate': rates[:, 1],
        'decrease_rate_1': rates[:, 2],
        'decrease_rate_3': rates[:, 3],
        'colour': colour,
        'A': fit_parameters[:, 0],
        'B': fit_parameters[:, 1],
        'C': fit_parameters[:, 2],
        'D': fit_parameters[:, 3],
        'A/B': fit_parameters[:, 0] / fit_parameters[:, 1],
        'chi2': fit_parameters[:, 4]
    }

    df_features = pd.DataFrame(data=features)

    # clean non-valid data
    df_features.replace([np.inf, -np.inf], 1000, inplace=True)
    df_features.fillna(0, inplace=True)

    features_norm = preprocessing.normalize(df_features, norm='max')

    # use Boosting Decision Tree classifier
    curdir = os.path.dirname(os.path.abspath(__file__))
    model_path = curdir + '/data/models/'
    clf = joblib.load(model_path + 'model_orphans.pkl')
    proba = clf.predict_proba(features_norm)

    # `True` for the objects that have a probability > 0.999999 to be an orphan, else `False`
    threshold = 0.999999
    res = (proba[:, 1] >= threshold)

    return pd.Series(res, dtype=bool)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)