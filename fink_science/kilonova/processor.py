from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

import pandas as pd
import numpy as np

import os

from PredictLightCurve import PredictLightCurve

from fink_science import __file__
from fink_science.conversion import mag2fluxcal_snana
from fink_science.utilities import load_scikit_model


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_kn_pca(jd, fid, magpsf, sigmapsf, model=None, num_pc_components=None, min_flux_threshold=None) -> pd.Series:

    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (bazin fit).
    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/kilonova_model.pkl` is loaded.
    num_pc_components: int
        Number of principle components to be considered for the fit. Default is none, in which case the default
        value of 3 is assigned
    min_flux_threshold: int
        Minimum value of amplitude of a band for prediction. Default is None, in which case the default
        value of 200 is assigned

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-kilonova) and 1 (kilonova).


    Example
    __________

    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F
    >>> df = spark.read.load(ztf_alert_sample)
    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']
    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

     Examples
    ----------

    # Perform the fit + classification (default model)
    >>> args = ['cjd', 'cfid', 'cmagpsf','csigmapsf']
    >>> df = df.withColumn('pKN',  rfscore_kn_pca(*args))

    >>>df_change.select(['pKN']).show()
    """

    if num_pc_components is None:
        num_pc_components = 3
    if min_flux_threshold is None:
        min_flux_threshold = 200

    if num_pc_components != 3:
        print('error ')
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 3
    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])

    # need to define this later
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/kilonova_model.pkl'
        clf = load_scikit_model(model)

    # else:
    #    curdir = os.path.dirname(os.path.abspath(__file__))
    #    model = curdir + 'models/pickle_model.pkl'
    #
    #    clf = load_scikit_model(model)

    # remember to initialize bands and
    bands = ['g', 'r']
    test_features = []
    ids = pd.Series(range(len(jd)))
    for id in ids[mask]:
        # compute flux and flux error
        data = [mag2fluxcal_snana(*args) for args in zip(
            magpsf[id],
            sigmapsf[id])]
        flux, error = np.transpose(data)

        # make a Pandas DataFrame with exploded series
        pdf_id = [id] * len(flux)
        pdf = pd.DataFrame.from_dict({
            'SNID': [int(i) for i in pdf_id],
            'MJD': [int(i) for i in jd[id]],
            'FLUXCAL': flux,
            'FLUXCALERR': error,
            'FLT': pd.Series(fid[id]).replace({1: 'g', 2: 'r'})
        })

        # move to dataframe class
        pc = PredictLightCurve(pdf, object_id=pdf['SNID'][0])
        features = pc.predict_lc_coeff(num_pc_components=num_pc_components,
                                       min_flux_threshold=min_flux_threshold,
                                       bands=bands,
                                       band_choice='u')

        test_features.append(features)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # Take only probabilities to be KN
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)
