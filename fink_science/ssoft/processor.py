# Copyright 2023-2024 AstroLab Software
# Author: Julien Peloton
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
""" This file contains scripts and definition for the SSO Fink Table
"""
import io
import re
import sys
import time
import requests
import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import MapType, FloatType, StringType

from fink_utils.sso.utils import get_miriade_data
from fink_utils.sso.spins import estimate_sso_params
from fink_utils.sso.periods import estimate_synodic_period

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.coordinates import SkyCoord
import astropy.units as u

import rocks

COLUMNS = {
    'ssnamenr': {'type': 'str', 'description': 'Designation (name or number) of the object from MPC archive as given by ZTF'},
    'sso_name': {'type': 'str', 'description': 'Official name or provisional designation of the SSO'},
    'sso_number': {'type': 'int', 'description': 'IAU number of the SSO'},
    'last_jd': {'type': 'double', 'description': 'Julian Date for the last detection in Fink, in UTC'},
    'H_1': {'type': 'double', 'description': 'Absolute magnitude for the ZTF filter band g'},
    'H_2': {'type': 'double', 'description': 'Absolute magnitude for the ZTF filter band r'},
    'err_H_1': {'type': 'double', 'description': 'Uncertainty on the absolute magnitude for the ZTF filter band g'},
    'err_H_2': {'type': 'double', 'description': 'Uncertainty on the absolute magnitude for the ZTF filter band r'},
    'min_phase': {'type': 'double', 'description': 'Minimum phase angle of the observations used to compute the phase function, in degree'},
    'min_phase_1': {'type': 'double', 'description': 'Minimum phase angle of the observations used to compute the phase function for the ZTF filter band g, in degree'},
    'min_phase_2': {'type': 'double', 'description': 'Minimum phase angle of the observations used to compute the phase function for the ZTF filter band r, in degree'},
    'max_phase': {'type': 'double', 'description': 'Maximum phase angle of the observations used to compute the phase function, in degree'},
    'max_phase_1': {'type': 'double', 'description': 'Maximum phase angle of the observations used to compute the phase function for the ZTF filter band g, in degree'},
    'max_phase_2': {'type': 'double', 'description': 'Maximum phase angle of the observations used to compute the phase function for the ZTF filter band r, in degree'},
    'chi2red': {'type': 'double', 'description': 'Reduced chi-square of the fit'},
    'rms': {'type': 'double', 'description': 'RMS of the fit, in magnitude'},
    'rms_1': {'type': 'double', 'description': 'RMS of the fit for the filter band g, in magnitude'},
    'rms_2': {'type': 'double', 'description': 'RMS of the fit for the filter band r, in magnitude'},
    'median_error_phot': {'type': 'double', 'description': 'Median of the 1-sigma uncertainty on the measurements, in magnitude'},
    'median_error_phot_1': {'type': 'double', 'description': 'Median of the 1-sigma uncertainty on the measurements for the filter band g, in magnitude'},
    'median_error_phot_2': {'type': 'double', 'description': 'Median of the 1-sigma uncertainty on the measurements for the filter band r, in magnitude'},
    'mean_astrometry': {'type': 'double', 'description': 'Astrometry: mean of the angular separation between observations and ephemerides, in arcsecond'},
    'std_astrometry': {'type': 'double', 'description': 'Astrometry: standard deviation of the angular separation between observations and ephemerides, in arcsecond'},
    'skew_astrometry': {'type': 'double', 'description': 'Astrometry: skewness of the angular separation between observations and ephemerides'},
    'kurt_astrometry': {'type': 'double', 'description': 'Astrometry: kurtosis of the angular separation between observations and ephemerides'},
    'synodic_period': {'type': 'double', 'description': 'Synodic period estimated, in hour'},
    'synodic_period_chi2red': {'type': 'double', 'description': 'Reduced chi-square for the period estimation'},
    'n_obs': {'type': 'int', 'description': 'Number of observations in Fink'},
    'n_obs_1': {'type': 'int', 'description': 'Number of observations for the ZTF filter band g in Fink'},
    'n_obs_2': {'type': 'int', 'description': 'Number of observations for the ZTF filter band r in Fink'},
    'n_days': {'type': 'int', 'description': 'Number of days between the first and the last observations in Fink'},
    'n_days_1': {'type': 'int', 'description': 'Number of days between the first and the last observations in Fink, for the ZTF filter band g'},
    'n_days_2': {'type': 'int', 'description': 'Number of days between the first and the last observations in Fink, for the ZTF filter band r'},
    'fit': {'type': 'int', 'description': 'Code to assess the quality of the fit: 0: success, 1: bad_vals, 2: MiriadeFail, 3: RunTimError, 4: LinalgError'},
    'status': {'type': 'int', 'description': 'Code for quality `status` (least square convergence): -2: failure, -1 : improper input parameters status returned from MINPACK, 0 : the maximum number of function evaluations is exceeded, 1 : gtol termination condition is satisfied, 2 : ftol termination condition is satisfied, 3 : xtol termination condition is satisfied, 4 : Both ftol and xtol termination conditions are satisfied.'},
    'flag': {'type': 'int', 'description': 'TBD'},
    'version': {'type': 'str', 'description': 'Version of the SSOFT YYYY.MM'},
}

COLUMNS_SHG1G2 = {
    'G1_1': {'type': 'double', 'description': 'G1 phase parameter for the ZTF filter band g'},
    'G1_2': {'type': 'double', 'description': 'G1 phase parameter for the ZTF filter band r'},
    'G2_1': {'type': 'double', 'description': 'G2 phase parameter for the ZTF filter band g'},
    'G2_2': {'type': 'double', 'description': 'G2 phase parameter for the ZTF filter band r'},
    'R': {'type': 'double', 'description': 'Oblateness of the object'},
    'alpha0': {'type': 'double', 'description': 'Right ascension of the spin axis (EQJ2000), in degree'},
    'delta0': {'type': 'double', 'description': 'Declination of the spin axis (EQJ2000), in degree'},
    'alpha0_alt': {'type': 'double', 'description': 'Flipped `alpha0`: (`alpha0` + 180) modulo 360, in degree'},
    'delta0_alt': {'type': 'double', 'description': 'Flipped `delta0`: -`delta0`, in degree'},
    'obliquity': {'type': 'double', 'description': 'Obliquity of the spin axis, in degree'},
    'err_G1_1': {'type': 'double', 'description': 'Uncertainty on the G1 phase parameter for the ZTF filter band g'},
    'err_G1_2': {'type': 'double', 'description': 'Uncertainty on the G1 phase parameter for the ZTF filter band r'},
    'err_G2_1': {'type': 'double', 'description': 'Uncertainty on the G2 phase parameter for the ZTF filter band g'},
    'err_G2_2': {'type': 'double', 'description': 'Uncertainty on the G2 phase parameter for the ZTF filter band r'},
    'err_R': {'type': 'double', 'description': 'Uncertainty on the oblateness'},
    'err_alpha0': {'type': 'double', 'description': 'Uncertainty on the right ascension of the spin axis (EQJ2000), in degree'},
    'err_delta0': {'type': 'double', 'description': 'Uncertainty on the declination of the spin axis (EQJ2000), in degree'},
    'max_cos_lambda': {'type': 'double', 'description': 'Maximum of the absolute value of the cosine for the aspect angle'},
    'mean_cos_lambda': {'type': 'double', 'description': 'Mean of the absolute value of the cosine for the aspect angle'},
    'min_cos_lambda': {'type': 'double', 'description': 'Minimum of the absolute value of the cosine for the aspect angle'},
}

COLUMNS_HG1G2 = {
    'G1_1': {'type': 'double', 'description': 'G1 phase parameter for the ZTF filter band g'},
    'G1_2': {'type': 'double', 'description': 'G1 phase parameter for the ZTF filter band r'},
    'G2_1': {'type': 'double', 'description': 'G2 phase parameter for the ZTF filter band g'},
    'G2_2': {'type': 'double', 'description': 'G2 phase parameter for the ZTF filter band r'},
    'err_G1_1': {'type': 'double', 'description': 'Uncertainty on the G1 phase parameter for the ZTF filter band g'},
    'err_G1_2': {'type': 'double', 'description': 'Uncertainty on the G1 phase parameter for the ZTF filter band r'},
    'err_G2_1': {'type': 'double', 'description': 'Uncertainty on the G2 phase parameter for the ZTF filter band g'},
    'err_G2_2': {'type': 'double', 'description': 'Uncertainty on the G2 phase parameter for the ZTF filter band r'},
}

COLUMNS_HG = {
    'G_1': {'type': 'double', 'description': 'G phase parameter for the ZTF filter band g'},
    'G_2': {'type': 'double', 'description': 'G phase parameter for the ZTF filter band r'},
    'err_G_1': {'type': 'double', 'description': 'Uncertainty on the G phase parameter for the ZTF filter band g'},
    'err_G_2': {'type': 'double', 'description': 'Uncertainty on the G phase parameter for the ZTF filter band r'},
}

def remove_leading_zeros(val):
    """ Iteratively remove leading zeros from a string

    Parameters
    ----------
    val: str
        A string

    Returns
    ----------
    The input string with leading zeros removed

    Examples
    ----------
    >>> string = '0abcd'
    >>> remove_leading_zeros(string)
    'abcd'

    >>> string = '000000a0bcd'
    >>> remove_leading_zeros(string)
    'a0bcd'

    >>> string = 'toto'
    >>> remove_leading_zeros(string)
    'toto'
    """
    if val.startswith('0'):
        return remove_leading_zeros(val[1:])
    else:
        return val

def process_regex(regex, data):
    """ Extract parameters from a regex given the data

    Parameters
    ----------
    regex: str
        Regular expression to use
    data: str
        Data entered by the user

    Returns
    ----------
    parameters: dict or None
        Parameters (key: value) extracted from the data
    """
    template = re.compile(regex)
    m = template.match(data)
    if m is None:
        return None

    parameters = m.groupdict()
    return parameters

@pandas_udf(MapType(StringType(), FloatType()), PandasUDFType.SCALAR)
def estimate_sso_params_spark(ssnamenr, magpsf, sigmapsf, jd, fid, ra, dec, method, model, sb_method):
    """ Extract phase and spin parameters from Fink alert data using Apache Spark

    Parameters
    ----------
    ssnamenr: str
        SSO name from ZTF alert packet
    magpsf: float
        Magnitude from ZTF
    sigmapsf: float
        Error estimate on magnitude
    jd: double
        Time of exposition (UTC)
    fid: int
        Filter ID (1=g, 2=r)
    ra: double
        RA
    dec: double
        Declination
    method: str
        Method to compute ephemerides: `ephemcc` or `rest`.
        Use only the former on the Spark Cluster (local installation of ephemcc),
        otherwise use `rest` to call the ssodnet web service.
    model: str
        Model name. Available: HG, HG1G2, SHG1G2
    sb_method: str
        Specify the single-band lomb scargle implementation to use.
        See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargleMultiband.html#astropy.timeseries.LombScargleMultiband.autopower
        If nifty-ls is installed, one can also specify fastnifty. Although
        in this case it does not work yet for Nterms_* higher than 1.

    Returns
    ----------
    out: pd.Series
        Series with dictionaries. Keys are parameter names (H, G, etc.)
        depending on the model chosen.
    """
    MODELS = {
        'HG': {
            'p0': [15.0, 0.15],
            'bounds': ([0, 0], [30, 1])
        },
        'HG1G2': {
            'p0': [15.0, 0.15, 0.15],
            'bounds': ([0, 0, 0], [30, 1, 1])
        },
        'SHG1G2': {
            'p0': [15.0, 0.15, 0.15, 0.8, np.pi, 0.0],
            'bounds': ([0, 0, 0, 3e-1, 0, -np.pi / 2], [30, 1, 1, 1, 2 * np.pi, np.pi / 2])
        },
    }

    # loop over SSO
    out = []
    for index, ssname in enumerate(ssnamenr.values):

        # First get ephemerides data
        pdf_sso = pd.DataFrame(
            {
                'i:ssnamenr': [ssname] * len(ra.values[index]),
                'i:magpsf': magpsf.values[index].astype(float),
                'i:sigmapsf': sigmapsf.values[index].astype(float),
                'i:jd': jd.values[index].astype(float),
                'i:fid': fid.values[index].astype(int),
                'i:ra': ra.values[index].astype(float),
                'i:dec': dec.values[index].astype(float)
            }
        )

        pdf_sso = pdf_sso.sort_values('i:jd')

        if method.values[0] == 'ephemcc':
            # hardcoded for the Spark cluster!
            parameters = {
                'outdir': '/tmp/ramdisk/spins',
                'runner_path': '/tmp/fink_run_ephemcc4.sh',
                'userconf': '/tmp/.eproc-4.3',
                'iofile': '/tmp/default-ephemcc-observation.xml'
            }
        elif method.values[0] == 'rest':
            parameters = {}
        pdf = get_miriade_data(
            pdf_sso,
            observer='I41',
            rplane='1',
            tcoor=5,
            withecl=False,
            method=method.values[0],
            parameters=parameters
        )

        if 'i:magpsf_red' not in pdf.columns:
            out.append({'fit': 2, 'status': -2})
        else:

            if model.values[0] == 'SHG1G2':
                outdic = estimate_sso_params(
                    pdf['i:magpsf_red'].values,
                    pdf['i:sigmapsf'].values,
                    np.deg2rad(pdf['Phase'].values),
                    pdf['i:fid'].values,
                    np.deg2rad(pdf['i:ra'].values),
                    np.deg2rad(pdf['i:dec'].values),
                    p0=MODELS[model.values[0]]['p0'],
                    bounds=MODELS[model.values[0]]['bounds'],
                    model=model.values[0],
                    normalise_to_V=False
                )
            else:
                outdic = estimate_sso_params(
                    pdf['i:magpsf_red'].values,
                    pdf['i:sigmapsf'].values,
                    np.deg2rad(pdf['Phase'].values),
                    pdf['i:fid'].values,
                    p0=MODELS[model.values[0]]['p0'],
                    bounds=MODELS[model.values[0]]['bounds'],
                    model=model.values[0],
                    normalise_to_V=False
                )

            filts = np.unique(pdf['i:fid'].values)
            is_fit_ok = np.all(["H_{}".format(filt) in outdic for filt in filts])
            if is_fit_ok:
                # Add synodic period estimation
                period, chi2red_period = estimate_synodic_period(
                    pdf=pdf,
                    phyparam=outdic,
                    flavor=method.values[0],
                    sb_method=sb_method.values[0],
                    Nterms_base=1,
                    Nterms_band=1,
                    period_range=(1. / 24., 30.)  # 1h to 1 month
                )

                outdic["synodic_period"] = period
                outdic["synodic_period_chi2red"] = chi2red_period
            else:
                outdic["synodic_period"] = np.nan
                outdic["synodic_period_chi2red"] = np.nan

            # Add astrometry
            fink_coord = SkyCoord(ra=pdf['i:ra'].values * u.deg, dec=pdf['i:dec'].values * u.deg)
            ephem_coord = SkyCoord(ra=pdf['RA'].values * u.deg, dec=pdf['Dec'].values * u.deg)

            separation = fink_coord.separation(ephem_coord).arcsecond

            outdic['mean_astrometry'] = np.mean(separation)
            outdic['std_astrometry'] = np.std(separation)
            outdic['skew_astrometry'] = skew(separation)
            outdic['kurt_astrometry'] = kurtosis(separation)

            # Time lapse
            outdic['n_days'] = pdf['i:jd'].max() - pdf['i:jd'].min()
            ufilters = np.unique(pdf['i:fid'].values)
            for filt in ufilters:
                mask = pdf['i:fid'].values == filt
                outdic['n_days_{}'.format(filt)] = pdf['i:jd'][mask].max() - pdf['i:jd'][mask].min()

            outdic['last_jd'] = pdf['i:jd'].max()

            out.append(outdic)
    return pd.Series(out)

def correct_ztf_mpc_names(ssnamenr):
    """ Remove trailing 0 at the end of SSO names from ZTF

    e.g. 2010XY03 should read 2010XY3

    Parameters
    ----------
    ssnamenr: np.array
        Array with SSO names from ZTF

    Returns
    ----------
    out: np.array
        Array with corrected names from ZTF

    Examples
    ----------
    >>> ssnamenr = np.array(['2010XY03', '2023AB0', '2023XY00', '345', '2023UY12'])
    >>> ssnamenr_alt = correct_ztf_mpc_names(ssnamenr)

    # the first ones changed
    >>> assert ssnamenr_alt[0] == '2010XY3'
    >>> assert ssnamenr_alt[1] == '2023AB'
    >>> assert ssnamenr_alt[2] == '2023XY'

    >>> assert np.all(ssnamenr_alt[3:] == ssnamenr[3:])
    """
    # remove numbered
    regex = "^\d+$"  # noqa: W605
    template = re.compile(regex)
    unnumbered = np.array([template.findall(str(x)) == [] for x in ssnamenr])

    # Extract names
    regex = "(?P<year>\d{4})(?P<letter>\w{2})(?P<end>\d+)$"  # noqa: W605
    processed = [process_regex(regex, x) for x in ssnamenr[unnumbered]]

    def f(x, y):
        """ Correct for trailing 0 in SSO names

        Parameters
        ----------
        x: dict, or None
            Data extracted from the regex
        y: str
            Corresponding ssnamenr

        Returns
        ----------
        out: str
            Name corrected for trailing 0 at the end (e.g. 2010XY03 should read 2010XY3)
        """
        if x is None:
            return y
        else:
            return "{}{}{}".format(x['year'], x['letter'], remove_leading_zeros(x['end']))

    corrected = np.array([f(x, y) for x, y in zip(processed, ssnamenr[unnumbered])])

    ssnamenr[unnumbered] = corrected

    return ssnamenr

def rockify(ssnamenr: pd.Series):
    """ Extract names and numbers from ssnamenr

    Parameters
    ----------
    ssnamenr: pd.Series of str
        SSO names as given in ZTF alert packets

    Returns
    -----------
    sso_name: np.array of str
        SSO names according to quaero
    sso_number: np.array of int
        SSO numbers according to quaero
    """
    # prune names
    ssnamenr_alt = correct_ztf_mpc_names(ssnamenr.values)

    # rockify
    names_numbers = rocks.identify(ssnamenr_alt)

    sso_name = np.transpose(names_numbers)[0]
    sso_number = np.transpose(names_numbers)[1]

    return sso_name, sso_number

def angular_separation(lon1, lat1, lon2, lat2):
    """
    Angular separation between two points on a sphere.
    Stolen from astropy -- for version <5

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : `~astropy.coordinates.Angle`, `~astropy.units.Quantity` or float
        Longitude and latitude of the two points. Quantities should be in
        angular units; floats in radians.

    Returns
    -------
    angular separation : `~astropy.units.Quantity` ['angle'] or float
        Type depends on input; ``Quantity`` in angular units, or float in
        radians.

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
    """
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.arctan2(np.hypot(num1, num2), denominator)

def extract_obliquity(sso_name, alpha0, delta0, bft_filename=None):
    """ Extract obliquity using spin values, and the BFT information

    Parameters
    ----------
    sso_name: np.array or pd.Series of str
        SSO names according to quaero (see `rockify`)
    alpha0: np.array or pd.Series of double
        RA of the pole [degree]
    delta0: np.array or pd.Series of double
        DEC of the pole [degree]
    bft_filename: str, optional
        If given, read the BFT (parquet format). If not specified,
        download the latest version of the BFT (can be long).
        Default is None (unspecified).

    Returns
    ----------
    obliquity: np.array of double
        Obliquity for each object [degree]
    """
    cols = ['sso_name', 'orbital_elements.node_longitude.value', 'orbital_elements.inclination.value']
    if bft_filename is None:
        print('BFT not found -- downloading...')
        r = requests.get('https://ssp.imcce.fr/data/ssoBFT-latest_Asteroid.parquet')
        pdf_bft = pd.read_parquet(io.BytesIO(r.content), columns=cols)
    else:
        pdf_bft = pd.read_parquet(bft_filename, columns=cols)

    sub = pdf_bft[cols]

    pdf = pd.DataFrame(
        {
            'sso_name': sso_name,
            'alpha0': alpha0,
            'delta0': delta0
        }
    )

    pdf = pdf.merge(sub[cols], left_on='sso_name', right_on='sso_name', how='left')

    # Orbit
    lon_orbit = (pdf['orbital_elements.node_longitude.value'] - 90).values
    lat_orbit = (90. - pdf['orbital_elements.inclination.value']).values

    # Spin -- convert to EC
    ra = np.nan_to_num(pdf.alpha0.values) * u.degree
    dec = np.nan_to_num(pdf.delta0.values) * u.degree

    # Trick to put the object "far enough"
    coords_spin = SkyCoord(ra=ra, dec=dec, distance=200 * u.parsec, frame='hcrs')

    # in radian
    lon_spin = coords_spin.heliocentricmeanecliptic.lon.value
    lat_spin = coords_spin.heliocentricmeanecliptic.lat.value

    obliquity = np.degrees(
        angular_separation(
            np.radians(lon_spin),
            np.radians(lat_spin),
            np.radians(lon_orbit),
            np.radians(lat_orbit)
        )
    )

    return obliquity

def aggregate_sso_data(output_filename=None):
    """ Aggregate all SSO data in Fink

    Data is read from HDFS on VirtualData

    Parameters
    ----------
    output_filename: str, optional
        If given, save data on HDFS. Cannot overwrite. Default is None.

    Returns
    ----------
    df_grouped: Spark DataFrame
        Spark DataFrame with aggregated SSO data.
    """
    spark = SparkSession.builder.getOrCreate()
    cols0 = ['candidate.ssnamenr']
    cols = [
        'candidate.ra',
        'candidate.dec',
        'candidate.magpsf',
        'candidate.sigmapsf',
        'candidate.fid',
        'candidate.jd'
    ]

    df = spark.read.format('parquet').option('basePath', 'archive/science').load('archive/science')
    df_agg = df.select(cols0 + cols)\
        .filter(F.col('roid') == 3)\
        .groupBy('ssnamenr')\
        .agg(*[F.collect_list(col.split('.')[1]).alias('c' + col.split('.')[1]) for col in cols])

    if output_filename is not None:
        df_agg.write.parquet(output_filename)

    return df_agg

def build_the_ssoft(aggregated_filename=None, bft_filename=None, nproc=80, nmin=50, frac=None, model='SHG1G2', version=None, sb_method="auto") -> pd.DataFrame:
    """ Build the Fink Flat Table from scratch

    Parameters
    ----------
    aggregated_filename: str, optional
        If given, read aggregated data on HDFS. Default is None.
    bft_filename: str, optional
        If given, read the BFT (parquet format). If not specified,
        download the latest version of the BFT (can be long).
        Default is None (unspecified).
    nproc: int, optional
        Number of cores to used. Default is 80.
    nmin: int, optional
        Minimal number of measurements to select objects (all filters). Default is 50.
    frac: float, optional
        If specified, sample a fraction of the dataset (between 0 and 1). Default is None.
    model: str, optional
        Model name among HG, HG1G2, SHG1G2. Default is SHG1G2.
    version: str, optional
        Version number of the table. By default YYYY.MM.
    sb_method: str
        Specify the single-band lomb scargle implementation to use.
        See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargleMultiband.html#astropy.timeseries.LombScargleMultiband.autopower
        If nifty-ls is installed, one can also specify fastnifty. Although
        in this case it does not work yet for Nterms_* higher than 1.

    Returns
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame with all the SSOFT data.
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if version is None:
        now = datetime.datetime.now()
        version = '{}.{:02d}'.format(now.year, now.month)

    if aggregated_filename is None:
        print('Reconstructing SSO data...')
        t0 = time.time()
        aggregated_filename = 'sso_aggregated_{}'.format(version)
        aggregate_sso_data(output_filename=aggregated_filename)
        print('Time to reconstruct SSO data: {:.2f} seconds'.format(time.time() - t0))
    df_ztf = spark.read.format('parquet').load(aggregated_filename)

    print('{:,} SSO objects in Fink'.format(df_ztf.count()))

    df = df_ztf\
        .withColumn('nmeasurements', F.size(df_ztf['cra']))\
        .filter(F.col('nmeasurements') >= nmin)\
        .repartition(nproc)\
        .cache()

    print('{:,} SSO objects with more than {} measurements'.format(df.count(), nmin))

    if frac is not None:
        if frac >= 1:
            print('`frac` should be between 0 and 1.')
            sys.exit()
        df = df.sample(fraction=frac, seed=0).cache()
        print('SAMPLE: {:,} SSO objects with more than {} measurements'.format(df.count(), nmin))

    cols = ['ssnamenr', 'params']
    t0 = time.time()
    pdf = df\
        .withColumn(
            'params',
            estimate_sso_params_spark(
                F.col('ssnamenr').astype('string'),
                'cmagpsf',
                'csigmapsf',
                'cjd',
                'cfid',
                'cra',
                'cdec',
                F.lit('ephemcc'),
                F.lit(model),
                F.lit(sb_method)
            )
        ).select(cols).toPandas()

    print('Time to extract parameters: {:.2f} seconds'.format(time.time() - t0))

    pdf = pd.concat([pdf, pd.json_normalize(pdf.params)], axis=1).drop('params', axis=1)

    sso_name, sso_number = rockify(pdf.ssnamenr.copy())
    pdf['sso_name'] = sso_name
    pdf['sso_number'] = sso_number

    if model == 'SHG1G2':
        # compute obliquity
        pdf['obliquity'] = extract_obliquity(
            pdf.sso_name,
            pdf.alpha0,
            pdf.delta0,
            bft_filename=bft_filename
        )

        # add flipped spins
        pdf['alpha0_alt'] = (pdf['alpha0'] + 180) % 360
        pdf['delta0_alt'] = - pdf['delta0']

    pdf['version'] = version

    pdf['flag'] = 0

    return pdf


if __name__ == "__main__":
    """
    """
    import doctest
    sys.exit(doctest.testmod()[0])
