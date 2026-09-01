# Copyright 2019-2026 AstroLab Software
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
"""file contains scripts and definition for the Rubin SSO Fink Table"""

import os
import sys
import time
import datetime

from line_profiler import profile

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType, ArrayType, FloatType

from fink_utils.sso.spins import estimate_sso_params
from fink_utils.sso.spins import extract_obliquity
from fink_utils.sso.utils import rockify, extract_array_from_series
from fink_utils.sso.utils import compute_light_travel_correction
from fink_utils.sso.cleaning import dxy_cleaning, iterative_cleaning

from fink_science import __file__
from fink_science.tester import spark_unit_tests

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.coordinates import SkyCoord
import astropy.units as u

from asteroid_spinprops.ssolib import modelfit

import logging


_LOG = logging.getLogger(__name__)


def sanitize_dict(outdic):
    """Replace arrays with lists"""
    outdic2 = {}
    for k, v in outdic.items():
        if isinstance(v, np.ndarray):
            outdic2.update({k: list(v)})
        else:
            outdic2.update({k: v})
    return outdic2


@pandas_udf(ArrayType(FloatType()))
def randn(cmagpsf: pd.Series) -> pd.Series:
    """Construct column with random values from standard normal distribution"""
    rng = np.random.default_rng(seed=3)
    out = [
        rng.standard_normal(len(vec), dtype=np.float32) for vec in cmagpsf.to_numpy()
    ]
    return pd.Series(out)


@pandas_udf(StringType())
@profile
def extract_ssoft_parameters(
    designation: pd.Series,
    magpsf: pd.Series,
    sigmapsf: pd.Series,
    jd: pd.Series,
    band: pd.Series,
    raobs: pd.Series,
    decobs: pd.Series,
    raephem: pd.Series,
    decephem: pd.Series,
    ra_s: pd.Series,
    dec_s: pd.Series,
    phase: pd.Series,
    dobs: pd.Series,
    dhelio: pd.Series,
    cdx: pd.Series,
    cdy: pd.Series,
    method: pd.Series,
    model: pd.Series,
) -> pd.Series:
    """Extract phase and spin parameters from Fink alert data using Apache Spark

    Notes
    -----
    Only works for HG, HG1G2, and SHG1G2. Rotation period
    is not estimated here. For SOCCA, see <TBD>

    Parameters
    ----------
    designation: str
        SSO designation from Rubin alert packet
    magpsf: float
        Difference magnitude infered from psfFlux in Rubin
    sigmapsf: float
        Error estimate on magnitude
    jd: double
        Time of exposition (JD/UTC)
    band: str
        Filter name
    raobs: double
        Observation RAs
    decobs: double
        Observation declinations
    phase:
        Phase in degree
    dobs:
        Topocentric distance
    dhelio:
        Heliocentric distance
    cdx:
        Difference raobs, raephem
    cdy:
        Difference decobs, decephem
    method: str
        Method to compute ephemerides: `ephemcc` or `rest`.
        Use only the former on the Spark Cluster (local installation of ephemcc),
        otherwise use `rest` to call the ssodnet web service.
    model: str
        Model name. Available: HG, HG1G2, SHG1G2, SOCCA


    Returns
    -------
    out: pd.Series
        Series with dictionaries. Keys are parameter names (H, G, etc.)
        depending on the model chosen.
    """
    MODELS = {
        "HG": {"p0": [15.0, 0.15], "bounds": ([-3, 0], [30, 1])},
        "HG1G2": {"p0": [15.0, 0.15, 0.15], "bounds": ([-3, 0, 0], [30, 1, 1])},
        "SHG1G2": {
            "p0": [15.0, 0.15, 0.15, 0.8, np.pi, 0.0],
            "bounds": None,  # initialised inside fit_spin
        },
        "SOCCA": {
            "p0": None,  # Not used initially in SOCCA
            "bounds": None,  # initialised inside fit_spin
        },
    }

    model_name = model.to_numpy()[0]

    assert model_name in MODELS.keys(), "{} is not supported. Choose among: {}".format(
        model_name, str(MODELS.keys())
    )

    # loop over SSO
    out = []
    for index, _ in enumerate(designation.to_numpy()):
        # Construct the dataframe
        magpsf_red = extract_array_from_series(magpsf, index, float) - 5 * np.log10(
            extract_array_from_series(dobs, index, float)
            * extract_array_from_series(dhelio, index, float)
        )
        if model_name == "SOCCA":
            jd_lt = compute_light_travel_correction(
                extract_array_from_series(jd, index, float),
                extract_array_from_series(dobs, index, float),
            )
            pdf = pd.DataFrame(
                {
                    "cmred": magpsf_red,
                    "csigmapsf": extract_array_from_series(sigmapsf, index, float),
                    "Phase": extract_array_from_series(phase, index, float),
                    "cfid": extract_array_from_series(band, index, str),
                    "ra": extract_array_from_series(raobs, index, float),
                    "dec": extract_array_from_series(decobs, index, float),
                    "cjd": jd_lt,
                    "r:raephem": extract_array_from_series(raephem, index, float),
                    "r:decephem": extract_array_from_series(decephem, index, float),
                    "ra_s": extract_array_from_series(ra_s, index, float),
                    "dec_s": extract_array_from_series(dec_s, index, float),
                    "cdx": extract_array_from_series(cdx, index, float),
                    "cdy": extract_array_from_series(cdy, index, float),
                    "Dhelio": extract_array_from_series(dhelio, index, float),
                }
            )
            pdf = pdf.sort_values("cjd")

            # Clean data in-place
            pdf["dxy"] = np.sqrt(pdf["cdx"] ** 2 + pdf["cdy"] ** 2)
            pdf, _ = dxy_cleaning(
                pdf,
                pdf["dxy"],
                pdf["cmred"],
                threshold=0.95,
            )

            pdf, _ = iterative_cleaning(
                pdf,
                pdf["cmred"],
                pdf["csigmapsf"],
                pdf["Phase"],
                pdf["cfid"],
                pdf["ra"],
                pdf["dec"],
            )

            # Wrap columns inplace
            pdf_transposed = pd.DataFrame(
                {colname: [pdf[colname].to_numpy()] for colname in pdf.columns}
            )

            base_kwargs = dict(
                use_angles=True,
                use_filter_dependent=True,
                use_phase=True,
                use_shape=True,
            )

            current_kwargs = base_kwargs.copy()

            outdic = modelfit.get_fit_params(
                data=pdf_transposed,
                flavor=model_name,
                shg1g2_constrained=True,
                period_blind=True,
                pole_blind=False,
                period_in=None,
                period_quality_flag=True,
                terminator=True,
                time_me=True,
                remap=True,
                remap_kwargs=current_kwargs,
            )

            outdic = sanitize_dict(outdic)

            # replace names inplace for the remaning computation
            pdf = pdf.rename(
                columns={
                    "ra": "r:ra",
                    "dec": "r:dec",
                    "cband": "r:band",
                    "cjd": "r:jd",  # FIXME: this is lighttime corrected
                }
            )
        else:
            pdf = pd.DataFrame(
                {
                    "r:sigmapsf": extract_array_from_series(sigmapsf, index, float),
                    "r:jd": extract_array_from_series(jd, index, float),
                    "r:band": extract_array_from_series(band, index, str),
                    "r:ra": extract_array_from_series(raobs, index, float),
                    "r:dec": extract_array_from_series(decobs, index, float),
                    "r:raephem": extract_array_from_series(raephem, index, float),
                    "r:decephem": extract_array_from_series(decephem, index, float),
                    "r:magpsf_red": magpsf_red,
                    "Phase": extract_array_from_series(phase, index, float),
                    "Dobs": extract_array_from_series(dobs, index, float),
                }
            )

            pdf = pdf.sort_values("r:jd")

            outdic = estimate_sso_params(
                pdf["r:magpsf_red"].to_numpy(),
                pdf["r:sigmapsf"].to_numpy(),
                np.deg2rad(pdf["Phase"].to_numpy()),
                pdf["r:band"].to_numpy(),
                np.deg2rad(pdf["r:ra"].to_numpy()),
                np.deg2rad(pdf["r:dec"].to_numpy()),
                jd=pdf["r:jd"].to_numpy(),
                p0=MODELS[model_name]["p0"],
                bounds=MODELS[model_name]["bounds"],
                model=model_name,
                normalise_to_V=False,
                remap=False,
            )

        # Add astrometry
        fink_coord = SkyCoord(
            ra=pdf["r:ra"].to_numpy() * u.deg, dec=pdf["r:dec"].to_numpy() * u.deg
        )
        ephem_coord = SkyCoord(
            ra=pdf["r:raephem"].to_numpy() * u.deg,
            dec=pdf["r:decephem"].to_numpy() * u.deg,
        )

        separation = fink_coord.separation(ephem_coord).arcsecond

        outdic["mean_astrometry"] = np.mean(separation)
        outdic["std_astrometry"] = np.std(separation)
        outdic["skew_astrometry"] = skew(separation)
        outdic["kurt_astrometry"] = kurtosis(separation)

        # Time lapse
        outdic["n_days"] = pdf["r:jd"].max() - pdf["r:jd"].min()
        ufilters = np.unique(pdf["r:band"].to_numpy())
        for filt in ufilters:
            mask = pdf["r:band"].to_numpy() == filt
            outdic["n_days_{}".format(filt)] = (
                pdf["r:jd"][mask].max() - pdf["r:jd"][mask].min()
            )

        outdic["last_jd"] = pdf["r:jd"].max()

        out.append(str(outdic))
    return pd.Series(out)


def build_the_ssoft(
    aggregated_filename,
    nparts=400,
    nmin=10,
    frac=None,
    model="SHG1G2",
    version=None,
    sb_method="auto",
    ephem_method="ephemcc",
) -> pd.DataFrame:
    """Build the Fink Flat Table from scratch

    Parameters
    ----------
    aggregated_filename: str
        Aggregated data on HDFS.
    nparts: int, optional
        Number of Spark partitions to used. Default is 400.
        Rule of thumb is nparts = 4 * ncores
    nmin: int, optional
        Minimal number of measurements to select objects (all filters). Default is 50.
    frac: float, optional
        If specified, sample a fraction of the dataset (between 0 and 1). Default is None.
    model: str, optional
        Model name among HG, HG1G2, SHG1G2. Default is SHG1G2.
    version: str, optional
        Version number of the table. By default YYYY.MM.
    ephem_method: str
        Method to compute ephemerides: `ephemcc` (default), or `rest`.

    Notes
    -----
    Only HG is tested on early operations. Other models might work at your own risk.

    Returns
    -------
    pdf: pd.DataFrame
        Pandas DataFrame with all the SSOFT data.

    Examples
    --------
    >>> from fink_utils.sso.ssoft import get_ssoft_columns
    >>> COLUMNS, COLUMNS_HG, COLUMNS_HG1G2, COLUMNS_SHG1G2, COLUMNS_SOCCA = get_ssoft_columns('lsst')
    >>> ssoft_hg = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nparts=1,
    ...     nmin=10,
    ...     frac=None,
    ...     model='HG',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    >>> assert len(ssoft_hg) == 1, ssoft_hg
    >>> assert "G_g" in ssoft_hg.columns

    >>> col_ssoft_hg = sorted(ssoft_hg.columns)
    >>> expected_cols = sorted({**COLUMNS, **COLUMNS_HG}.keys())
    >>> assert col_ssoft_hg == expected_cols, (col_ssoft_hg, expected_cols)
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if version is None:
        now = datetime.datetime.now()
        version = "{}{:02d}".format(now.year, now.month)

    _LOG.info("Reading {} ephemerides".format(aggregated_filename))
    df_ztf = spark.read.format("parquet").load(aggregated_filename)

    _LOG.info("{:,} SSO objects in Fink".format(df_ztf.count()))

    df = df_ztf.withColumn("finkmeasurements", F.size(df_ztf["cra"])).filter(
        F.col("finkmeasurements") >= nmin
    )

    _LOG.info(
        "{:,} SSO objects with more than {} measurements".format(df.count(), nmin)
    )

    # Note: we compute the size of Phase
    # because Phase can be null due to no ephemerides
    df = (
        df.withColumn("ephemmeasurements", F.size(df["cphaseAngle"]))
        .filter(F.col("ephemmeasurements") >= nmin)
        .filter(F.size("cmagpsf") == F.size("cphaseAngle"))
        .repartition(nparts)
        .cache()
    )

    _LOG.info(
        "{:,} SSO objects with more than {} measurements and ephemerides".format(
            df.count(), nmin
        )
    )

    if frac is not None:
        if frac >= 1:
            _LOG.warning("`frac` should be between 0 and 1.")
            sys.exit()
        df = df.sample(fraction=frac, seed=0).cache()
        _LOG.info(
            "SAMPLE: {:,} SSO objects with more than {} measurements".format(
                df.count(), nmin
            )
        )

    # cdx, cdy only required for SOCCA
    if ("cephOffsetRa" not in df.columns) or ("cephOffsetDec" not in df.columns):
        _LOG.warning(
            "cephOffsetRa or cephOffsetDec not found in columns. Drawing from standard normal distribution"
        )
        df = df.withColumn("cephOffsetRa", randn("cmagpsf"))
        df = df.withColumn("cephOffsetDec", randn("cmagpsf"))

    cols = ["designation", "params_str"]
    t0 = time.time()
    pdf = (
        df.withColumn(
            "params_str",
            extract_ssoft_parameters(
                F.col("designation").astype("string"),
                "cmagpsf",
                "csigmapsf",
                "cjdUtc",
                "cband",
                "cra",
                "cdec",
                "cephRa",
                "cephDec",
                "chelioRa",
                "chelioDec",
                "cphaseAngle",
                "ctopoRange",
                "chelioRange",
                "cephOffsetRa",
                "cephOffsetDec",
                F.lit(ephem_method),
                F.lit(model),
            ),
        )
        .select(cols)
        .toPandas()
    )

    _LOG.info("Time to extract parameters: {:.2f} seconds".format(time.time() - t0))

    glob = globals()
    glob["nan"] = np.nan
    glob["inf"] = np.inf

    pdf["params_dict"] = pdf["params_str"].apply(lambda string: eval(string, glob))

    pdf = pd.concat([pdf, pd.json_normalize(pdf.params_dict)], axis=1).drop(
        columns=["params_dict", "params_str"]
    )

    sso_name, sso_number = rockify(pdf.designation.copy(), prune=False)
    pdf["sso_name"] = sso_name
    pdf["sso_number"] = sso_number

    if model == "SHG1G2":
        # compute obliquity
        pdf["obliquity"] = extract_obliquity(
            pdf.sso_name,
            pdf.alpha0,
            pdf.delta0,
        )

        # add flipped spins
        pdf["alpha0_alt"] = (pdf["alpha0"] + 180) % 360
        pdf["delta0_alt"] = -pdf["delta0"]

    pdf["version"] = version

    pdf["flag"] = 0

    return pdf


if __name__ == "__main__":
    """
    """
    globs = globals()
    path = os.path.dirname(__file__)

    aggregated_filename = (
        "file://{}/data/alerts/sso_rubin_lc_aggregated_202608_one_obj.parquet".format(
            path
        )
    )
    globs["aggregated_filename"] = aggregated_filename

    # Run the test suite
    spark_unit_tests(globs)
