# Copyright 2020-2026 AstroLab Software
# Author: Rupesh Durgesh
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
import os

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, MapType, StringType

from fink_science.rubin.hostless_detection.run_pipeline import (
    HostLessExtragalacticRubin,
)
from fink_science.ztf.hostless_detection.pipeline_utils import load_json
from fink_science.tester import spark_unit_tests

from fink_science import __file__


# Share configuration with ZTF
CONFIGS_BASE = load_json(
    "{}/rubin/hostless_detection/config_base.json".format(os.path.dirname(__file__))
)
CONFIGS = load_json(
    "{}/rubin/hostless_detection/config.json".format(os.path.dirname(__file__))
)
CONFIGS.update(CONFIGS_BASE)

ZP_NJY = 31.4

BAD_VALUES = ["Unknown", "Fail", "Fail 504", None, np.nan]


@F.pandas_udf(MapType(StringType(), FloatType()))
@profile
def run_potential_hostless(
    cutoutScience: pd.Series,
    cutoutTemplate: pd.Series,
    ssObjectId: pd.Series,
    nDiaSources: pd.Series,
    psfFlux: pd.Series,
    templateFlux: pd.Series,
    templateFluxErr: pd.Series,
    midpointMjdTai: pd.Series,
    firstDiaSourceMjdTaiFink: pd.Series,
    simbad_otype: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    mangrove_2MASS_name: pd.Series,
    mangrove_HyperLEDA_name: pd.Series,
    legacydr8_zphot: pd.Series,
    spicy_class: pd.Series,
) -> pd.Series:
    """Runs potential hostless candidate detection for Rubin without any filtering

    Parameters
    ----------
    cutoutScience: pd.Series
        science stamp images
    cutoutTemplate: pd.Series
        template stamp images
    ssObjectId: pd.Series
        SSO objectId. NaN if not existing.
    nDiaSources: pd.Series
        Number of previous detections
    psfFlux: pd.Series
        Flux in the difference image
    templateFlux: pd.Series,
        Flux in template image
    templateFluxErr: pd.Series
        Flux error in template image
    midpointMjdTai: pd.Series
        Emission date for the alert
    firstDiaSourceMjdTaiFink: pd.Series
        Date for the first time the alert was seen in Rubin
    simbad_otype: pd.Series
        SIMBAD type. NaN if not existing
    gaiadr3_DR3Name: pd.Series
        Name in Gaia DR3. NaN if not existing
    mangrove_2MASS_name: pd.Series
        Name in mangrove 2MASS. NaN if not existing
    mangrove_HyperLEDA_name: pd.Series
        Name in mangrove HyperLEDA. NaN if not existing
    legacydr8_zphot: pd.Series
        Photo-z estimate from Legacy Surveys DR8 South Photometric Redshifts catalog.
         NaN if not existing
    spicy_class
        closest source from SPICY catalog. Nan if not existing

    Notes
    -----
    Cuts are applied before running the pipeline. Process if:
    - Number of detections >= 3
    - mag < 20
    - first alert earlier than 30 days ago
    - not in MPC, SIMBAD, Gaia DR3
    Otherwise, the returned values for kstest are nulls.

    Returns
    -------
    pd.Series
        Scores (array of 2 floats) for being hostless

    References
    ----------
    1. ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
     Transients
        https://arxiv.org/abs/2404.18165

    Examples
    --------
    >>> df = spark.read.format('parquet').load(rubin_alert_sample)
    >>> df.count()
    25

    # No cuts
    >>> df = df.withColumn('elephant_kstest_no_cuts',
    ...     run_potential_hostless(
    ...         df["cutoutScience"],
    ...         df["cutoutTemplate"],
    ...         F.lit(None),
    ...         F.lit(3),
    ...         F.lit(100000000),
    ...         F.lit(100000000),
    ...         F.lit(100),
    ...         df["diaSource.midpointMjdTai"],
    ...         df["diaSource.midpointMjdTai"],
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),))
    >>> df.filter(df.elephant_kstest_no_cuts.kstest_science.isNotNull()).count()
    0

    # SSO cuts
    >>> df = df.withColumn('elephant_kstest_sso_cuts',
    ...     run_potential_hostless(
    ...         df["cutoutScience"],
    ...         df["cutoutTemplate"],
    ...         df["ssSource.ssObjectId"],
    ...         F.lit(3),
    ...         F.lit(100000000),
    ...         F.lit(100000000),
    ...         F.lit(100),
    ...         df["diaSource.midpointMjdTai"],
    ...         df["diaSource.midpointMjdTai"],
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),
    ...         F.lit(None),))
    >>> df.filter(df.elephant_kstest_sso_cuts.kstest_science.isNotNull()).count()
    0

    # All cuts
    >>> df = df.withColumn('elephant_kstest',
    ...     run_potential_hostless(
    ...         df["cutoutScience"],
    ...         df["cutoutTemplate"],
    ...         df["ssSource.ssObjectId"],
    ...         df["diaObject.nDiaSources"],
    ...         df["diaSource.psfFlux"],
    ...         df["diaSource.templateFlux"],
    ...         df["diaSource.templateFluxErr"],
    ...         df["diaSource.midpointMjdTai"],
    ...         F.array_min("prvDiaSources.midpointMjdTai"),
    ...         F.lit("star"),
    ...         F.lit("DR3 toto"),
    ...         F.lit("galaxy"),
    ...         F.lit("galaxy"),
    ...         F.lit("galaxy"),
    ...         F.lit("galaxy"),))
    >>> df.filter(df.elephant_kstest.kstest_science.isNotNull()).count()
    0
    """
    f_min_point = nDiaSources >= CONFIGS["minimum_number_of_alerts"] - 1  # N + 1
    # FIXME: put the conversion formula in fink-utils
    f_bright = (-2.5 * np.log10(psfFlux) + ZP_NJY) < CONFIGS["cutout_magnitude"]
    f_not_in_simbad = simbad_otype.apply(lambda val: val in BAD_VALUES or pd.isna(val))
    f_not_in_gaia = gaiadr3_DR3Name.apply(lambda val: val in BAD_VALUES or pd.isna(val))
    f_not_in_2mass = mangrove_2MASS_name.apply(
        lambda val: val in BAD_VALUES or pd.isna(val)
    )
    f_not_in_hyperlda = mangrove_HyperLEDA_name.apply(
        lambda val: val in BAD_VALUES or pd.isna(val)
    )
    f_not_in_legacydr8 = legacydr8_zphot.apply(
        lambda val: val in BAD_VALUES or pd.isna(val)
    )
    f_not_in_spicy = spicy_class.apply(lambda val: val in BAD_VALUES or pd.isna(val))

    f_not_sso = ssObjectId.apply(lambda val: val in BAD_VALUES or pd.isna(val))
    f_early = (midpointMjdTai - firstDiaSourceMjdTaiFink) < CONFIGS["cutout_timeframe"]

    # Cuts on psfFlux and templateFlux
    PSF_FLUX_THRESHOLD = 0
    TEMPLATE_FLUX_THRESHOLD = [300, 2000]
    psfFlux_cut = psfFlux > PSF_FLUX_THRESHOLD
    templateFlux_cut = (templateFlux > TEMPLATE_FLUX_THRESHOLD[0]) & (
        templateFlux < TEMPLATE_FLUX_THRESHOLD[1]
    )
    # SNR cut > 5
    template_snr_cut = (templateFlux / templateFluxErr) > 5
    good_candidate = (
        f_min_point
        & f_bright
        & f_not_in_simbad
        & f_not_in_gaia
        & f_not_sso
        & f_early
        & psfFlux_cut
        & templateFlux_cut
        & template_snr_cut
        & f_not_in_2mass
        & f_not_in_hyperlda
        & f_not_in_legacydr8
        & f_not_in_spicy
    )
    # Process full image
    default_result = {
        "kstest_science": None,
        "kstest_template": None,
    }
    kstest_results = []

    hostless_science_class = HostLessExtragalacticRubin(CONFIGS_BASE)
    for index in range(cutoutScience.shape[0]):
        if good_candidate[index]:
            science_stamp = cutoutScience[index]
            template_stamp = cutoutTemplate[index]
            kstest_science, kstest_template = (
                hostless_science_class.process_candidate_fink_rubin(
                    science_stamp, template_stamp
                )
            )
            kstest_results.append({
                "kstest_science": kstest_science,
                "kstest_template": kstest_template,
            })
        else:
            kstest_results.append(default_result)
    return pd.Series(kstest_results)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    rubin_alert_sample = "file://{}/data/alerts/hostless_detection/rubin_sample_data_10_0.parquet".format(
        path
    )
    globs["rubin_alert_sample"] = rubin_alert_sample
    spark_unit_tests(globs)
