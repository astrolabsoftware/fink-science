from line_profiler import profile

import logging
import pandas as pd
import numpy as np
import light_curve as lc

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, IntegerType, MapType, StructType, StructField


logger = logging.getLogger(__name__)

# Mapping LSST filter bands to integers for consistency with ZTF structure
# ZTF used 1:g, 2:r (3:i ??).
LSST_FILTER_MAP = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}


def create_extractor():
    """
    Features definition - identical to ZTF implementation for consistency.

    Reference: https://arxiv.org/pdf/2012.01419.pdf#section.A1
    Note: Features applied on Flux (nJy) instead of Magnitude.
    """
    return lc.Extractor(
        lc.Mean(),  # A1.0.2  Mean
        lc.WeightedMean(),  # A1.0.3  Weighted Mean
        lc.StandardDeviation(),  # A1.0.4  Standard Deviation
        lc.Median(),  # A1.0.7  Median
        lc.Amplitude(),  # A2.0.1  Amplitude
        lc.BeyondNStd(nstd=1),  # A2.0.2  Beyond n Std
        lc.Cusum(),  # A2.0.3  Cusum
        lc.InterPercentileRange(quantile=0.1),  # A2.0.6  Inter-percentile range
        lc.Kurtosis(),  # A2.0.7  Kurtosis
        lc.LinearTrend(),  # A2.0.8  Linear Trend
        lc.LinearFit(),  # A2.0.9  Linear Fit
        lc.MagnitudePercentageRatio(  # A2.0.10  Magnitude Percentage Ratio
            quantile_numerator=0.4,
            quantile_denominator=0.05,
        ),
        lc.MagnitudePercentageRatio(
            quantile_numerator=0.2,
            quantile_denominator=0.1,
        ),
        lc.MaximumSlope(),  # A2.0.11  Maximum Slope
        lc.MedianAbsoluteDeviation(),  # A2.0.13  Median Absolute Deviation
        lc.MedianBufferRangePercentage(
            quantile=0.1
        ),  # A2.0.14  Median Buffer Range Percentage
        lc.PercentAmplitude(),  # A2.0.15  Percent Amplitude
        lc.MeanVariance(),
        lc.AndersonDarlingNormal(),
        lc.ReducedChi2(),  # A2.0.25  Reduced Chi2
        lc.Skew(),  # A2.0.26  Skew
        lc.StetsonK(),  # A2.0.28  Stetson K
    )


# 'lc.Extractor' cannot be pickled, so we recreate it needed,
# but we define column names globally.
FEATURES_COLS = create_extractor().names


@profile
def extract_features_ad_rubin_raw(
    midpointMjdTai, psfFlux, psfFluxErr, band, objectId
) -> pd.Series:
    """Returns features extracted from measurements using light_curve package for LSST alerts.

    Parameters
    ----------
    midpointMjdTai: Spark DataFrame Column
        MJD TAI times (vectors of floats)
    psfFlux: Spark DataFrame Column
        PSF Flux in nJy (vectors of floats)
    psfFluxErr: Spark DataFrame Column
        PSF Flux error in nJy (vectors of floats)
    band: Spark DataFrame Column
        Filter bands (vectors of strings: 'u', 'g', 'r', 'i', 'z', 'y')
    objectId: Spark DataFrame Column
        Object IDs (vectors of str or int)

    Returns
    -------
    out: dict
        Returns dict of dict.
        Outer keys: filters (int mapped from chars),
        Inner keys: names of features.
    """
    midpointMjdTai = np.asarray(midpointMjdTai, dtype=np.float64)
    psfFlux = np.asarray(psfFlux, dtype=np.float64)
    psfFluxErr = np.asarray(psfFluxErr, dtype=np.float64)
    band = np.asarray(band, dtype=str)

    extractor = create_extractor()

    try:
        df = pd.DataFrame({
            "time": midpointMjdTai,
            "flux": psfFlux,
            "err": psfFluxErr,
            "band": band,
        })
    except ValueError:
        logger.error(f"Array length mismatch for object {objectId}")
        return {}

    # Remove NaNs
    mask = df[["time", "flux", "err"]].notna().all(axis=1)
    df = df[mask]

    if df.empty:
        return {}

    # Sort by time (crucial for some features like Cusum, LinearTrend)
    df = df.sort_values(
        "time"
    ).drop_duplicates(
        subset="time"
    )  # Just in case, we delete duplicates, in case there is the same problem as with ZTF.

    full_result = {}

    # Iterate over unique bands present in the alert history
    unique_bands = df["band"].unique()

    for filter_name in unique_bands:
        # Map string band to integer ID
        if filter_name not in LSST_FILTER_MAP:
            continue

        filter_id = LSST_FILTER_MAP[filter_name]

        # Extract subset for this band
        sub = df[df["band"] == filter_name]

        if len(sub) < 1:
            continue

        try:
            result = extractor(
                sub["time"].to_numpy(),
                sub["flux"].to_numpy(),
                sub["err"].to_numpy(),
                sorted=True,  # We sorted beforehand
                fill_value=np.nan,
            )
        except ValueError as err:
            logger.error(f"Value Error for {objectId} in band {filter_name}: {err}")
            continue
        except Exception as e:
            logger.exception(
                f"Unknown exception for {objectId} in band {filter_name}: {e}"
            )
            continue

        # Pack into dictionary
        full_result[int(filter_id)] = dict(
            zip(FEATURES_COLS, [float(v) for v in result])
        )

    return full_result


# Register the UDF
extract_features_ad_rubin = udf(
    f=extract_features_ad_rubin_raw,
    returnType=MapType(
        IntegerType(),  # passband_id
        StructType([  # features name -> value
            StructField(name, DoubleType(), True) for name in FEATURES_COLS
        ]),
    ),
)

if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()

    # We need a normal test here, but I don't know how to do it yet
    np.random.seed(42)
    N = 100
    mjds = np.sort(np.random.uniform(59000, 60000, N))
    flux = np.random.normal(1000, 100, N)  # fake nJy
    fluxerr = np.random.uniform(5, 20, N)
    # Randomly assign bands g and r
    bands = np.random.choice(["g", "r"], N)

    # Test raw function
    features = extract_features_ad_rubin_raw(mjds, flux, fluxerr, bands, "TestObject1")

    # Basic assertions
    assert len(features) == 2, "Should have features for 2 bands (g=1, r=2)"
    assert 1 in features, "Band g (mapped to 1) missing"
    assert 2 in features, "Band r (mapped to 2) missing"
    assert len(features[1]) == len(FEATURES_COLS), (
        "Incorrect number of features extracted"
    )

    print("Test passed: Features extracted successfully.")
    print(f"Example feature (Amplitude band g): {features[1]['amplitude']}")

    # Setup for Spark unit tests (if data files are available)
    # path = os.path.dirname(__file__)
    # rubin_alert_sample = "file://{}/data/alerts/rubin_sample.parquet".format(path)
    # globs["rubin_alert_sample"] = rubin_alert_sample

    # spark_unit_tests(globs)
