import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u

from pyspark import SparkFiles

from sklearn.neighbors import BallTree

from fink_fat.seeding.dbscan_seeding import dist_3d
from fink_fat.kalman.kalman_prediction import kalmanDf_prediction
from fink_fat.associations.associations import angle_three_point_vect
from fink_fat.others.utils import init_logging

from fink_science.tester import spark_unit_tests


def roid_mask(
    ra: np.ndarray,
    dec: np.ndarray,
    jd: np.ndarray,
    magpsf: np.ndarray,
    fid: np.ndarray,
    flags: np.ndarray,
    confirmed_sso: bool,
):
    """
    Return the inputs masked as solar sytem objects (confirmed or canddiates depending of confirmed_sso)

    Parameters
    ----------
    ra : np.ndarray
        right ascension
    dec : np.ndarray
        declination
    jd : np.ndarray
        julian date
    magpsf : np.ndarray
        estimated magnitude of the psf
    fid : np.ndarray
        filter identifier
    flags : np.ndarray
        roid flags
    confirmed_sso : bool
        if true, used confirmed solar system object,
        used candidates otherwise

    Returns
    -------
    ra_mask : np.ndarray
        sso masked right ascension
    dec_mask : np.ndarray
        sso masked declination
    coord_alerts : np.ndarray
        sso masked coordinates
    mag_mask : np.ndarray
        sso masked magpsf
    fid_mask : np.ndarray
        sso masked filter id
    jd_mask : np.ndarray
        sso masked julian date
    jd_unique : np.ndarray
        sso masked unique julian date
    idx_keep_mask : np.ndarray
        idx in the non masked array of the masked data

    Examples
    --------
    >>> roid_mask(
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([15, 16, 17, 18, 19, 20]),
    ... np.array([1, 1, 2, 1, 2, 2]),
    ... np.array([1, 3, 2, 2, 3, 0]),
    ... True
    ... )
    (array([1, 4]), array([1, 4]), <SkyCoord (ICRS): (ra, dec) in deg
        [( 1.,  1.), ( 4.,  4.)]>, array([16, 19]), array([1, 2]), array([1, 4]), array([1, 4]), array([1, 4]))

    >>> roid_mask(
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([15, 16, 17, 18, 19, 20]),
    ... np.array([1, 1, 2, 1, 2, 2]),
    ... np.array([1, 3, 2, 2, 3, 0]),
    ... False
    ... )
    (array([0, 2, 3]), array([0, 2, 3]), <SkyCoord (ICRS): (ra, dec) in deg
        [( 0.,  0.), ( 2.,  2.), ( 3.,  3.)]>, array([15, 17, 18]), array([1, 2, 1]), array([0, 2, 3]), array([0, 2, 3]), array([0, 2, 3]))
    """
    if confirmed_sso:
        keep_mask = flags == 3
    else:
        mask_first_time = flags == 1
        mask_possible_sso = flags == 2
        keep_mask = mask_first_time | mask_possible_sso

    ra_mask = ra[keep_mask]
    dec_mask = dec[keep_mask]
    coord_alerts = SkyCoord(
        ra_mask,
        dec_mask,
        unit="deg",
    )

    jd_mask = jd[keep_mask]
    mag_mask = magpsf[keep_mask]
    fid_mask = fid[keep_mask]
    jd_unique = np.unique(jd_mask)
    idx_keep_mask = np.where(keep_mask)[0]
    return (
        ra_mask,
        dec_mask,
        coord_alerts,
        mag_mask,
        fid_mask,
        jd_mask,
        jd_unique,
        idx_keep_mask,
    )


def kalman_window(kalman_pdf: pd.DataFrame, coord_alerts: SkyCoord) -> pd.DataFrame:
    """
    Filter the kalman in the kalman_pdf to keep only those close to the alerts.

    Parameters
    ----------
    kalman_pdf : pd.DataFrame
        dataframe containing the kalman filters
    coord_alerts : SkyCoord
        coordinates of the alerts

    Returns
    -------
    pd.DataFrame
        the kalman dataframe with the kalman only close to the alerts

    Examples
    --------
    >>> kalman_pdf = pd.DataFrame({
    ...     "ra_1": [0, 10, 30, 50],
    ...     "dec_1": [0, 10, 30, 50],
    ...     "trajectory_id": [0, 1, 2, 3]
    ... })

    >>> coord_alerts = SkyCoord([10, 50], [10, 50], unit="deg")

    >>> kalman_window(kalman_pdf, coord_alerts)
       ra_1  dec_1  trajectory_id
    1    10     10              1
    3    50     50              3
    """
    coord_kalman = SkyCoord(
        kalman_pdf["ra_1"].values,
        kalman_pdf["dec_1"].values,
        unit="deg",
    )
    (
        idx_kalman,
        _,
        _,
        _,
    ) = search_around_sky(
        coord_kalman,
        coord_alerts,
        2 * u.deg,
    )
    kalman_to_keep = kalman_pdf[
        kalman_pdf["trajectory_id"].isin(
            kalman_pdf.iloc[idx_kalman]["trajectory_id"].unique()
        )
    ]
    return kalman_to_keep


def kalman_association(
    ra: np.ndarray,
    dec: np.ndarray,
    jd: np.ndarray,
    magpsf: np.ndarray,
    fid: np.ndarray,
    flags: np.ndarray,
    confirmed_sso: bool,
    estimator_id: pd.Series,
    ffdistnr: pd.Series,
    mag_criterion_same_fid: float,
    mag_criterion_diff_fid: float,
    angle_criterion: float,
):
    """
    Associates the alerts with the kalman filters

    Parameters
    ----------
    ra : np.ndarray
        right ascension of the alerts (degree)
    dec : np.ndarray
        declination of the alerts (degree)
    jd : np.ndarray
        exposure time of the alerts (julian date)
    magpsf : np.ndarray
        psf magnitude of the alerts
    fid : np.ndarray
        filter identifier of the alerts
    flags : np.ndarray
        roid flags
    confirmed_sso : bool
        if true, run the associations with the alerts flagged as 3 (confirmed sso)
        otherwise, run the association with the alerts flagged as 1 or 2 (candidates sso)
    estimator_id : pd.Series
        will contains the identifier of the orbits associated with the alerts
    ffdistnr : pd.Series
        will contains the distance between the ephemeries and the alerts
    mag_criterion_same_fid : float
        the criterion to filter the alerts with the same filter identifier for the magnitude
        as the last point used to compute the orbit
    mag_criterion_diff_fid : float
        the criterion to filter the alerts with the filter identifier for the magnitude
        different from the last point used to compute the orbit
    angle_criterion : float
        angle between the last two point of the trajectories and the associated point close to a kalman prediction.
        keep only the associated alerts with an angle below this treshold.

    Returns
    -------
    flags: pd.Series
        contains the flags of the roid module
        see processor.py
    estimator_id:
        contains the orbit identifier, same as the ssoCandId column
    ffdistnr:
        contains the distance between the alerts and the ephemeries (degree)

    Examples
    --------
    >>> from fink_science.tester import add_roid_datatest
    >>> add_roid_datatest(spark)
    >>> flags, estimator_id, ffdistnr = kalman_association(
    ...     np.array([0.254, 48.147, 34.741, 0.198, 0.192]),
    ...     np.array([1.036, 65.036, -0.214, 0.987, 0.943]),
    ...     np.array(
    ...         [
    ...             2460160.0004537117,
    ...             2460160.0007537117,
    ...             2460160.0009437117,
    ...             2460160.0000537117,
    ...             2460160.0009537117
    ...         ]
    ...     ),
    ...     np.array([14.234, 18.3, 21.4, 14.429, 14.231]),
    ...     np.array([1, 1, 2, 2, 1]),
    ...     np.array([3, 3, 1, 2, 2]),
    ...     False,
    ...     pd.Series([[], [], [], [], []]),
    ...     pd.Series([[], [], [], [], []]),
    ...     2,
    ...     2,
    ...     30,
    ... )

    >>> flags
    array([3, 3, 1, 4, 4])

    >>> estimator_id
    0        []
    1        []
    2        []
    3       [0]
    4    [0, 3]
    dtype: object

    >>> ffdistnr
    0                                   []
    1                                   []
    2                                   []
    3                    [0.0794180462412]
    4    [0.0693217578247, 0.128325098098]
    dtype: object


    >>> flags, estimator_id, ffdistnr = kalman_association(
    ...     np.array([0.254, 48.147, 34.741, 0.198]),
    ...     np.array([1.036, 65.036, -0.214, 0.987]),
    ...     np.array(
    ...         [
    ...             2460160.0004537117,
    ...             2460160.0007537117,
    ...             2460160.0009437117,
    ...             2460160.0000537117
    ...         ]
    ...     ),
    ...     np.array([14.234, 18.3, 21.4, 14.429]),
    ...     np.array([1, 1, 2, 2]),
    ...     np.array([3, 3, 1, 2]),
    ...     True,
    ...     pd.Series([[], [], [], []]),
    ...     pd.Series([[], [], [], []]),
    ...     2,
    ...     2,
    ...     30,
    ... )

    >>> flags
    array([4, 4, 1, 2])

    >>> estimator_id
    0    [0]
    1    [2]
    2     []
    3     []
    dtype: object

    >>> ffdistnr
    0    [0.151780047003]
    1    [0.805110874669]
    2                  []
    3                  []
    dtype: object
    """
    logger = init_logging()
    (
        ra_mask,
        dec_mask,
        coord_masked_alerts,
        mag_mask,
        fid_mask,
        jd_mask,
        jd_unique,
        idx_keep_mask,
    ) = roid_mask(ra, dec, jd, magpsf, fid, flags, confirmed_sso)

    try:
        # path where are stored the kalman filters
        kalman_pdf = pd.read_pickle(SparkFiles.get("kalman.pkl"))
    except FileNotFoundError:
        logger.warning("files containing the kalman filters not found", exc_info=1)
        return flags, estimator_id, ffdistnr

    # filter the kalman estimators to keep only those inside the current exposures.
    kalman_to_keep = kalman_window(kalman_pdf, coord_masked_alerts)

    if len(kalman_to_keep) == 0:
        return flags, estimator_id, ffdistnr

    # make predictions based on the kalman estimators
    kal_pred = kalmanDf_prediction(
        kalman_to_keep,
        jd_unique,
    ).reset_index(drop=True)

    cart_coord_masked_alert = coord_masked_alerts.cartesian.xyz.T.value
    coord_kalman = SkyCoord(
        kal_pred["ra"].values,
        kal_pred["dec"].values,
        unit="deg",
    )

    # construct the ball tree (slowest instruction: bottleneck of this function)
    tree = BallTree(cart_coord_masked_alert, leaf_size=15, metric="euclidean")

    X_err_kalman = dist_3d(kal_pred["delta_ra"].values.astype(np.float64))

    # find all the alerts inside the error circle of each kalman prediction
    res = tree.query_radius(
        coord_kalman.cartesian.xyz.T.value,
        X_err_kalman,
    )

    # construct a dataframe with the results (sequential instruction)
    traj_id_kal_pred = kal_pred["trajectory_id"].values

    # idx_mask columns are the indices in the masked array
    # idx_alert columns are the indices in the original array
    results_pdf = pd.DataFrame(
        [
            [
                idx_alert,
                i,
                idx_keep_mask[idx_alert],
                traj_id_kal_pred[i],
                ra_mask[idx_alert],
                dec_mask[idx_alert],
                jd_mask[idx_alert],
                mag_mask[idx_alert],
                fid_mask[idx_alert],
            ]
            for i, idx_l in enumerate(res)
            for idx_alert in idx_l
        ],
        columns=[
            "idx_mask",
            "idx_pred",
            "idx_alert",
            "trajectory_id",
            "ra",
            "dec",
            "jd",
            "mag",
            "fid",
        ],
    )
    # compute the separation between the predictions and the alerts
    # WARNING: the separation is computed between all the predictions and the matched alerts
    # The prediction time does not necessarily correspond to the alerts time
    # however, the separation return by the module is sufficiently close to the true separation
    results_pdf["sep"] = (
        coord_masked_alerts[results_pdf["idx_mask"]]
        .separation(coord_kalman[results_pdf["idx_pred"]])
        .deg
    )
    # keep only the minimum separation for each associations
    sep_min_tmp = (
        results_pdf.groupby(["trajectory_id", "idx_alert"])
        .agg(sep_min=("sep", "min"))
        .reset_index()
    )
    results_sep = results_pdf.merge(sep_min_tmp, on=["trajectory_id", "idx_alert"])
    # compute the angle between the last two points of the trajectories and the alerts
    result_pdf_kalman = results_sep.merge(kalman_pdf, on="trajectory_id")
    angle = angle_three_point_vect(
        result_pdf_kalman[["ra_0", "dec_0"]].values,
        result_pdf_kalman[["ra_1", "dec_1"]].values,
        result_pdf_kalman[["ra", "dec"]].values,
    )

    # compute the difference of magnitude between the last measurements and the alerts
    diff_mag = np.abs(result_pdf_kalman["mag"] - result_pdf_kalman["mag_1"])
    diff_jd = result_pdf_kalman["jd"] - result_pdf_kalman["jd_1"]
    assert np.all(diff_jd) > 0

    mag_rate = np.where(diff_jd > 1, diff_mag / diff_jd, diff_mag)
    angle_rate = np.where(diff_jd > 1, angle / diff_jd, angle)
    mag_criterion = np.where(
        result_pdf_kalman["fid"] == result_pdf_kalman["fid_1"],
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
    )

    # keep only the associations satisfying the criterion
    result_filtered = result_pdf_kalman[
        (angle_rate < angle_criterion) & (mag_rate < mag_criterion)
    ]

    with pd.option_context("mode.chained_assignment", None):
        result_filtered["trajectory_id"] = result_filtered["trajectory_id"].astype(str)

    filtered_group = (
        result_filtered.groupby("idx_alert")
        .agg(tr_unique=("trajectory_id", "unique"), sep_min_l=("sep_min", "unique"))
        .reset_index()
    )
    idx_to_update = filtered_group["idx_alert"].values.astype(int)

    flags[idx_to_update] = 4
    estimator_id[idx_to_update] = filtered_group["tr_unique"]
    ffdistnr[idx_to_update] = filtered_group["sep_min_l"]

    return flags, estimator_id, ffdistnr


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()

    # Run the test suite
    spark_unit_tests(globs)
