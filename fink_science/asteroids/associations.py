import numpy as np
import pandas as pd
import math

from sklearn.neighbors import BallTree

from astropy.time import Time
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u

import sbpy.data as sso_py

from pyspark import SparkFiles

from fink_fat.seeding.dbscan_seeding import dist_3d
from fink_fat.kalman.kalman_prediction import kalmanDf_prediction
from fink_fat.associations.associations import angle_three_point_vect


def roid_mask(ra, dec, jd, magpsf, fid, flags, real_sso):
    if real_sso:
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


def df_to_orb(df_orb):

    df_orb["targetname"] = df_orb["ssoCandId"]
    df_orb["orbtype"] = "KEP"

    df_orb["H"] = 14.45
    df_orb["G"] = 0.15

    orb_dict = df_orb.to_dict(orient="list")

    orb_dict["a"] = orb_dict["a"] * u.au
    orb_dict["i"] = orb_dict["i"] * u.deg
    orb_dict["node"] = orb_dict["long. node"] * u.deg
    orb_dict["argper"] = orb_dict["arg. peric"] * u.deg
    orb_dict["M"] = orb_dict["mean anomaly"] * u.deg
    orb_dict["epoch"] = Time(orb_dict["ref_epoch"], format="jd")
    orb_dict["H"] = orb_dict["H"] * u.mag

    ast_orb_db = sso_py.Orbit.from_dict(orb_dict)
    return ast_orb_db


def compute_ephem(orbits, epochs, location="I41"):
    """
    epochs in jd
    """
    orb_table = df_to_orb(orbits)

    return sso_py.Ephem.from_oo(
        orb_table, epochs=Time(epochs, format="jd"), location="I41", scope="basic"
    ).table.to_pandas()


def orbit_window(orbit_pdf: pd.DataFrame, coord_alerts: SkyCoord, jd: np.ndarray, orbit_tw: int) -> pd.DataFrame:
    """
    Filter the orbits in orbit_pdf to keep only those close to the alert 
    and those that are the most recently updated (within orbit_tw). 

    Parameters
    ----------
    orbit_pdf : pd.DataFrame
        dataframe containing the orbit
    coord_alerts : SkyCoord
        _description_
    jd : np.ndarray
        _description_
    orbit_tw : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    jd_min = np.min(jd.values)
    jd_max = np.max(jd.values)
    min_night_jd = Time(math.modf(jd_min)[1], format="jd").jd
    max_night_jd = Time(math.modf(jd_max)[1], format="jd").jd
    last_orbits = orbit_pdf[
            (orbit_pdf["ref_epoch"] <= max_night_jd)
            & (orbit_pdf["ref_epoch"] >= (min_night_jd - orbit_tw))
        ]

    coord_kalman = SkyCoord(
        last_orbits["last_ra"].values,
        last_orbits["last_dec"].values,
        unit="deg",
    )
    (
        idx_orbit,
        _,
        _,
        _,
    ) = search_around_sky(
        coord_kalman,
        coord_alerts,
        2 * u.deg,
    )

    orbit_to_keep = orbit_pdf[
        orbit_pdf["ssoCandId"].isin(
            orbit_pdf.iloc[orbit_pdf]["ssoCandId"].unique()
        )
    ]
    return orbit_to_keep


def orbit_association(
    ra: np.ndarray,
    dec: np.ndarray,
    jd: np.ndarray,
    magpsf: np.ndarray,
    fid: np.ndarray,
    flags: np.ndarray,
    real_sso: bool,
    estimator_id: pd.Series,
    ffdistnr: pd.Series,
    mag_criterion_same_fid: float,
    mag_criterion_diff_fid: float,
    orbit_tw: int
):
    (
        ra_mask,
        dec_mask,
        coord_alerts,
        mag_mask,
        fid_mask,
        jd_mask,
        jd_unique,
        idx_keep_mask,
    ) = roid_mask(ra, dec, jd, magpsf, fid, flags, real_sso)
    
    # get latest detected orbit
    orbit_pdf = pd.read_parquet(SparkFiles.get("orbital.parquet"))
    orbit_to_keep = orbit_window(orbit_pdf, coord_alerts, jd_unique)

    if len(orbit_to_keep) == 0:
        return flags, estimator_id, ffdistnr

    # compute ephem from the latest orbits and find the matching alerts coordinates
    ephem = compute_ephem(orbit_to_keep, jd.unique())
    ephem_coord = SkyCoord(ephem["RA"], ephem["DEC"], unit=u.degree)

    # return the closest alerts of each ephemerides
    res_search = coord_alerts.match_to_catalog_sky(ephem_coord)

    #TODO Match idx_mask with idx_keep_mask to get the associated alerts

    # return the distance to the ephem and the associated orbit id
    return flags, estimator_id, ffdistnr


# pd.DataFrame({
#         "ffdistnr": res_search[1],
#         "ssoCandId": ephem.loc[res_search[0], "targetname"]
#     })


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
    real_sso: bool,
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
    ra : np.
        _description_
    dec : Spark DataFrame Column
        _description_
    jd : Spark DataFrame Column
        _description_
    magpsf : Spark DataFrame Column
        _description_
    fid : _type_
        _description_
    flags : _type_
        _description_
    real_sso : _type_
        _description_
    t_estimator : _type_
        _description_
    estimator_id : _type_
        _description_
    ffdistnr : _type_
        _description_
    mag_criterion_same_fid : _type_
        _description_
    mag_criterion_diff_fid : _type_
        _description_
    angle_criterion : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    (
        ra_mask,
        dec_mask,
        coord_masked_alerts,
        mag_mask,
        fid_mask,
        jd_mask,
        jd_unique,
        idx_keep_mask,
    ) = roid_mask(ra, dec, jd, magpsf, fid, flags, real_sso)

    # path where are stored the kalman filters
    kalman_pdf = pd.read_pickle(SparkFiles.get("kalman.pkl"))

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
    diff_mag = result_pdf_kalman["mag"] - result_pdf_kalman["mag_1"]
    diff_jd = result_pdf_kalman["jd"] - result_pdf_kalman["jd_1"]
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


def fink_fat_association(
    ra,
    dec,
    magpsf,
    fid,
    jd,
    ndethist,
    flags,
    real_sso,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
):
    """
    Find the alerts close to the predictions made by the kalman filters.

    Parameters
    ----------
    ra : Spark DataFrame Column
        _description_
    dec : Spark DataFrame Column
        _description_
    magpsf : _type_
        _description_
    fid : Spark DataFrame Column
        _description_
    jd : Spark DataFrame Column
        _description_
    ndethist : _type_
        Spark DataFrame Column
    flags : _type_
        Spark DataFrame Column
    real_sso : _type_
        Spark DataFrame Column
    mag_criterion_same_fid : _type_
        Spark DataFrame Column
    mag_criterion_diff_fid : _type_
        Spark DataFrame Column
    angle_criterion : _type_
        Spark DataFrame Column

    Returns
    -------
    t_estimator: pd.Series

    estimator_id:
    ffdistnr:
    """
    # fink_fat associations
    ffdistnr = pd.Series([[] for _ in range(len(ndethist))])
    estimator_id = pd.Series([[] for _ in range(len(ndethist))])

    ra = ra.values
    dec = dec.values
    magpsf = magpsf.values
    fid = fid.values
    jd = jd.values

    real_sso = real_sso.values[0]
    mag_criterion_same_fid = mag_criterion_same_fid.values[0]
    mag_criterion_diff_fid = mag_criterion_diff_fid.values[0]
    angle_criterion = angle_criterion.values[0]

    # associates the alerts with the kalman filters
    flags, estimator_id, ffdistnr = kalman_association(
        ra,
        dec,
        jd,
        magpsf,
        fid,
        flags,
        real_sso,
        estimator_id,
        ffdistnr,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion,
    )

    return flags, estimator_id, ffdistnr
