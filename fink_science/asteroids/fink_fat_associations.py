import pandas as pd
import sys
import doctest

from fink_science.asteroids.kalman_assoc import kalman_association
from fink_science.asteroids.orbit_assoc import orbit_association


def fink_fat_association(
    ra,
    dec,
    magpsf,
    fid,
    jd,
    ndethist,
    flags,
    confirmed_sso,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    orbit_tw
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
    confirmed_sso : _type_
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

    confirmed_sso = confirmed_sso.values[0]
    mag_criterion_same_fid = mag_criterion_same_fid.values[0]
    mag_criterion_diff_fid = mag_criterion_diff_fid.values[0]
    angle_criterion = angle_criterion.values[0]

    flags, estimator_id, ffdistnr = orbit_association(
        ra, 
        dec, 
        jd, 
        magpsf, 
        fid, 
        flags, 
        confirmed_sso, 
        estimator_id,
        ffdistnr, 
        mag_criterion_same_fid, 
        mag_criterion_diff_fid, 
        orbit_tw
    )


    # associates the alerts with the kalman filters
    flags, estimator_id, ffdistnr = kalman_association(
        ra,
        dec,
        jd,
        magpsf,
        fid,
        flags,
        confirmed_sso,
        estimator_id,
        ffdistnr,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion,
    )

    return flags, estimator_id, ffdistnr


if __name__ == "__main__":  # pragma: no cover
    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
