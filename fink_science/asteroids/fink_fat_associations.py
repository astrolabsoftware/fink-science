import pandas as pd

from fink_science.asteroids.kalman_assoc import kalman_association
from fink_science.asteroids.orbit_assoc import orbit_association

from fink_science.tester import spark_unit_tests


def fink_fat_association(
    ra: pd.Series,
    dec: pd.Series,
    magpsf: pd.Series,
    fid: pd.Series,
    jd: pd.Series,
    flags: pd.Series,
    confirmed_sso: pd.Series,
    mag_criterion_same_fid: pd.Series,
    mag_criterion_diff_fid: pd.Series,
    angle_criterion: pd.Series,
    orbit_tw: pd.Series,
):
    """
    Associates the alerts with the orbit or the kalman filters estimates from the trajectories

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
    orbit_tw : int
        time window used to filter the orbit

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
    >>> flags, estimator_id, ffdistnr = fink_fat_association(
    ...     pd.Series([
    ...         46.328490, 108.603010, 97.600172, 98.928007, 2.05, 46.55802234,
    ...         0.254, 48.147, 34.741, 0.198, 0.192
    ...     ]),
    ...     pd.Series([
    ...         18.833964, -67.879693, 32.281571, 23.230676, 3.01, 18.91096543,
    ...         1.036, 65.036, -0.214, 0.987, 0.943
    ...     ]),
    ...     pd.Series([
    ...         16.2, 18.3, 21.4, 19.5, 17.4, 16.4,
    ...         14.234, 18.3, 21.4, 14.429, 14.231
    ...     ]),
    ...     pd.Series([
    ...         1, 1, 2, 2, 2, 1,
    ...         1, 1, 2, 2, 1
    ...     ]),
    ...     pd.Series([
    ...         2460139.8717433237, 2460140.8717433237, 2460139.9917433237, 2460139.8717433237, 2460140.8717432237, 2460140.4217432237,
    ...         2460160.0004537117, 2460160.0007537117, 2460160.0009437117, 2460160.0000537117, 2460160.0009537117
    ...     ]),
    ...     pd.Series([
    ...         2, 3, 1, 2, 3, 3,
    ...         3, 3, 1, 2, 2
    ...     ]),
    ...     pd.Series([True]),
    ...     pd.Series([2]),
    ...     pd.Series([2]),
    ...     pd.Series([30]),
    ...     pd.Series([30]),
    ... )

    >>> flags
    0     2
    1     3
    2     1
    3     2
    4     3
    5     5
    6     4
    7     4
    8     1
    9     2
    10    2
    dtype: int64

    >>> estimator_id
    0                      []
    1                      []
    2                      []
    3                      []
    4                      []
    5     [FF20230802aaaaaaa]
    6                     [0]
    7                     [2]
    8                      []
    9                      []
    10                     []
    dtype: object

    >>> ffdistnr
    0                          []
    1                          []
    2                          []
    3                          []
    4                          []
    5     [5.325306762831802e-06]
    6            [0.151780047003]
    7            [0.805110874669]
    8                          []
    9                          []
    10                         []
    dtype: object
    """
    # fink_fat associations
    ffdistnr = pd.Series([[] for _ in range(len(ra))])
    estimator_id = pd.Series([[] for _ in range(len(ra))])

    ra = ra.values
    dec = dec.values
    magpsf = magpsf.values
    fid = fid.values
    jd = jd.values

    confirmed_sso = confirmed_sso.values[0]
    mag_criterion_same_fid = mag_criterion_same_fid.values[0]
    mag_criterion_diff_fid = mag_criterion_diff_fid.values[0]
    angle_criterion = angle_criterion.values[0]
    orbit_tw = orbit_tw.values[0]

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
        orbit_tw,
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


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()

    # Run the test suite
    spark_unit_tests(globs)
