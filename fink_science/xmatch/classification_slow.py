# Copyright 2019 AstroLab Software
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
import io
import logging
import requests
import numpy as np
import pandas as pd

from astroquery.simbad import Simbad
import astropy.coordinates as coord
import astropy.units as u

from fink_science.tester import regular_unit_tests
from fink_science.xmatch.classification import refine_search

def xmatch_slow(
        ra: list, dec: list, id: list,
        distmaxarcsec: int = 1) -> pd.DataFrame:
    """ Build a catalog of (ra, dec, id) as pandas DataFrame,
    cross-match using astroquery module of SIMBAD, and decode the output.

    See https://astroquery.readthedocs.io/ for more information.

    Parameters
    ----------
    ra: list of float
        List of RA
    dec: list of float
        List of Dec of the same size as ra.
    id: list of str
        List of object ID (custom)
    distmaxarcsec: int
        Radius used for searching match. extcatalog sources lying within
        radius of the center (ra, dec) will be considered as matches.

    Returns
    ----------
    data_filt_new: pd.DataFrame
        Formatted decoded data returned by the astroquery module
    """

    # Reset the fields to a minimal number
    Simbad.reset_votable_fields()
    # Add new fields to the query
    Simbad.add_votable_fields('otype')
    Simbad.add_votable_fields('ra(d)')
    Simbad.add_votable_fields('dec(d)')

    list_keys = ['MAIN_ID', 'RA_d', 'DEC_d', 'OTYPE']
    list_old_keys = ['main_id', 'ra', 'dec', 'main_type']
    dictionary_simbad_to_new = dict(zip(list_keys, list_old_keys))

    # create a mask with the entries of the query
    nans = [np.nan] * len(ra)
    mask = pd.DataFrame(list(zip(nans, ra, dec, nans, id)))
    mask.columns = list_old_keys + ['objectId']

    # Send requests in vector form and obtain a table as a result
    units = tuple([u.deg, u.deg])
    query_new = (
        Simbad.query_region(
            coord.SkyCoord(
                ra=ra,
                dec=dec,
                unit=units
            ), radius=u.deg / 3600.
        )
    )

    if query_new is not None:
        # if table not empy, convert it to a pandas DataFrame
        data_new = query_new[list_keys].to_pandas()\
            .rename(columns=dictionary_simbad_to_new)

        # convert from bytes to ascii
        data_new['main_id'] = data_new['main_id'].values[0].decode('ascii')
        data_new['main_type'] = data_new['main_type'].values[0].decode('ascii')

        # locate object id rounding to the first 3 digits
        # This is buggy: if there is no match, then place_objid is False...
        place_objid = data_new['dec'].round(4).values == mask['dec']\
            .round(4).values
        data_new['objectId'] = mask['objectId'].loc[place_objid]

        # create a complementary mask
        complementary_mask = data_new['dec'].round(4).values != mask['dec']\
            .round(4).values

        # concatenate table with mask if needed
        data_filt_new = pd.concat([mask.loc[complementary_mask], data_new])\
            .replace(np.nan, 'Unknown')

        # sort if needed
        data_filt_new = data_filt_new.sort_values(by=['objectId'])

    else:
        logging.warning("Empty query - setting xmatch to Unknown")
        data_filt_new = mask.replace(np.nan, 'Unknown')\
            .sort_values(by=['objectId'])

    return data_filt_new

def cross_match_alerts_raw_slow(oid: list, ra: list, dec: list) -> list:
    """ Query the CDSXmatch service to find identified objects
    in alerts. The catalog queried is the SIMBAD database using the
    astroquery module, as an alternative to the xmatch method.

    Parameters
    ----------
    oid: list of str
        List containing object ids (custom)
    ra: list of float
        List containing object ra coordinates
    dec: list of float
        List containing object dec coordinates

    Returns
    ----------
    out: List of Tuple
        Each tuple contains (objectId, ra, dec, name, type).
        If the object is not found in Simbad, name & type
        are marked as Unknown. In the case several objects match
        the centroid of the alert, only the closest is returned.

    Examples
    ----------
    >>> ra = [26.8566983, 26.24497]
    >>> dec = [-26.9677112, -26.7569436]
    >>> id = ["1", "2"]
    >>> objects = cross_match_alerts_raw_slow(id, ra, dec)
    >>> print(objects) # doctest: +NORMALIZE_WHITESPACE
    [('1', 26.8566983, -26.9677112, 'TYC 6431-115-1', 'Star'),
    ('2', 26.24497, -26.7569436, 'Unknown', 'Unknown')]
    """

    if len(ra) == 0:
        return []

    try:
        data_new = xmatch_slow(ra, dec, oid, distmaxarcsec=1)
    except (ConnectionError, TimeoutError, ValueError) as ce:
        logging.warning("XMATCH failed " + repr(ce))
        return []

    # Fields of interest (their indices in the output)
    if "main_id" not in data_new.columns:
        return []

    main_id = 'main_id'
    main_type = 'main_type'
    oid_ind = 'objectId'

    # Get the objectId of matches
    id_out = list(data_new[oid_ind].values)

    # Get the names of matches
    names = data_new[main_id].values

    # Get the types of matches
    types = data_new[main_type].values

    # Assign names and types to inputs
    out = refine_search(ra, dec, oid, id_out, names, types)

    return out


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    regular_unit_tests(globals())
