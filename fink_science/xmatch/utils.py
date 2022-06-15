# Copyright 2019-2021 AstroLab Software
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
import csv
import pandas as pd

from fink_science.tester import regular_unit_tests

def extract_gcvs(filename):
    """ Read the gcvs catalog and extract useful columns

    Parameters
    ----------
    filename: str
        Path to the catalog (parquet file)

    Returns
    ----------
    out: pd.Series, pd.Series, pd.Series
        ra, dec, VarType from the catalog

    Examples
    ----------
    >>> import os
    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> catalog = curdir + '/../data/catalogs/gcvs.parquet'
    >>> ra, dec, vartype = extract_gcvs(catalog)
    """
    pdf = pd.read_parquet(filename)
    return pdf['ra'], pdf['dec'], pdf['VarType']

def extract_vsx(filename):
    """ Read the vsx catalog and extract useful columns

    Parameters
    ----------
    filename: str
        Path to the catalog (parquet file)

    Returns
    ----------
    out: pd.Series, pd.Series, pd.Series
        ra, dec, VarType from the catalog

    Examples
    ----------
    >>> import os
    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> catalog = curdir + '/../data/catalogs/vsx.parquet'
    >>> ra, dec, vtype = extract_vsx(catalog)
    """
    pdf_vsx = pd.read_parquet(filename)
    return pdf_vsx['RAJ2000'], pdf_vsx['DEJ2000'], pdf_vsx['VType']

def generate_csv(s: str, lists: list) -> str:
    """ Make a string (CSV formatted) given lists of data and header.
    Parameters
    ----------
    s: str
        String which will contain the data.
        Should initially contain the CSV header.
    lists: list of lists
        List containing data.
        Length of `lists` must correspond to the header.

    Returns
    ----------
    s: str
        Updated string with one row per line.

    Examples
    ----------
    >>> header = "toto,tata\\n"
    >>> lists = [[1, 2], ["cat", "dog"]]
    >>> table = generate_csv(header, lists)
    >>> print(table)
    toto,tata
    1,"cat"
    2,"dog"
    <BLANKLINE>
    """
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
    _ = [writer.writerow(row) for row in zip(*lists)]
    return s + output.getvalue().replace('\r', '')


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    regular_unit_tests(globals())
