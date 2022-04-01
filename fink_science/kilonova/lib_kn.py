# Copyright 2021 Fink Software
# Author: Emille E. O. Ishida
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

def return_list_of_kn_host():
    """ Return potential KN host names

    This includes:
    - List of object names in SIMBAD that would correspond to extra-galactic object
    - Unknown objects
    - objects with failed crossmatch

    In practice, this exclude galactic objects from SIMBAD.

    """
    list_simbad_galaxies = [
        "galaxy",
        "Galaxy",
        "EmG",
        "Seyfert",
        "Seyfert_1",
        "Seyfert_2",
        "BlueCompG",
        "StarburstG",
        "LSB_G",
        "HII_G",
        "High_z_G",
        "GinPair",
        "GinGroup",
        "BClG",
        "GinCl",
        "PartofG",
    ]

    keep_cds = \
        ["Unknown", "Candidate_SN*", "SN", "Transient", "Fail"] + \
        list_simbad_galaxies

    return keep_cds
