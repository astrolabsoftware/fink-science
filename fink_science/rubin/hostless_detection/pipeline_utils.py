# Copyright 2024-2025 AstroLab Software
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
"""Implementation of the paper: ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical Transients https://arxiv.org/abs/2404.18165"""

import io

from astropy.io import fits
import numpy as np


def read_cutout_stamp(fits_bytes: bytes) -> np.ndarray:
    """
    Reads Rubin cutout stamps

    Parameters
    ----------
    fits_bytes
       input byte string
    """
    fits_buffer = io.BytesIO(fits_bytes)
    with fits.open(fits_buffer) as hdulist:
        return hdulist[0].data
