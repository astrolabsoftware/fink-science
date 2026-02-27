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
from typing import Dict

from astropy.io import fits
import numpy as np

from fink_science.ztf.hostless_detection.pipeline_utils import (
    apply_sigma_clipping,
    _check_hostless_conditions,
    crop_center_patch,
)


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


def is_outliers_in_template(clipped_image, number_of_pixels: int = 20):
    """Checks if is a big host pixels by applying sigma clipping

    The threshold value is decided based on the samples we have seen.
    But should be updated eventually
    """
    num_template_pixels_masked = np.ma.count_masked(clipped_image)
    print(num_template_pixels_masked)
    if num_template_pixels_masked > number_of_pixels:
        return True
    return False


def run_hostless_detection_with_clipped_data(
    science_stamp: np.ndarray,
    template_stamp: np.ndarray,
    configs: Dict,
    check_outliers_in_template: bool = True,
) -> bool:
    """Detects potential hostless candidates

    Notes
    -----
    We use sigma clipped stamp images by
     cropping an image patch from the center of the image.
    If pixels are rejected in scientific image but not in corresponding
     template image, such candidates are flagged as potential hostless

    Parameters
    ----------
    science_stamp
       science image
    template_stamp
        template image
    configs
        detection configs with detection threshold
    """
    sigma_clipping_config = configs["sigma_clipping_kwargs"]

    science_clipped = apply_sigma_clipping(science_stamp, sigma_clipping_config)
    template_clipped = apply_sigma_clipping(template_stamp, sigma_clipping_config)

    if check_outliers_in_template:
        if is_outliers_in_template(template_clipped):
            return False
    detection_config = configs["hostless_detection_with_clipping"]

    is_hostless_candidate = _check_hostless_conditions(
        science_clipped, template_clipped, detection_config
    )
    if is_hostless_candidate:
        return is_hostless_candidate

    # Check again at half resolution
    crop_radius = int(detection_config["crop_radius"] / 2)
    science_stamp = crop_center_patch(science_stamp, crop_radius)
    template_stamp = crop_center_patch(template_stamp, crop_radius)

    science_clipped = apply_sigma_clipping(science_stamp, sigma_clipping_config)
    template_clipped = apply_sigma_clipping(template_stamp, sigma_clipping_config)
    is_hostless_candidate = _check_hostless_conditions(
        science_clipped, template_clipped, detection_config
    )
    return is_hostless_candidate
