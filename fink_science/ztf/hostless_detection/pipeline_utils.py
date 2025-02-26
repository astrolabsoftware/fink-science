# Copyright 2024-2025 AstroLab Software
# Author: R. Durgesh
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

import gzip
import io
import json
from typing import Dict, List
import warnings

from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np

import fink_science.ztf.hostless_detection.powerspectrum as ps

np.random.seed(1337)
warnings.filterwarnings("ignore")


def load_json(file_path: str) -> Dict:
    """
    Loads json file

    Parameters
    ----------
    file_path
       input json file path
    """
    with open(file_path) as json_file:
        return json.load(json_file)


def read_bytes_image(bytes_str: bytes) -> np.ndarray:
    """
    Reads bytes image stamp

    Parameters
    ----------
    bytes_str
       input byte string
    """
    hdu_list = fits.open(gzip.open(io.BytesIO(bytes_str)))
    primary_hdu = hdu_list[0]
    return primary_hdu.data


def apply_sigma_clipping(
    input_data: np.ndarray, sigma_clipping_kwargs: Dict
) -> [np.ma.masked_array, np.ma.masked]:
    """
    Applies sigma clippng

    Parameters
    ----------
    input_data
        stacked input data
    sigma_clipping_kwargs
        parameters for astropy sigma_clip function
    """
    return sigma_clip(input_data, **sigma_clipping_kwargs)


def crop_center_patch(input_image: np.ndarray, patch_radius: int = 7) -> np.ndarray:
    """
    Crops rectangular patch around image center with a given patch scale

    Parameters
    ----------
    input_image
       input image
    patch_radius
        patch radius in pixels
    """
    image_shape = input_image.shape[0:2]
    center_coords = [image_shape[0] / 2, image_shape[1] / 2]
    center_patch_x = int(center_coords[0] - patch_radius)
    center_patch_y = int(center_coords[1] - patch_radius)
    return input_image[
        center_patch_x : center_patch_x + patch_radius * 2,
        center_patch_y : center_patch_y + patch_radius * 2,
    ]


def _check_hostless_conditions(
    science_clipped: np.ndarray, template_clipped: np.ndarray, detection_config: Dict
) -> bool:
    """Counts the number of masked sigma clipping pixels and checks if they are within the range defined in the config

    Parameters
    ----------
    science_clipped
        sigma clipped science image
    template_clipped
        sigma clipped template image
    detection_config
        configs with detection threshold
    """
    num_science_pixels_masked = np.ma.count_masked(science_clipped)
    num_template_pixels_masked = np.ma.count_masked(template_clipped)
    if (
        num_science_pixels_masked > detection_config["max_number_of_pixels_clipped"]
    ) and (
        num_template_pixels_masked < detection_config["min_number_of_pixels_clipped"]
    ):
        return True
    if (
        num_template_pixels_masked > detection_config["max_number_of_pixels_clipped"]
    ) and (
        num_science_pixels_masked < detection_config["min_number_of_pixels_clipped"]
    ):
        return True
    return False


def run_hostless_detection_with_clipped_data(
    science_stamp: np.ndarray, template_stamp: np.ndarray, configs: Dict
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
    detection_config = configs["hostless_detection_with_clipping"]
    is_hostless_candidate = _check_hostless_conditions(
        science_clipped, template_clipped, detection_config
    )
    if is_hostless_candidate:
        return is_hostless_candidate
    science_stamp = crop_center_patch(science_stamp, detection_config["crop_radius"])
    template_stamp = crop_center_patch(template_stamp, detection_config["crop_radius"])
    science_clipped = apply_sigma_clipping(science_stamp, sigma_clipping_config)
    template_clipped = apply_sigma_clipping(template_stamp, sigma_clipping_config)
    is_hostless_candidate = _check_hostless_conditions(
        science_clipped, template_clipped, detection_config
    )
    return is_hostless_candidate


def create_noise_filled_mask(
    image_data: np.ndarray, mask_data: np.ndarray, image_size: List
) -> np.ndarray:
    """
    Creates input image data with noise filled mask

    Parameters
    ----------
    image_data
        input stacked image data
    mask_data
        corresponding input masked data
    image_size
        output image size
    """
    mask = mask_data > 0
    for_filling = np.random.normal(
        np.median(image_data[~mask]), np.std(image_data[~mask]), image_size
    )
    for_filling = np.where(mask, for_filling, 0)
    to_fill = np.where(mask, 0, image_data)
    return to_fill + for_filling


def run_powerspectrum_analysis(
    science_image: np.ndarray,
    template_image: np.ndarray,
    science_mask: np.ndarray,
    template_mask: np.ndarray,
    image_size: List,
    number_of_iterations: int = 200,
) -> Dict:
    """Runs powerspectrum analysis

    Notes
    -----
    transforming the stamps to fourier space as described in the paper:
    https://arxiv.org/abs/2404.18165

    Parameters
    ----------
    science_image
        science stamp
    template_image
        template stamp
    science_mask
        sigma clipped science image
    template_mask
        sigma clipped template image
    image_size
        output image size
    number_of_iterations
        number of iterations for powerspectrum analysis shuffling

    """
    science_data = create_noise_filled_mask(science_image, science_mask, image_size)
    template_data = create_noise_filled_mask(template_image, template_mask, image_size)
    _, kstest_results_dict, _, _ = ps.detect_host_with_powerspectrum(
        science_data,
        template_data,
        number_of_iterations=number_of_iterations,
        metric="kstest",
    )
    return kstest_results_dict
