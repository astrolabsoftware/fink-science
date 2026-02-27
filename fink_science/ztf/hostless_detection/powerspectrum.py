# Copyright 2024 AstroLab Software
# Author: E. E. Hayes, R. Durgesh
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

from typing import Tuple, Dict

import astropy.table as at
from line_profiler import profile
import numpy as np
from scipy.stats import binned_statistic, kstest


def searchsorted_rowwise(
    a: np.ndarray, v: np.ndarray, side: str = "right"
) -> np.ndarray:
    """
    Vectorized numpy search sorted

    Parameters
    ----------
    a
        input array
    v
        values to insert into array
    side
        if left, index of the first suitable location would be selected
        if right index of the last suitable location would be selected

    Returns
    -------
    results
        pairwise wasserstein distance
    """
    if side == "right":
        return np.sum(v[..., None] >= a[:, None, :], axis=-1)
    else:
        return np.sum(v[..., None] > a[:, None, :], axis=-1)


def pairwise_wasserstein_distance(
    u_values: np.ndarray, v_values: np.ndarray
) -> np.ndarray:
    """
    Computes wasserstein distance pairwise.

    Notes
    -----
    extended version of scipy.stats.wasserstein_distance

    Parameters
    ----------
    u_values
        first distribution
    v_values
        second distribution

    Returns
    -------
    results
        pairwise wasserstein distance
    """
    u_values = np.sort(u_values, axis=1)
    v_values = np.sort(v_values, axis=1)

    all_values = np.concatenate((u_values, v_values), axis=1)
    all_values.sort(axis=1)

    deltas = np.diff(all_values, axis=1)

    u_cdf = searchsorted_rowwise(u_values, all_values[:, :-1]) / u_values.shape[1]
    v_cdf = searchsorted_rowwise(v_values, all_values[:, :-1]) / v_values.shape[1]

    return np.sum(np.abs(u_cdf - v_cdf) * deltas, axis=1)


def prepare_powerspectrum(size: int):
    """FFT related values for a given cutout size"""
    # Frequency bins
    kfreq = np.fft.fftfreq(size) * size
    kx, ky = np.meshgrid(kfreq, kfreq, indexing="ij")

    # Magnitudes of wave vectors
    knrm = np.sqrt(kx**2 + ky**2).ravel()
    kbins = np.arange(0.5, size // 2 + 1, 1.0)

    # Area of annulus
    bin_areas = np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return knrm, kbins, bin_areas


def get_powerspectrum(
    image: np.ndarray,
    knrm: np.ndarray,
    kbins: np.ndarray,
    bin_areas: np.ndarray,
) -> np.ndarray:
    """Function to compute power spectrum."""
    # Compute Fourier transform
    fourier_image = np.fft.fftn(image)
    amplitudes = (np.abs(fourier_image) ** 2).ravel()

    # Binned power spectrum
    Abins, _, _ = binned_statistic(knrm, amplitudes, statistic="mean", bins=kbins)
    return Abins * bin_areas


@profile
def detect_host_with_powerspectrum(
    sci_image: np.ndarray = None,
    tpl_image: np.ndarray = None,
    number_of_iterations: int = 1000,
    cutout_size: int = 15,
    metric: str = "kstest",
) -> Tuple[at.Table, Dict, Dict, Dict]:
    """
    Function to detect host with power spectrum analysis.

    Parameters
    ----------
    sci_image
        science stamp image
    tpl_image
        template stamp image
    number_of_iterations
        Number of iterations for shuffling
    cutout_size
        stamp cutout size for analysis
    metric
        metric for distribution comparison ('kstest')

    """
    output_table = at.Table(
        names=["IMAGE_TYPE", "CUTOUT_SIZE", "STATISTIC", "PVALUE"],
        dtype=["str", "int", "float", "float"],
    )

    image_type_dict = {0: "SCIENCE", 1: "TEMPLATE"}
    output_result_dict = {}

    if metric not in {"anderson-darling", "kstest"}:
        raise ValueError(
            "Input metric has not been integrated into the"
            " pipeline yet. Please choose either 'anderson-darling'"
            " or 'kstest'."
        )

    real_Abins_dict = {}  # Dictionary to store real Abins
    shuffled_Abins_dict = {}  # Dictionary to store shuffled Abins

    for idx, image in enumerate([sci_image, tpl_image]):
        if image is None:
            continue

        image_type = image_type_dict[idx]
        full_length = image.shape[0]
        if cutout_size is not None:
            start = (full_length - cutout_size) // 2
            stop = start + cutout_size
            real_cutout = image[start:stop, start:stop]
            knrm, kbins, bin_areas = prepare_powerspectrum(cutout_size)

        else:
            real_cutout = image.copy()
            knrm, kbins, bin_areas = prepare_powerspectrum(full_length)

        n_bins = len(bin_areas)

        shuffled_Abins = np.empty((number_of_iterations, n_bins))

        real_Abins = get_powerspectrum(real_cutout, knrm, kbins, bin_areas)
        real_Abins_dict[cutout_size] = real_Abins

        image_flattened = image.ravel()
        for n in range(number_of_iterations):
            shuffled_flattened = image_flattened.copy()
            np.random.shuffle(shuffled_flattened)
            shuffled_image = shuffled_flattened.reshape(image.shape)
            if cutout_size is not None:
                cutout = shuffled_image[start:stop, start:stop]
            else:
                cutout = shuffled_image.copy()
            shuffled_Abins[n] = get_powerspectrum(cutout, knrm, kbins, bin_areas)

        shuffled_Abins_dict[cutout_size] = shuffled_Abins

        real_repeat = np.repeat(real_Abins[None, :], number_of_iterations, axis=0)
        # Calculate distances and perform statistical tests
        WD_real_to_shuffled = pairwise_wasserstein_distance(shuffled_Abins, real_repeat)

        i_idx, j_idx = np.triu_indices(number_of_iterations, k=1)

        WD_shuffled_to_shuffled = pairwise_wasserstein_distance(
            shuffled_Abins[i_idx],
            shuffled_Abins[j_idx],
        )

        # Small hack to prevent pipeline failing
        if (
            np.unique(WD_real_to_shuffled).size < 3
            or np.unique(WD_shuffled_to_shuffled).size < 3
        ):
            statistic, pvalue = -1.0, -1.0
        else:
            if metric == "kstest":
                res = kstest(
                    WD_real_to_shuffled,
                    WD_shuffled_to_shuffled,
                )
                statistic, pvalue = res.statistic, res.pvalue

        output_table.add_row([image_type, 0, statistic, pvalue])

        output_result_dict[f"{metric}_{image_type}_statistic"] = statistic
        output_result_dict[f"{metric}_{image_type}_pvalue"] = pvalue

    return (
        output_table,
        output_result_dict,
        real_Abins_dict,
        shuffled_Abins_dict,
    )
