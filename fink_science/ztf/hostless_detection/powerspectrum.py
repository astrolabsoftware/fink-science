# Copyright 2024 AstroLab Software
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

from line_profiler import profile
from typing import Dict, Tuple
import numpy as np
import astropy.table as at
from scipy.stats import binned_statistic, kstest

np.random.seed(1337)


def searchsorted_2d(a, v, side="right", sorter=None):
    """Vectorized numpye searchsorted method from here:https://stackoverflow.com/a/52825077

    Parameters
    ----------
    a
        input array
    v
        values to insert into array
    side
        if left, index of the first suitable location would be selected
        if right index of the last suitable location would be selected
    sorter
        sorted method

    Returns
    -------
    results
        pairwise wasserstein distance
    """
    a = np.asarray(a)
    v = np.asarray(v)

    # Augment a with row id
    ai = np.empty(a.shape, dtype=[("row", int), ("value", a.dtype)])
    ai["row"] = np.arange(a.shape[0]).reshape(-1, 1)
    ai["value"] = a

    # Augment v with row id
    vi = np.empty(v.shape, dtype=[("row", int), ("value", v.dtype)])
    vi["row"] = np.arange(v.shape[0]).reshape(-1, 1)
    vi["value"] = v
    # Perform searchsorted on augmented array.
    # The row information is embedded in the values, so only the equivalent rows
    # between a and v are considered.
    result = np.searchsorted(ai.flatten(), vi.flatten(), side=side, sorter=sorter)
    # Restore the original shape, decode the searchsorted indices so
    # they apply to the original data.
    result = result.reshape(vi.shape) - vi["row"] * a.shape[1]
    return result


def pairwise_wasserstein_distance(
    u_values: np.ndarray, v_values: np.ndarray
) -> np.ndarray:
    """Computes wasserstein distance pairwise.

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
    all_values = np.concatenate((u_values, v_values), axis=1)
    all_values.sort(kind="mergesort")
    deltas = np.diff(all_values)
    u_cdf_indices = searchsorted_2d(np.sort(u_values), all_values[:, :-1])
    v_cdf_indices = searchsorted_2d(np.sort(v_values), all_values[:, :-1])
    v_cdf = v_cdf_indices / v_values.shape[1]
    u_cdf = u_cdf_indices / u_values.shape[1]
    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas), axis=1)


def get_powerspectrum(data: np.ndarray, size: int) -> np.ndarray:
    """Function to compute power spectrum.

    Parameters
    ----------
    data
        image data
    size
        stamp cutout size
    """
    fourier_image = np.fft.fftn(data)  # Compute Fourier transform
    # Compute Fourier amplitudes
    fourier_amplitudes = np.abs(fourier_image) ** 2
    kfreq = np.fft.fftfreq(size) * size  # Frequency bins
    kfreq2D = np.meshgrid(kfreq, kfreq)
    # Magnitudes of wave vectors
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, size // 2 + 1, 1.0)  # Bins for averaging
    Abins, _, _ = binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )  # Binned power spectrum
    # Scale by area of annulus
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return Abins


@profile
def detect_host_with_powerspectrum(
    sci_image: np.ndarray = None,
    tpl_image: np.ndarray = None,
    number_of_iterations: int = 1000,
    cutout_size: int = 15,
    metric: str = "kstest",
) -> Tuple[at.Table, Dict, Dict, Dict]:
    """Function to detect host with power spectrum analysis.

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

    # Check if the chosen metric is valid
    if np.isin(metric, ["anderson-darling", "kstest"], invert=True):
        raise Exception(
            "Input metric has not been integrated into the"
            " pipeline yet. Please choose either 'anderson-darling'"
            " or 'kstest'."
        )

    # Loop through science and template images
    for i, image in enumerate([sci_image, tpl_image]):
        if image is None:
            continue

        full_len = len(image)

        real_Abins_dict = {}  # Dictionary to store real Abins
        shuffled_Abins_dict = {}  # Dictionary to store shuffled Abins

        # Iterate through shuffling process
        for n in range(number_of_iterations):
            copy = np.copy(image)
            copy = copy.reshape(full_len * full_len)
            np.random.shuffle(copy)
            copy = copy.reshape((full_len, full_len))
            start = int((full_len - cutout_size) / 2)
            stop = int((full_len + cutout_size) / 2)
            N_bins = len(np.arange(0.5, cutout_size // 2 + 1, 1.0)) - 1
            if n == 0:
                shuffled_Abins_dict[cutout_size] = np.zeros((
                    number_of_iterations,
                    N_bins,
                ))
                image_resized = image[start:stop, start:stop]
                Abins = get_powerspectrum(image_resized, cutout_size)
                real_Abins_dict[cutout_size] = Abins

            copy_resized = copy[start:stop, start:stop]
            Abins = get_powerspectrum(copy_resized, cutout_size)
            shuffled_Abins_dict[cutout_size][n] = Abins

        # Calculate distances and perform statistical tests
        WD_dist_real_to_shuffled = pairwise_wasserstein_distance(
            shuffled_Abins_dict[cutout_size],
            np.concatenate(
                [[real_Abins_dict[cutout_size]]] * number_of_iterations, axis=0
            ),
        )
        indices = [i for i in range(1, number_of_iterations) for k in range(i)]
        shuffled_Abins_dict_continuous_repeat = shuffled_Abins_dict[cutout_size][
            indices
        ]
        shuffled_Abins_dict_slice_repeat = np.concatenate([
            shuffled_Abins_dict[cutout_size][0:i] for i in range(number_of_iterations)
        ])
        WD_dist_shuffled_to_shuffled = pairwise_wasserstein_distance(
            shuffled_Abins_dict_continuous_repeat, shuffled_Abins_dict_slice_repeat
        )

        WD_dist_real_to_shuffled = np.array(WD_dist_real_to_shuffled)
        WD_dist_shuffled_to_shuffled = np.array(WD_dist_shuffled_to_shuffled)
        # Small hack to prevent pipeline failing
        if (
            np.unique(WD_dist_real_to_shuffled).size < 3
            or np.unique(WD_dist_shuffled_to_shuffled).size < 3
        ):
            new_row = [image_type_dict[i], cutout_size, -1, -1]
        else:
            if metric == "kstest":
                res = kstest(WD_dist_real_to_shuffled, WD_dist_shuffled_to_shuffled)
                new_row = [image_type_dict[i], cutout_size, res.statistic, res.pvalue]

        output_table.add_row(new_row)
        statistic_name = (
            metric + "_" + image_type_dict[i] + "_" + str(cutout_size) + "_statistic"
        )
        pvalue_name = (
            metric + "_" + image_type_dict[i] + "_" + str(cutout_size) + "_pvalue"
        )

        output_result_dict[statistic_name] = new_row[2]
        output_result_dict[pvalue_name] = new_row[3]

    return (output_table, output_result_dict, real_Abins_dict, shuffled_Abins_dict)
