"""
    Implementation of the paper:
    ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
    Transients
    https://arxiv.org/abs/2404.18165
"""

import numpy as np
import astropy.table as at
from scipy.stats import binned_statistic, wasserstein_distance, kstest


def detect_host_with_powerspectrum(
        sci_image=None, tpl_image=None, number_of_iterations=1000,
        cutout_sizes=[7, 15, 29], metric='kstest'):
    """
    Function to detect host with power spectrum analysis.

    Parameters:
    - sci_image: Science image (default: None)
    - tpl_image: Template image (default: None)
    - number_of_iterations: Number of iterations for shuffling (default: 1000)
    - cutout_sizes: List of cutout sizes for analysis (default: [7, 15, 29])
    - metric: Metric for comparison ('kstest')

    Returns:
    - output_table: Astropy Table containing results
    """

    def get_powerspectrum(data, size):
        """
        Function to compute power spectrum.

        Parameters:
        - data: Image data
        - size: Size of cutout

        Returns:
        - Abins: Binned power spectrum
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
        kbins = np.arange(0.5, size // 2 + 1, 1.)  # Bins for averaging
        Abins, _, _ = binned_statistic(
            knrm, fourier_amplitudes,
            statistic="mean", bins=kbins)  # Binned power spectrum
        # Scale by area of annulus
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        return Abins

    output_table = at.Table(
        names=['IMAGE_TYPE', 'CUTOUT_SIZE', 'STATISTIC', 'PVALUE'],
        dtype=['str', 'int', 'float', 'float'])
    image_type_dict = {0: 'SCIENCE', 1: 'TEMPLATE'}
    output_result_dict = {}

    # Check if the chosen metric is valid
    if np.isin(metric, ['anderson-darling', 'kstest'], invert=True):
        raise Exception(
            "Input metric has not been integrated into the"
            " pipeline yet. Please choose either 'anderson-darling'"
            " or 'kstest'.")

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

            for size in cutout_sizes:
                start = int((full_len - size) / 2)
                stop = int((full_len + size) / 2)

                N_bins = len(np.arange(0.5, size // 2 + 1, 1.)) - 1

                if n == 0:
                    shuffled_Abins_dict[size] = np.zeros(
                        (number_of_iterations, N_bins))

                    image_resized = image[start: stop, start: stop]
                    Abins = get_powerspectrum(image_resized, size)
                    real_Abins_dict[size] = Abins

                copy_resized = copy[start: stop, start: stop]
                Abins = get_powerspectrum(copy_resized, size)
                shuffled_Abins_dict[size][n] = Abins

        # Calculate distances and perform statistical tests
        for size in cutout_sizes:

            WD_dist_real_to_shuffled = []
            WD_dist_shuffled_to_shuffled = []

            for n in range(number_of_iterations):

                iter1 = shuffled_Abins_dict[size][n]
                wd = wasserstein_distance(iter1, real_Abins_dict[size])
                WD_dist_real_to_shuffled.append(wd)

                for m in range(number_of_iterations):
                    if m >= n:
                        continue
                    iter2 = shuffled_Abins_dict[size][m]
                    wd = wasserstein_distance(iter1, iter2)
                    WD_dist_shuffled_to_shuffled.append(wd)

            WD_dist_real_to_shuffled = np.array(WD_dist_real_to_shuffled)
            WD_dist_shuffled_to_shuffled = np.array(
                WD_dist_shuffled_to_shuffled)
            # Small hack to prevent pipeline failing
            if (np.unique(WD_dist_real_to_shuffled).size < 3 or np.unique(
                    WD_dist_shuffled_to_shuffled).size < 3):
                new_row = [image_type_dict[i], size, -1, -1]
            else:
                if metric == 'kstest':
                    res = kstest(WD_dist_real_to_shuffled,
                                 WD_dist_shuffled_to_shuffled)
                    new_row = [image_type_dict[i], size, res.statistic,
                               res.pvalue]

            output_table.add_row(new_row)
            statistic_name = metric + "_" + image_type_dict[
                i] + "_" + str(size) + "_statistic"
            pvalue_name = metric + "_" + image_type_dict[
                i] + "_" + str(size) + "_pvalue"

            output_result_dict[statistic_name] = new_row[2]
            output_result_dict[pvalue_name] = new_row[3]

    return (output_table, output_result_dict,
            real_Abins_dict, shuffled_Abins_dict)
