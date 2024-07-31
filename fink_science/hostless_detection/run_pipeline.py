"""
    Implementation of the paper:
    ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
    Transients
    https://arxiv.org/abs/2404.18165
"""
from line_profiler import profile
from typing import Dict, Tuple

import numpy as np
from fink_science.hostless_detection.pipeline_utils import (
    apply_sigma_clipping,
    run_hostless_detection_with_clipped_data, read_bytes_image,
    run_powerspectrum_analysis)


class HostLessExtragalactic:
    """
    Potential hostless candidates detection for extragalactic class

    Parameters
    ----------
    configs
       input config file with different input parameters to use in the class
    """
    def __init__(self, configs: Dict):
        self.configs = configs
        self._image_shape = self.configs["image_shape"]  # List

    @profile
    def process_candidate_fink(self, science_stamp: bytes,
                               template_stamp: bytes) -> float:
        """
        Processes each candidate

        Parameters
        ----------
        science_stamp
           science stamp data
        template_stamp
           template stamp data
        """
        science_stamp = read_bytes_image(science_stamp)

        template_stamp = read_bytes_image(template_stamp)
        if (science_stamp.shape != tuple(self._image_shape)) or (
                template_stamp.shape != tuple(self._image_shape)):
            return -99

        science_stamp_clipped, template_stamp_clipped = (
            self._run_sigma_clipping(science_stamp, template_stamp))
        is_hostless_candidate = run_hostless_detection_with_clipped_data(
            science_stamp_clipped, template_stamp_clipped,
            self.configs)
        if is_hostless_candidate:
            power_spectrum_results = run_powerspectrum_analysis(
                science_stamp, template_stamp,
                science_stamp_clipped.mask.astype(int),
                template_stamp_clipped.mask.astype(int), self._image_shape)
            return power_spectrum_results["kstest_SCIENCE_15_statistic"]
        return -99

    def _run_sigma_clipping(
            self, science_stamp: np.ndarray,
            template_stamp: np.ndarray) -> Tuple[np.ma.masked_array,
                                                 np.ma.masked_array]:
        """
        Runs sigma clipping

        Parameters
        ----------
        science_stamp
           science stamp images
        template_stamp
            template stamp images
        """
        science_stamp_clipped = apply_sigma_clipping(
            science_stamp, self.configs["sigma_clipping_kwargs"])
        template_stamp_clipped = apply_sigma_clipping(
            template_stamp, self.configs["sigma_clipping_kwargs"])
        return science_stamp_clipped, template_stamp_clipped


if __name__ == '__main__':
    pass
