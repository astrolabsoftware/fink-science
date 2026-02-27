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

from typing import Union, Any

from fink_science.ztf.hostless_detection.pipeline_utils import (
    run_powerspectrum_analysis,
    crop_center_patch,
)
from fink_science.ztf.hostless_detection.run_pipeline import HostLessExtragalactic
from fink_science.rubin.hostless_detection.pipeline_utils import (
    read_cutout_stamp,
    run_hostless_detection_with_clipped_data,
)


class HostLessExtragalacticRubin(HostLessExtragalactic):
    """
    Detects potential hostless candidates for extragalactic transients.

    Inherits from the HostLessExtragalactic base class.

    """

    def process_candidate_fink_rubin(
        self, science_stamp: bytes, template_stamp: bytes
    ) -> Union[tuple[None, None], tuple[Any, Any]]:
        """
        Processes each candidate

        Parameters
        ----------
        science_stamp
           science stamp data
        template_stamp
           template stamp data
        """
        science_stamp = read_cutout_stamp(science_stamp)
        template_stamp = read_cutout_stamp(template_stamp)
        crop_radius = self.configs["hostless_detection_with_clipping"]["crop_radius"]
        science_stamp = crop_center_patch(science_stamp, crop_radius)
        template_stamp = crop_center_patch(template_stamp, crop_radius)

        if science_stamp.shape != template_stamp.shape:
            return None, None

        if (science_stamp.shape[0] < self._image_shape[0]) or (
            template_stamp.shape[0] < self._image_shape[0]
        ):
            return None, None

        science_stamp = crop_center_patch(science_stamp, crop_radius)
        template_stamp = crop_center_patch(template_stamp, crop_radius)

        is_hostless_candidate = run_hostless_detection_with_clipped_data(
            science_stamp, template_stamp, self.configs
        )
        if is_hostless_candidate:
            science_stamp_clipped, template_stamp_clipped = self._run_sigma_clipping(
                science_stamp, template_stamp
            )
            power_spectrum_results = run_powerspectrum_analysis(
                science_stamp,
                template_stamp,
                science_stamp_clipped.mask.astype(int),
                template_stamp_clipped.mask.astype(int),
                crop_radius=crop_radius,
            )
            return power_spectrum_results[
                "kstest_SCIENCE_statistic"
            ], power_spectrum_results["kstest_TEMPLATE_statistic"]
        return None, None
