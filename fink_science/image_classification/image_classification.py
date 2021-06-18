# Copyright 2021 AstroLab Software
# Author: Roman Le Montagner
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
import pandas as pd
import gzip
import io
from astropy.io import fits
import numpy as np
from fink_science.image_classification.utils import img_normalizer

from skimage.exposure import equalize_adapthist
from skimage.filters import median
from skimage.filters import threshold_triangle
from skimage.measure import label
from skimage.measure import regionprops_table
from skimage.segmentation import chan_vese

def is_neg(img):
    """ Test if an image contains negative values

    Parameters
    ----------
    img: 2D numpy array
        alert image after extraction from gzip format

    Returns
    -------
    out: bool
        return True if img contains negative values, False otherwise

    Examples
    --------
    >>> test_1 = [[-1, 1, 1], [3, 5, -1], [0, 4, -3]]
    >>> test_2 = [[1, 1, 1], [3, 5, 1], [0, 4, 3]]

    >>> is_neg(test_1)
    True
    >>> is_neg(test_2)
    False
    """
    return not np.all(np.greater_equal(img, 0))

def peak_snr(img):
    """ Estimate the noise level of an image

    NB: The noise level threshold for the image classification is set to 3.5

    Parameters
    ----------
    img: 2D numpy array
        alert image after extraction from gzip format

    Returns
    -------
    out: float
        a noise level estimation of the image

    Examples
    --------
    >>> test_1 = [[0, 1, 2], [3, 40, 5], [2, 1, 0]]
    >>> peak_snr(test_1)
    6.666666666666667
    >>> test_2 = [[0, 0, 0], [1, 0.5, 1], [1, 1, 1]]
    >>> peak_snr(test_2)
    1.6363636363636362
    """
    return np.max(img) / np.mean(img)

def img_labelisation(stamp, noise_threshold=3.5):
    """ Perform image classification based on their visual content.
    Two final labels available for images which are not noisy and not corrupted.
    Star label means this image contains only ponctual objects.
    Extend label means this image contains at least one extend object.

    Object size is only based on a perimeter calculation and custom thresholding, false positive
    can occur when ponctual objects is sufficiently large or multiple ponctual object is
    sufficiently close to pass thresholds.

    Parameters
    ----------
    stamp: gzip format file
        an image in fits file compressed into a gzip format file

    Returns
    -------
    out: string
        a string which contains all the labels assigned during the classification process
    All possible returns are:
        - 'corrupted_noisy'
        - 'corrupted_clear'
        - 'safe_noisy'
        - 'safe_clear_star'
        - 'safe_clear_extend'

    Examples
    --------
    >>> df = spark.read.format('parquet').load(ztf_alert_sample).select(['objectId', 'cutoutScience']).toPandas()

    >>> example_byte_array = list(df[df['objectId'] == 'ZTF18acrunkm']['cutoutScience'])[0]['stampData']
    >>> img_labelisation(example_byte_array)
    'safe_clear_star'

    >>> example_byte_array = list(df[df['objectId'] == 'ZTF20aafdzuq']['cutoutScience'])[0]['stampData']
    >>> img_labelisation(example_byte_array)
    'safe_noisy'

    >>> example_byte_array = list(df[df['objectId'] == 'ZTF18aabipja']['cutoutScience'])[0]['stampData']
    >>> img_labelisation(example_byte_array)
    'corrupted_clear'

    >>> example_byte_array = list(df[df['objectId'] == 'ZTF18abuajuu']['cutoutScience'])[0]['stampData']
    >>> img_labelisation(example_byte_array)
    'safe_clear_extend'
    """
    with gzip.open(io.BytesIO(stamp), 'rb') as fits_file:
        with fits.open(io.BytesIO(fits_file.read())) as hdul:
            img = hdul[0].data[::-1]

            label_img = ""

            # detect if image is corrupted or/and is noisy
            if np.any(np.isnan(img)):
                label_img += "corrupted_"

            # shift the image if it contains negative values
            elif is_neg(img):
                img = img + np.abs(np.min(img))
                label_img += "safe_"
            else:
                label_img += "safe_"

            if peak_snr(img) <= noise_threshold:
                label_img += "noisy"
            else:
                label_img += "clear"

            # if image is not corrupted and not noisy
            if label_img == "safe_clear":
                label_img += "_"
                # define threshold between ponctual object and extend object for the first pass
                star_limit = 30

                # remove background of the image and keep only high value signal
                threshold = threshold_triangle(img)
                # binarize the image with the threshold
                thresh_img = np.where(img < threshold, 0, 1).astype(np.bool)
                # labeled segmented part and create region
                labeled_img = label(thresh_img, connectivity=1).astype(np.byte)

                # define the properties that we want to compute on the segmented part of the image
                properties = ('label', 'perimeter')
                region_props = regionprops_table(labeled_img, intensity_image=img, properties=properties)
                region_df = pd.DataFrame(region_props)

                object_max_size = list(region_df['perimeter'])
                if len(object_max_size) > 0:

                    # get the object of maximal size in the segmented image
                    object_max_size = np.max(object_max_size)

                    # if the maximal size object is small enough then the image is classed as star
                    # else the image go to the second pass
                    if object_max_size < star_limit:
                        label_img += "star"
                    else:
                        # image is normalized between -1 and 1
                        norm_img = img_normalizer(img, -1, 1)
                        # then a median filter is applied to reduce some noise
                        norm_img = median(norm_img)
                        # and finally the image contrast is enhanced by an histogram equalization method
                        norm_img = equalize_adapthist(norm_img, clip_limit=0.01, nbins=512)

                        # the enhanced image is then processed by the chan vese algorithm.
                        # the image is segmented between high intensity region and low intensity region.
                        # source: https://arxiv.org/abs/1107.2782
                        cv = chan_vese(norm_img, mu=0, lambda1=1, lambda2=2, tol=1e-9, max_iter=600,
                                       dt=100, init_level_set="checkerboard").astype(np.bool)

                        # the segmented region is then labeled in order to compute some information
                        labeled_img_cv = label(cv, connectivity=1).astype(np.byte)
                        # the properties computed are the same as the first part but the area is added. It is just a properties
                        # that return the number of pixels of the region.
                        properties = ('label', 'area', 'perimeter')
                        region_props_cv = regionprops_table(labeled_img_cv, intensity_image=img, properties=properties)
                        region_df_chan_vese = pd.DataFrame(region_props_cv)
                        # a small filter remove the regions with only one pixels. we assume that one pixel area are just noise.
                        zero_filter = region_df_chan_vese[region_df_chan_vese['area'] != 1]

                        object_size = list(zero_filter['perimeter'])

                        if len(object_size) > 0:

                            object_max_size = np.max(object_size)

                            # a new higher threshold is used because median filtering and histogram equalization tend to
                            # expand the size of ponctual object.
                            if object_max_size < 40:
                                label_img += "star"
                            else:
                                # extend label is given to images that pass all steps.
                                label_img += "extend"

                        else:
                            label_img += "errorchanvese"

                else:
                    label_img += "errorthreshold"

            return label_img
