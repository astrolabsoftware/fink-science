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
import gzip
import io
from astropy.io import fits
import numpy as np

def unzip_cutout(stamp):
    """ Extract an image from a gzip format file
    Image is contains on a fits format file. Due to a significant number of corrupted images,
    a correction step is applied to remove nan values, negative values and corrected the wrong
    shapes of the images

    Parameters
    ----------
    stamp: gzip format file
        an image in a fits file compressed into a gzip format file

    Returns
    -------
    out: 2D numpy array
        alert image after extraction from gzip format and correction of all kinds of problems
    """
    with gzip.open(io.BytesIO(stamp), 'rb') as fits_file:
        with fits.open(io.BytesIO(fits_file.read())) as hdul:
            img = hdul[0].data[::-1]
            img = np.where(img < 0, 0, img)
            if np.shape(img) != (63, 63):
                img_zeros = np.zeros((63, 63))
                idx = np.where(np.logical_not(np.isnan(img)))
                img_zeros[idx] = img[idx]
                return img_zeros
            return np.nan_to_num(img)

def sigmoid(img):
    """ Compute the sigmoid term of the normalization function, the alpha parameter is
    the standard deviation of the image and the beta parameter is the mean of the image

    Parameters:
    img: 2D numpy array
        alert image after extraction from gzip format

    Returns
    -------
    out: float
        the sigmoid term for the image normalisation

    Examples
    --------
    >>> test_1 = np.array([[0, 1, 2], [3, 40, 5], [2, 1, 0]])
    >>> test_2 = np.array([[0, 0, 0], [1, 0.5, 1], [1, 1, 1]])

    >>> sigmoid(test_1)
    array([[ 0.37861437,  0.39822621,  0.41817028],
           [ 0.43838554,  0.94307749,  0.47936865],
           [ 0.41817028,  0.39822621,  0.37861437]])
    >>> sigmoid(test_2)
    array([[ 0.20850741,  0.20850741,  0.20850741],
           [ 0.70033103,  0.43966158,  0.70033103],
           [ 0.70033103,  0.70033103,  0.70033103]])
    """
    img_mean, img_std = img.mean(), img.std()
    img_normalize = (img - img_mean) / img_std
    inv_norm = -img_normalize
    exp_norm = np.exp(inv_norm)
    return 1 / (1 + exp_norm)

def img_normalizer(img, vmin=0, vmax=1):
    """ Compute a non-linear normalisation thanks to sigmoid function of the image.

    Parameters
    ----------
    img: 2D numpy array
        alert image after extraction from gzip format

    Returns
    -------
    out: 2D numpy array
        image where all values is now bounded between vmin and vmax.
        The range is distributed in a non-linear manner due to sigmoid function

    Examples
    --------
    >>> test_1 = np.array([[0, 1, 2], [3, 40, 5], [2, 1, 0]])
    >>> test_2 = np.array([[0, 0, 0], [1, 0.5, 1], [1, 1, 1]])

    >>> img_normalizer(test_1)
    array([[ 0.37861437,  0.39822621,  0.41817028],
           [ 0.43838554,  0.94307749,  0.47936865],
           [ 0.41817028,  0.39822621,  0.37861437]])
    >>> img_normalizer(test_2, vmin = -255, vmax = 255)
    array([[-148.66122095, -148.66122095, -148.66122095],
           [ 102.16882492,  -30.77259383,  102.16882492],
           [ 102.16882492,  102.16882492,  102.16882492]])
    """
    return (vmax - vmin) * sigmoid(img) + vmin
