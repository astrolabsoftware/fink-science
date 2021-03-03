# Copyright 2020 AstroLab Software
# Author: Biswajit Biswas
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

from fink_science.kilonova.LightCurve import LightCurve
import numpy as np
from scipy.optimize import minimize

import os
from fink_science import __file__


def calc_prediction(coeff, PCs, bias=None):
    """ prediction helper

        Parameters
        ----------
        coeff: np.array
            initial prediction
        PCs: 1D np.array
            Principal components generated for each band
        bias:
            adds bias to the predictions. zero by default

        Returns
        ----------
        error: 1D np.array
            curve predicted with current set of coefficients
    """
    predicted_lc = np.zeros_like(PCs.shape[1])
    for a, b in zip(PCs, coeff):
        predicted_lc = np.add(predicted_lc, b * a)
    if bias is not None:
        predicted_lc = predicted_lc + bias
    return predicted_lc


def calc_loss(coeff, PCs, light_curve_seg, bias=None):
    """ Loss function

        Parameters
        ----------
        coeff: np.array
            initial prediction
        PCs: 1D np.array
            Principal components generated for each band
        light_curve_seg: np.array
            actual flux measurements binned in segments of 2 days each.
            If no measurement is made in a bin, value should be set to 0
        bias:
            adds bias to the predictions. zero by default

        Returns
        ----------
        error: np.float64
            least square error between prediction and actual
    """
    index = light_curve_seg != 0
    y_pred = calc_prediction(coeff, PCs, bias=bias)
    diff = light_curve_seg - y_pred
    neg_index = y_pred < 0
    diff = diff[index | neg_index]
    error = np.sum(np.square(diff, diff))
    return error


class PredictLightCurve:

    def __init__(self, data, object_id, num_pc_components=3):

        self.lc = LightCurve(data, object_id)
        self.current_date = None
        self.num_pc_components = num_pc_components
        self.bands = None
        self.pcs = None

        self.min_flux_threshold = 200
        self.num_prediction_points = 51
        self.mid_point_dict = None

    def get_pcs(self, decouple_pc_bands=False, band_choice='u') -> dict:
        """ Fetch principal components already generated

        Parameters
        ----------
        decouple_pc_bands: bool
            selection of same/different pcs for each band
        band_choice:
            choice of band in case of single pcs

        Returns
        ----------
            pc_out: dict
                Dictionary of Principal components for fitting
        """

        curdir = os.path.dirname(os.path.abspath(__file__))
        pc_path = curdir + "/data/models/KN_PC_all_bands.npy"

        if decouple_pc_bands:
            pc_dict = np.load(pc_path, allow_pickle=True)
            pc_dict = pc_dict.item()
            pc_out = {0: pc_dict['u'][0:self.num_pc_components], 1: pc_dict['r'][0:self.num_pc_components],
                      2: pc_dict['i'][0:self.num_pc_components], 3: pc_dict['g'][0:self.num_pc_components],
                      4: pc_dict['z'][0:self.num_pc_components], 5: pc_dict['Y'][0:self.num_pc_components]}

        else:
            pc_out = {}
            pc_dict = np.load(pc_path, allow_pickle=True)
            pc_dict = pc_dict.item()
            for band in self.bands:
                pc_out[band] = pc_dict[band_choice][0:self.num_pc_components]

        return pc_out

    def get_binned_time(self, df):
        return df[self.lc.time_col_name] - df[self.lc.time_col_name] % 2

    def get_mid_pt_dict(self):
        """ Calculate the "predicted mid point" for each band

        Returns
        ----------
            mid_point_dict: dict
                Dictionary of mid points of each band
        """
        mid_point_dict = {}

        event_df = self.lc.df

        for band in self.bands:
            band_index = event_df[self.lc.band_col_name] == band
            band_df = event_df[band_index]
            if len(band_df) > 0:
                max_index = np.argmax(band_df[self.lc.brightness_col_name])
                if band_df[self.lc.brightness_col_name][max_index] > self.min_flux_threshold:
                    mid_point_dict[band] = band_df[self.lc.time_col_name][max_index]
                else:
                    mid_point_dict[band] = None
            else:
                mid_point_dict[band] = None

        return mid_point_dict

    def predict_lc_coeff(self, num_pc_components, bands, decouple_pc_bands=False, band_choice='u',
                         min_flux_threshold=200):
        """ Generate features/coefficients of the fit

            Parameters
            ----------
            num_pc_components: int
                Number of principle components to be considered for the fit
            bands: list
                list of values taken by the filter
            decouple_pc_bands: bool
                selection of same/different pcs for each band
            band_choice: char
                choice of band in case of single pcs
            min_flux_threshold: int
                Minimum value of amplitude of a band for prediction

            Returns
            ----------
                features: 1D np.array
                    features for a given band
        """
        self.num_pc_components = num_pc_components
        self.bands = bands
        self.pcs = self.get_pcs(decouple_pc_bands=decouple_pc_bands, band_choice=band_choice)
        self.min_flux_threshold = min_flux_threshold
        self.num_prediction_points = 51

        self.mid_point_dict = self.get_mid_pt_dict()

        coeff_all_band = {}
        num_points_dict = {}

        if self.mid_point_dict is not None:

            event_df = self.lc.df[:]
            for band in self.bands:
                mid_point_date = self.mid_point_dict[band]
                if mid_point_date is None:
                    coeff_all_band[band] = np.zeros(num_pc_components)
                    num_points_dict[band] = 0
                    continue

                band_index = event_df[self.lc.band_col_name] == band
                band_df = event_df[band_index]
                pcs = self.pcs[band]
                if len(band_df) > 0:

                    binned_dates = self.get_binned_time(band_df)
                    b2 = (binned_dates - mid_point_date + self.num_prediction_points - 1) / 2
                    b2 = b2.astype(int)
                    light_curve_seg = np.zeros(self.num_prediction_points)
                    light_curve_seg[b2[:]] = band_df[self.lc.brightness_col_name]
                    initial_guess = np.zeros(self.num_pc_components)
                    result = minimize(calc_loss, initial_guess, args=(pcs, light_curve_seg))
                    coeff_all_band[band] = list(result.x)
                    num_points_dict[band] = len(b2)

                else:
                    coeff_all_band[band] = np.zeros(num_pc_components)
                    num_points_dict[band] = 0

        else:
            for band in self.bands:
                coeff_all_band[band] = np.zeros(num_pc_components)
                num_points_dict[band] = 0

        features = np.zeros((self.num_pc_components + 1) * len(bands))
        for i, band in enumerate(self.bands):
            for j in range(self.num_pc_components):
                if j == 0:
                    features[i * 4] = num_points_dict[band]
                features[i * 4 + j + 1] = coeff_all_band[band][j]

        return features
