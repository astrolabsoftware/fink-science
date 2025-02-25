# Copyright 2020-2025 AstroLab Software
# Author: Andre Santos, Bernardo Fraga, Clecio de Bom
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
import numpy as np


def extract_max_prob(arr: list) -> list:
    """Extract main class and associated probability from a vector of probabilities"""
    if np.isnan(arr[0]):
        return [-1, 0.0]
    array = np.array(arr)
    index = np.argmax(array)
    return [index, array[index]]


def norm_column(col: list) -> np.array:
    col = np.array(col)

    if len(col) == 1:
        norm = [1.0]
    else:
        norm = (col - col.min()) / np.ptp(col)

    return norm
