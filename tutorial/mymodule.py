# Copyright 2020 AstroLab Software
# Author: Julien Peloton
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
import pandas as pd

def compute_delta(magpsf: np.array) -> float:
    """ Compute the difference between 2 consecutive magnitude measurements,
    and returns the last one.
    
    Parameters
    ----------
    magpsf: 1d array
        Vector of magnitude measurements from the most ancient
        to the most recent.
        
    Returns
    ----------
    out: float
        Difference between the last 2 measurements. NaN is the difference
        cannot be computed.
    """
    if len(magpsf) <= 1:
        return None
    return np.diff(magpsf)[-1] 