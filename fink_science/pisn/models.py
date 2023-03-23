# Copyright 2022 Fink Software
# Author: Etienne Russeil
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


def mvsr_right_transient(X, p1, p2, p3):
    newX = X + p3
    protec = np.where(p1 * newX < 100, p1 * newX, 100)
    return p2 * newX * np.exp(protec)


def mvsr_left_transient(X, p1, p2, p3):
    newX = -X + p3
    protec = np.where(p1 * newX < 100, p1 * newX, 100)
    return p2 * newX * np.exp(protec)


if __name__ == "__main__":

    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
