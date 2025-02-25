# Random Forest Classifier

This module returns the probability of an alert to be a SNe Ia using a Random Forest Classifier (binary classification). It benefits from the work in [Ishida et al](https://arxiv.org/abs/1804.03765). The module implicitly assumes that the model has been pre-trained, and it is loaded on-the-fly by all executors (the model is currently ~20 MB).

Publication: [Leoni et al](https://arxiv.org/abs/2111.11438).