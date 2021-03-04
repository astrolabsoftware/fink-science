# Default models used for the Fink science modules

| model name | used in | Description |
|------------|---------|-------------|
| `snn_models/snn_snia_vs_nonia/model.pt`| `snn` | Ia vs core-collapse SNe |
| `snn_models/snn_sn_vs_all/model.pt`| `snn` | SNe vs. anything else (var star and other stuff in training) |
| `default-model.obj` | `random_forest_snia` | scikit-learn model used in Random Forest |
| `rf.sav` | `microlensing` | scikit-learn model used in Random Forest |
| `pca.sav` | `microlensing` | scikit-learn (pca) model used in Random Forest |
|`model_1PC_2KN_Cosmins_models.pkl`| `kilonova`| KNe model using 1 principal component (default)|
|`model_3PC_2KN_Cosmins_models.pkl`| `kilonova`| KNe model using 3 principal component|
|`components.csv`| `kilonova`| file containing principal components from simulations|
