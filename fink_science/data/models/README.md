# Default models used for the Fink science modules

To load and run model properly, you need to check carefuly the dependencies used in Fink: [deps](https://github.com/astrolabsoftware/fink-broker/tree/master/deps).


## ZTF

Default models used for ZTF:

| model name | used in | Description |
|------------|---------|-------------|
| `snn_models/snn_snia_vs_nonia/model.pt`| `snn` | Ia vs core-collapse SNe |
| `snn_models/snn_sn_vs_all/model.pt`| `snn` | SNe vs. anything else (var star and other stuff in training) |
| `default-model_sigmoid.obj` | `random_forest_snia` | Early SN Ia classification using lightcurve features (random forest). Binary classification. `sklearn==1.0.2` |
| `rf.sav` | `microlensing` | scikit-learn model used in Random Forest. `sklearn==1.0.2` |
| `pca.sav` | `microlensing` | scikit-learn (pca) model used in Random Forest. `sklearn==1.0.2` |
|`components.csv`| `kilonova`| file containing principal components from simulations|
| `for_al_loop/*.pkl` | active learning loop | Models used in the context of Active Learning for early SN Ia detection. `sklearn==1.0.2` |
|`anomaly_detection`| `anomaly_detection` | Zip files with different models used for Anomaly detection |

Models for Kilonova are in [kndetect](https://github.com/b-biswas/kndetect).

## Elasticc/Rubin

More detailed can be found in [2404.08798](https://arxiv.org/abs/2404.08798).

| model name | used in | Description |
|------------|---------|-------------|
| `elasticc_rainbow_earlyIa_after_leak.pkl` | `random_forest_snia` | Early SN Ia classification using Rainbow features. Binary classification. `sklearn==1.0.2`. |
| `snn_models/elasticc_ia` | `snn` | Binary classifier using SuperNNova. Ia vs core-collapse SNe. |
| `snn_models/snn_sn_vs_others` | `snn` | Binary classifier using SuperNNova. SNe vs others |
| `snn_models/snn_periodic_vs_others` | `snn` | Binary classifier using SuperNNova. Periodic vs others |
|`snn_models/snn_nonperiodic_vs_others`| `snn` | Binary classifier using SuperNNova.  Non-periodic vs others |
|`snn_models/snn_long_vs_others`| `snn` | Binary classifier using SuperNNova. Long vs others |
|`snn_models/snn_fast_vs_others`| `snn` | Binary classifier using SuperNNova. Fast vs others |
|`SLSN_rainbow_MD.joblib`| `slsn` | SLSN classifier using Rainbow features. Binary classificatio. Random Forest. `sklearn==1.0.2`.
|`cats_models`| `cats` | Broad class classifier using CATS. Tensorflow.


