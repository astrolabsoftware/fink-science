# Anomaly detection module

This module adds new column **anomaly_score** (lower values mean more anomalous observations). It uses 2 pre-trained IsolationForests (1 for each passband) and calculates the score as a mean of their predictions.
