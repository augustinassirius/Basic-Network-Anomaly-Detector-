# Basic-Network-Anomaly-Detector-

- The code generates pretend traffic records (packet size, duration, bytes sent, etc.).
Some of them are marked “normal” (0) and some as “anomaly” (1).
It scales (normalizes) the numbers so that models can learn properly.

- Then it splits the data into training and testing parts.
  Random Forest → learns patterns of “normal” vs. “anomaly” from labelled data.
  Isolation Forest → looks for “weird” data points without needing labels.
  
- The code picks random samples (pretend they’re live traffic).
  Each sample is checked by both models.
  If either model thinks it’s suspicious → it prints [ALERT] Anomaly.
  Otherwise, it prints Normal.
- Every detection (alert or normal) is also written into a log file called anomaly_alerts.log with a timestamp.
