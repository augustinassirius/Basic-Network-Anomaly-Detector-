import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import logging

# Setup logging for alerts
logging.basicConfig(
    filename="anomaly_alerts.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Simulated Network Data (simple)
np.random.seed(1)
n_samples = 2000

data = pd.DataFrame({
    'packet_size': np.random.randint(200, 1500, n_samples),
    'duration': np.random.rand(n_samples) * 10,
    'src_bytes': np.random.randint(0, 5000, n_samples),
    'dst_bytes': np.random.randint(0, 5000, n_samples),
    'connections': np.random.randint(1, 200, n_samples),
    'label': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 0 = normal, 1 = anomaly
})

# Preprocessing
X = data.drop("label", axis=1)
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

# Random Forest (supervised)
rf = RandomForestClassifier(n_estimators=50, random_state=1)
rf.fit(X_train, y_train)

# Isolation Forest (unsupervised)
iso = IsolationForest(contamination=0.1, random_state=1)
iso.fit(X_train)

# Real-time Detection with Logging
def detect(sample):
    sample_scaled = scaler.transform([sample])
    rf_pred = rf.predict(sample_scaled)[0]
    iso_pred = iso.predict(sample_scaled)[0]
    iso_pred = 1 if iso_pred == -1 else 0

    if rf_pred == 1 or iso_pred == 1:
        msg = f"[ALERT] Anomaly: {sample}"
        print(msg)
        logging.info(msg)
    else:
        msg = f"Normal: {sample}"
        print(msg)
        logging.info(msg)

print("\n[Real-time Simulation]\n")
for i in range(5):
    sample = X.sample(1).values[0]
    detect(sample)
    time.sleep(1)