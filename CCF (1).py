import pandas as pd

# This link pulls the official Credit Card Fraud dataset directly into your Colab session
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print("Dataset Loaded Successfully!")
print(f"Dataset Shape: {df.shape}")
print(df['Class'].value_counts()) # This shows how many are Fraud (1) vs Normal (0)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
from imblearn.over_sampling import SMOTE

# 1. Preprocessing
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time'], axis=1)

# 2. Split Data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Handle Imbalance (SMOTE)
print("Balancing data... please wait.")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 4. Train the Model
# We use a smaller n_estimators for speed in this test
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_res, y_train_res)

# 5. Results
y_pred = model.predict(X_test)
print("\n--- ARCHTECHNOLOGIES TASK 4 REPORT ---")
print(classification_report(y_test, y_pred))
print(f"AUPRC Score: {average_precision_score(y_test, y_pred):.4f}")