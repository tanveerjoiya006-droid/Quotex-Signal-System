import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load trained model
model = joblib.load("models/quotex_model.pkl")

# Load test data
data = pd.read_csv("data/market_features.csv")

# Select features and target
features = ["open", "high", "low", "close", "volume"]
target = "signal"

X = data[features]
y_true = data[target]

# Predict using trained model
y_pred = model.predict(X)

# Accuracy & Report
acc = accuracy_score(y_true, y_pred)
print(f"\n📈 Model Accuracy: {acc:.2%}\n")
print(classification_report(y_true, y_pred))
