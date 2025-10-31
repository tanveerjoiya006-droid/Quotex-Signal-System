import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import os

# Check and create model directory
os.makedirs("models", exist_ok=True)

# Load your data
data = pd.read_csv("data/market_features.csv")

# Feature columns and target
features = ["open", "high", "low", "close", "volume"]
target = "signal"

# Split data
X = data[features]
y = data[target]

tscv = TimeSeriesSplit(n_splits=5)
best_model = None
best_acc = 0

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Fold accuracy: {acc:.2%}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"\n✅ Best model accuracy: {best_acc:.2%}")
joblib.dump(best_model, "models/quotex_model.pkl")
print("\nModel saved to models/quotex_model.pkl")
