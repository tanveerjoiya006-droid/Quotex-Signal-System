from fastapi import FastAPI
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Quotex Signal API", version="1.0")

# Load trained model
model = joblib.load("models/quotex_model.pkl")

@app.get("/")
def home():
    return {"message": "✅ Quotex Signal API is running"}

@app.get("/signal/")
def get_signal(open: float, high: float, low: float, close: float, volume: float):
    df = pd.DataFrame([[open, high, low, close, volume]],
                      columns=["open", "high", "low", "close", "volume"])
    pred = model.predict(df)[0]
    signal = "BUY" if pred == 1 else "SELL"
    return {"signal": signal, "prediction": int(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
