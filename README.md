# Quotex-Signal-System

AI-based Quotex Signal System | Starter repo
============================================
یہ repository ایک runnable starter ہے جو آپ کو:
- data feature pipeline
- LightGBM training + TimeSeriesSplit
- backtest script
- FastAPI backend (signal endpoint + websocket stub)
- simple React frontend stub (Signal viewer)

**Quick start**
1. unzip the repo
2. create a virtualenv: `python -m venv venv` and activate it
3. `pip install -r requirements.txt`
4. Put your historical OHLC data as `data/market_features.csv` (columns: timestamp, open, high, low, close, volume)
5. `python src/train.py`
6. `python src/backtest.py`
7. `uvicorn backend.main:app --reload --port 8000`  # run backend
8. Serve frontend (open `frontend/README.md` for local dev steps)

**Notes**
- This is a starter template. The included model is not trained—train with your own historical data.
- For high-precision operating points (e.g., 92%+), follow the feature-engineering, calibration and walk-forward validation steps in the doc.
