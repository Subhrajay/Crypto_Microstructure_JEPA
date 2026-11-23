# 🪙 Crypto Trading Pipeline

A high-frequency cryptocurrency trading pipeline that collects real-time market data using WebSockets, performs feature engineering, and predicts price movements using a Self-Supervised, Energy-Based Joint Embedding Predictive Architecture (JEPA).

## Project Structure

```text
.
├── crypto_trading_pipeline.py   # Main trading pipeline: data collection, processing, and prediction
├── main.py                      # FastAPI backend for serving model predictions
├── index.html                   # Simple frontend UI for interaction
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Features

- Real-time WebSocket data collection from Binance
- Tick-level market data processing with feature extraction
- JEPA-inspired model for price and feature prediction
- FastAPI backend for prediction and WebSocket endpoints
- Browser-based frontend interface
- Efficient data storage and model persistence using `joblib` and PyTorch

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Subhrajay/Crypto_Microstructure_JEPA.git
cd Crypto_Microstructure_JEPA
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train or load the model

Run the pipeline and enter a Binance trading symbol when prompted:

```bash
python crypto_trading_pipeline.py
```

The pipeline fetches or loads historical one-minute klines and generates `jepa_model.pth` and `scaler.pkl` when a model is not already available.

### 4. Run the backend server

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` in a browser to interact with the interface.

## Model Overview

The implementation uses a Transformer-based encoder to process 11 market features from recent observations. It then predicts five future feature vectors and combines mean-squared prediction loss with a VICReg-style variance and covariance regularization term.

The model uses:

- Log return
- Scaled volume
- Order-flow imbalance
- Scaled volatility
- Log-scaled trade count
- Hour and weekday encodings
- RSI
- MACD
- Bollinger-band width
- Rolling volatility

A model-predictive controller samples buy, sell, and hold action sequences and estimates their predicted return, risk, and transaction cost.

The implementation is JEPA-inspired: it learns predictive representations, but the current objective compares predicted feature vectors directly with observed future features rather than using a separate target encoder and latent target representations.

## API and Interfaces

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Serves the browser interface |
| WebSocket | `/ws` | Streams prices, indicators, predictions, and simulated actions |

## Requirements

The dependencies are listed in [`requirements.txt`](requirements.txt), including:

- `websocket-client`
- `python-binance`
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `fastapi`
- `uvicorn`

## Notes

- Binance public market-data streams are used by default.
- The live implementation uses BTCUSDT unless another symbol is selected through the interface.
- The controller only proposes actions; it does not submit exchange orders or manage a live account.
- The implementation should be evaluated with chronological train, validation, and test splits before drawing conclusions about predictive or trading performance.
- The action, feature, and prediction outputs should not be treated as financial advice or as evidence of profitability.

## License

This project is licensed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).
