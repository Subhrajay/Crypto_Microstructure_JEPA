# Crypto Market Microstructure with JEPA-Inspired Representation Learning

## Abstract

This repository is an experimental pipeline for learning from cryptocurrency market data and exploring whether predictive latent representations can support short-horizon market-state forecasting. The implementation retrieves historical one-minute Binance klines for model training, derives price-, volume-, order-flow-, and volatility-related features, and trains a Transformer-based predictive model. A separate real-time path aggregates Binance trade-stream messages into minute-level features and exposes market data, model forecasts, and heuristic actions through a FastAPI application and browser interface.

The project is a research prototype, not a validated trading system. In particular, its current training objective predicts future feature vectors directly and should be described as JEPA-inspired rather than as a canonical implementation of Yann LeCun's Joint-Embedding Predictive Architecture. No profitability, execution quality, or out-of-sample predictive performance is established by the code itself.

## Research motivation

Financial market observations are noisy, non-stationary, and affected by changing liquidity and trading behavior. A useful representation-learning question is whether a model can infer the structure of a future market state from a context of recent observations, rather than relying only on hand-designed rules or a single next-price estimate.

JEPA provides a conceptual framework for this question. In a Joint-Embedding Predictive Architecture, an encoder maps observed context and target observations into representation spaces, and a predictor estimates the target representation from the context representation. The prediction is made in representation space, not necessarily by reconstructing every target observation. LeCun's broader proposal emphasizes learning predictive world models that capture relevant structure while avoiding the need to predict unpredictable details.

This repository adapts that general predictive-representation motivation to engineered, minute-level market features. Its current model is a useful starting point for experimentation, but it does not implement all elements commonly associated with modern JEPA systems, such as a distinct target encoder and an explicitly maintained target representation. See [Research context](#research-context) for background reading.

## Research questions

The codebase can support exploratory investigation of questions such as:

1. Do sequences of market microstructure features contain information about subsequent feature states?
2. How do short-horizon predictions compare with simple persistence or statistical baselines under a chronological evaluation?
3. Do order-flow and activity features add measurable value beyond price-derived indicators?
4. Does a latent-space predictive objective improve representation quality or downstream performance relative to direct feature prediction?

These are research questions, not conclusions. A rigorous study should specify hypotheses and evaluation criteria before examining test results.

## Current implementation

### Data collection and feature construction

Historical training data are requested from Binance as one-minute klines. The pipeline constructs the following 11 model features:

- Log return
- Volume normalized by a historical average
- Order-flow imbalance
- Volatility normalized by a historical average
- Log-transformed trade count
- Normalized hour of day and day of week
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger-band width
- Rolling return volatility

A `StandardScaler` is fitted to the feature data and saved as `scaler.pkl`. Historical klines are cached in a symbol-specific pickle file.

For live processing, a Binance trade WebSocket stream is aggregated into minute-level bars and features. Those features are normalized with the saved scaler. The implementation uses historical kline data for training and an aggregated trade stream for inference; the equivalence of these two feature-generation paths has not been established and should be tested before drawing scientific conclusions.

### Model and objective

`JEPAModel` projects each feature vector to a 128-dimensional representation and processes a sequence using a six-layer Transformer encoder with eight attention heads. A predictor maps the final context representation to five future 11-dimensional feature vectors. Training combines mean-squared error between these predicted vectors and observed future feature vectors with a weighted VICReg-style regularization term.

As written, the training objective compares predictions directly with observed future features; it does not use a separate target encoder to produce target embeddings. Accordingly, the model is JEPA-inspired, but its objective is not the standard latent-target prediction formulation. The auxiliary regularization and naming in the source should not be taken as evidence that the implementation reproduces a published JEPA method.

### Real-time interface and action simulation

`main.py` provides a FastAPI application with a root page and a `/ws` WebSocket endpoint. The app streams recent prices, technical indicators, model forecasts, and actions to connected clients. The pipeline also contains a model-based controller that samples candidate action sequences and scores simulated futures using configured return, risk, and transaction-cost weights.

The controller does not submit orders to Binance. Actions are experimental outputs only; the project does not include exchange execution, portfolio accounting, or production risk controls.

## Repository contents

| Path | Purpose |
| --- | --- |
| `crypto_trading_pipeline.py` | Data processing, feature engineering, model, training utilities, streaming client, and action simulation |
| `main.py` | FastAPI application and WebSocket interface |
| `index.html` | Browser-based interface |
| `requirements.txt` | Python package dependencies |

## Installation and execution

### Requirements

Use Python 3.9 or later and install the listed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

### Train a model

Run the pipeline and enter a Binance symbol, for example `BTCUSDT`:

```bash
python crypto_trading_pipeline.py
```

If model artifacts are absent, the script requests historical data, trains the model, and writes `jepa_model.pth` and `scaler.pkl`. The script then starts the real-time processing loop. Historical market-data access is required; API credentials are not configured by this project for the public market-data calls shown in the code.

### Start the web application

After the model artifacts have been generated, start the API:

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` to load the interface. The application initializes a market-data stream at startup and serves streaming updates on `/ws`.

## Experimental protocol and evaluation

The repository currently provides training and inference code, but does not provide a complete scientific evaluation protocol or report. Before interpreting model behavior, an experiment should at minimum:

1. Define the prediction target, forecast horizon, asset universe, and sampling period in advance.
2. Split data chronologically into training, validation, and test periods. Do not randomly shuffle overlapping time windows across these splits.
3. Fit preprocessing statistics on training data only, then apply the frozen transformation to validation and test data.
4. Compare against transparent baselines, such as persistence, zero return, and simple autoregressive or linear models.
5. Report appropriate forecast metrics (for example, MAE or RMSE for features and returns), uncertainty or variability across time periods, and results by asset and market regime.
6. If evaluating trading decisions, include fees, spread, slippage, latency, position constraints, and risk-adjusted metrics; use a strictly held-out backtest and avoid treating simulated actions as executable or profitable trades.
7. Record the data retrieval date, symbol, exact configuration, random seeds, software versions, and model artifact provenance so the experiment can be reproduced.

The current training loader shuffles sequence examples, and the repository does not implement a dedicated chronological validation/test workflow. The reported training loss alone is not evidence of generalization or trading value.

## Implementation limitations to address

The following issues are important when extending or evaluating this prototype:

- **JEPA formulation:** The current loss predicts future feature values directly. A canonical latent-prediction experiment would define context and target encoders, target representations, and a predictor, and would document how target parameters are updated.
- **Sequence-length inconsistency:** Training uses 60 context steps, while the FastAPI initialization uses a 10-step feature buffer. Make the sequence length a shared configuration and validate inference with the same context length used during training.
- **Feature-order consistency:** The live feature buffer orders features by sorted dictionary keys, whereas training uses the explicit `feature_cols` order. Ensure both paths use one named, fixed feature ordering before inference.
- **Training/inference consistency:** Historical kline-derived and live trade-derived features may have different statistical definitions and sampling behavior. Align the computations and verify distributions before comparing predictions.
- **Scaling and leakage:** Historical averages and scaling are computed from the downloaded period. A valid time-series evaluation must estimate all preprocessing parameters from the training interval only.
- **Evaluation and reproducibility:** No benchmark suite, held-out evaluation, fixed seed protocol, or published results are included. Add these before making claims about forecasting performance.
- **Operational safety:** The app uses permissive CORS settings and the controller is not an execution or risk-management system. Do not expose it as a production service without security review and appropriate controls.

## Safety and scope

This software is provided for research and educational experimentation. Cryptocurrency markets involve substantial risk. Model forecasts and actions may be incorrect, unstable, or invalidated by changing market conditions. Nothing in this repository is financial advice or a recommendation to buy or sell any asset. Do not use the prototype to place live orders or make financial decisions without independent validation, appropriate safeguards, and qualified review.

## Research context

The following publications provide background on the conceptual motivation and related architectures. They are references for the research direction, not evidence that this code reproduces their methods or results.

1. Yann LeCun. “A Path Towards Autonomous Machine Intelligence.” OpenReview, 2022. [Paper](https://openreview.net/forum?id=BZ5a1r-kVsf).
2. Mahmoud Assran et al. “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” CVPR, 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
3. Adrien Bardes et al. “Revisiting Feature Prediction for Learning Visual Representations from Video.” arXiv, 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471).

## License

This project is distributed under the GNU General Public License, version 3. See [`LICENSE`](LICENSE) for the complete terms.
