<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-Regressor-EC6C37?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Data-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Time%20Series-Forecasting-0EA5E9?style=for-the-badge" alt="Time Series">
  <img src="https://img.shields.io/badge/SHAP-Explainability-4B0082?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
</p>

<h1 align="center">The Power Hour</h1>

<p align="center">
  <strong>XGBoost hourly energy demand forecaster for PJM East - 145,366 observations across 16 years, with temporal feature engineering, hyperparameter tuning, and SHAP interpretability.</strong>
</p>

<p align="center">
  <em>Watts next? This model knows.</em>
</p>

---

## Results

| Metric | Validation | Test |
|--------|-----------|------|
| **MAPE** | 0.77% | **0.84%** |
| **RMSE** | 346 MW | 360 MW |
| **MAE** | 246 MW | 267 MW |

Sub-1% MAPE on an out-of-sample test year, with no weather data, no holiday calendars, and no external features. Just timestamps and lagged demand. The tight val/test gap (0.77% vs 0.84%) confirms the model generalizes across time periods without overfitting.

---

## What It Does

Takes 16 years of hourly electricity consumption data for the PJM East region, engineers 20+ temporal features (calendar, cyclical encoding, lag values, rolling statistics), and trains a hyperparameter-tuned XGBoost regressor to predict next-hour demand. The model is evaluated on a held-out test year (2017-2018) with detailed error analysis by hour and month, and explained with SHAP feature importance.

## Dataset

[Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) - PJM East Interconnection hourly demand in megawatts (2002-2018). CC0 Public Domain license.

## How to Run

1. Clone and install:
```bash
git clone https://github.com/lenamonj/the-power-hour.git
cd the-power-hour
pip install -r requirements.txt
```

2. Download `PJME_hourly.csv` from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) and place it in `data/`.

3. Run:
```bash
jupyter notebook the_power_hour.ipynb
```

## Engineered Features

| Category | Features |
|----------|----------|
| **Calendar** | hour, dayofweek, month, dayofyear, weekofyear, quarter, is_weekend |
| **Cyclical** | hour_sin/cos, month_sin/cos, dow_sin/cos |
| **Lag** | 1h, 24h, 48h, 168h (1 week) |
| **Rolling** | 24h mean/std, 7-day mean/std |
| **Trend** | days_since_start |

## Design Decisions

- **XGBoost over LSTM/Prophet.** Gradient-boosted trees with engineered features consistently outperform deep learning and statistical models on tabular time series. Simpler, faster, more interpretable.
- **Temporal splits only.** Train on 2002-2016, validate on 2016-2017, test on 2017-2018. No random shuffling - that would leak future information into training.
- **No external data.** Weather, holidays, and economic indicators would improve accuracy but add dependencies. This notebook demonstrates what pure time series feature engineering can achieve.
- **Low learning rate + early stopping.** Tuned from [0.005, 0.01, 0.02] with up to 5,000 trees. The model finds its own stopping point on validation MAPE.
- **SHAP over feature importance.** XGBoost's built-in importance is biased toward high-cardinality features. SHAP values are theoretically grounded.

## Project Structure

```
.
├── the_power_hour.ipynb    # Full pipeline - one notebook
├── data/                   # Kaggle data (not tracked in git)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # You are here
```

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with Python, XGBoost, and public data. No proprietary tools required.</sub>
</p>
