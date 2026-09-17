# AI Smart Traffic Flow Prediction

An AI-driven traffic analytics platform that predicts **traffic volume**, **travel time**, **congestion level**, and **accident risk** using sensor, weather, and calendar data — built to help city traffic authorities move from reactive to proactive traffic management.

## Live Dashboard

The `dashboard/` folder contains a Streamlit app that visualizes all four models' performance and provides a live "what-if" prediction console.

```bash
git clone https://github.com/poonam18-djn/AI-Smart-Traffic-Flow-Prediction-Problem.git
cd AI-Smart-Traffic-Flow-Prediction-Problem/dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Overview

| Target | Type | Model | Score |
|---|---|---|---|
| Traffic Volume | Regression | Random Forest | R² = 0.992 |
| Travel Time | Regression | Random Forest (leakage-free) | R² = 0.787 |
| Congestion Level | Classification (4-class) | Random Forest (leakage-free) | Accuracy = 0.849 |
| Accident Risk | Classification (binary, imbalanced) | Logistic Regression (leakage-free) | ROC-AUC = 0.637 |

A supplementary **LSTM sequence model** was also trained for Traffic Volume, using only raw historical volume (no engineered features), achieving R² = 0.847 — a strong result given it sees far less information than the Random Forest.

## Data

Three sources, merged on timestamp and date:
- **Traffic sensor logs** — 171,887 readings across 25 roads, 30-minute intervals, Jan–May 2025 (volume, speed, occupancy, congestion, travel time, accidents, signal timing, road capacity)
- **Weather observations** — hourly readings per station (3 weather stations), matched to traffic readings via nearest timestamp
- **Calendar events** — daily flags for public holidays, special events, and roadwork

## Methodology Highlights

**Data leakage was identified and fixed.** Initial models for Travel Time and Congestion Level scored near-perfectly (R²/Accuracy > 0.99) because both targets were mathematically derived from other columns already present in the same row:
- `travel_time = fixed_road_distance / avg_speed`
- `congestion_level` was a deterministic threshold on `traffic_volume / road_capacity`

Both models were rebuilt using only **lagged (past) values** instead of concurrent readings, turning them into genuine forecasting tasks appropriate for proactive traffic management. Performance dropped to more honest, defensible levels (Travel Time R² 0.99 → 0.79; Congestion Accuracy 0.99 → 0.85).

**Cross-validation exposed further temporal leakage.** Standard K-Fold cross-validation (random shuffling) substantially overestimated performance compared to `TimeSeriesSplit` (chronological, train-on-past/test-on-future) due to autocorrelation between nearby time periods. `TimeSeriesSplit` was adopted as the methodologically appropriate validation strategy for this forecasting task.

**Accident risk has a natural performance ceiling.** Despite adding weather/calendar features, richer lag features, and SMOTE oversampling, ROC-AUC plateaued around 0.64. This suggests accidents are driven substantially by factors outside macro-level sensor and weather monitoring (e.g., driver behavior, momentary hazards) — a limitation consistent with broader traffic safety literature, and a natural direction for future work.

## Repository Structure

```
├── dashboard/
│   ├── app.py                          # Streamlit dashboard (7 pages)
│   ├── dashboard_data.json             # Precomputed evaluation data
│   ├── model_volume.pkl                # Traffic volume model
│   ├── model_traveltime_fixed.pkl      # Travel time model (leakage-free)
│   ├── model_congestion.pkl            # Congestion level model
│   ├── model_accident.pkl              # Accident risk model
│   ├── model_*_features.pkl            # Feature lists for each model
│   ├── model_accident_scaler.pkl       # Scaler for accident risk features
│   └── requirements.txt
├── full_pipeline.py                    # End-to-end: raw data -> cleaned, merged, 4 trained models
├── travel_time_fixed_model.py          # Rebuilds travel time with lagged (leakage-free) features
├── dl_model.py                         # LSTM sequence model for traffic volume
├── generate_dashboard_data.py          # Computes evaluation data for the dashboard
├── evaluation_charts.py                # Actual vs predicted, confusion matrix, ROC curve charts
├── cross_validation_all_targets.py     # KFold vs TimeSeriesSplit validation
├── hyperparameter_tuning_all_targets.py
└── README.md
```

## Reproducing the Models

Run in this order from the repo root (raw CSVs `weather_observations.csv`, `calendar_events.csv`, `traffic_sensor_log.csv` must be present):

```bash
pip install -r dashboard/requirements.txt
pip install seaborn matplotlib imbalanced-learn xgboost

python full_pipeline.py              # cleans data, merges sources, trains 4 models
python travel_time_fixed_model.py    # replaces travel time model with leakage-free version
python dl_model.py                   # trains the LSTM
python generate_dashboard_data.py    # computes charts data for the dashboard

cd dashboard
streamlit run app.py
```

## Dashboard Pages

1. **Overview** — KPI summary and key methodology callouts
2. **Traffic Volume** — actual vs. predicted scatter, feature importance
3. **Travel Time** — actual vs. predicted scatter, feature importance
4. **Congestion Level** — confusion matrix, feature importance, class distribution
5. **Accident Risk** — ROC curve, model coefficients
6. **Deep Learning (LSTM)** — LSTM vs. Random Forest comparison for traffic volume
7. **Live Predictor** — interactive form for real-time predictions using the trained models

## Limitations & Future Work

- Accident risk prediction is bounded by the features available; incorporating driver behavior data, incident reports, or higher-frequency sensor readings could improve this.
- The dataset spans 5 months (Jan–May 2025); a full annual cycle would better capture seasonal effects.
- Congestion and accident risk models use lagged volume as their primary signal; a spatio-temporal graph-based model (accounting for neighboring road segments) is a natural next step.

## Author

Poonam Baruah
