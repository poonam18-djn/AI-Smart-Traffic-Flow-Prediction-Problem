# FlowCast 🚦

**AI-driven traffic analytics platform** that predicts traffic volume, travel time, congestion level, and accident
risk using machine learning. FlowCast integrates data from traffic sensor logs, weather observations, and 
calendar events to generate accurate forecasts and support intelligent traffic management.

## Overview

FlowCast is an end-to-end machine learning pipeline built to help city planners, traffic authorities, and mobility platforms anticipate road conditions before they happen. By combining historical traffic patterns with real-time and contextual signals — weather, holidays, and events — the platform produces multi-target predictions that can power dashboards, alerting systems, and route optimization tools.

## Key Features

📈 **Traffic Volume Prediction** — Forecasts vehicle counts across road segments and time windows
⏱️ **Travel Time Estimation** — Predicts expected trip duration under current and forecasted conditions
🚧 **Congestion Level Classification** — Categorizes road segments by congestion severity
⚠️ **Accident Risk Scoring** — Estimates the likelihood of incidents based on traffic, weather, and temporal factors
🔗 **Multi-Source Data Integration** — Combines sensor logs, weather data, and calendar/event data into a unified feature set
🧪 **Robust Feature Engineering** — Includes lagged and time-aware features, with explicit safeguards against data leakage

## Data Sources

| Source | Description |
|---|---|
| Traffic Sensor Logs | Historical and near-real-time readings from road/traffic sensors |
| Weather Observations | Temperature, precipitation, visibility, and other conditions affecting road use |
| Calendar Events | Holidays, weekends, and scheduled events that influence traffic patterns |

## Methodology

1. **Data Ingestion & Cleaning** — Merging and aligning multi-source data on time and location keys
2. **Feature Engineering** — Constructing time-based, lagged, and contextual features
3. **Data Leakage Prevention** — Careful handling of lagged/rolling features to ensure no future information leaks into training data
4. **Model Training** — Training separate models/targets for volume, travel time, congestion, and accident risk
5. **Evaluation** — Assessing model performance using appropriate regression/classification metrics
6. **Forecast Generation** — Producing predictions to support downstream traffic management decisions

> **Note:** An early version of this pipeline surfaced a data leakage issue caused by improperly aligned lagged features. This was identified and resolved by correctly shifting lag windows relative to the prediction target, ensuring realistic, leakage-free performance estimates.

## Project Structure

```
flowcast/
├── data/                  # Raw and processed datasets (not committed)
├── notebooks/             # Exploratory analysis and model development notebooks
├── src/                   # Core pipeline code (data processing, feature engineering, models)
├── models/                # Saved/trained model artifacts
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/<your-username>/flowcast.git
cd flowcast
pip install -r requirements.txt
```

### Usage

```bash
# Example: run the training pipeline
python src/train.py

# Example: generate predictions
python src/predict.py --input data/sample_input.csv
```

*(Update the commands above to match your actual scripts/entry points.)*

## Tech Stack

- **Python** — Core language
- **pandas / NumPy** — Data processing
- **scikit-learn** — Machine learning models
- **Jupyter Notebook** — Experimentation and analysis

## Results

*(Add key metrics, charts, or sample forecasts here once finalized.)*

## Future Work

- Real-time data streaming integration
- Deployment via API (e.g., FastAPI) for live predictions
- Interactive dashboard for visualization
- Expanded accident risk modeling with additional external data sources

## Author
Poonam Baruah


