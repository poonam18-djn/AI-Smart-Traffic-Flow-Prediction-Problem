"""
=============================================================================
FLOWCAST - Traffic Analytics Dashboard (dashboard/app.py)
=============================================================================
Streamlit dashboard for the AI-driven traffic analytics platform.
Loads the trained models (Random Forest x3, Logistic Regression, LSTM) and
precomputed evaluation data to present:
  - Model performance overview (all 5 models)
  - Regression diagnostics (Volume, Travel Time)
  - Classification diagnostics (Congestion, Accident Risk)
  - Deep learning comparison (LSTM vs Random Forest for Volume)
  - A live "what-if" predictor using the real trained models

Run with:  streamlit run app.py
Expects these files in the same folder (produced by src/ml_models.py,
src/dl_model.py, and the evaluation scripts):
  model_volume.pkl, model_volume_features.pkl
  model_traveltime_fixed.pkl, model_traveltime_fixed_features.pkl
  model_congestion.pkl, model_congestion_features.pkl
  model_accident.pkl, model_accident_scaler.pkl, model_accident_features.pkl
  model_volume_lstm.keras, model_volume_lstm_scaler.pkl
  dashboard_data.json
=============================================================================
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="FlowCast | Traffic Analytics", layout="wide", page_icon="🚦")

# =============================================================================
# LOAD MODELS + DATA (cached so the app doesn't reload on every interaction)
# =============================================================================

@st.cache_resource
def load_models():
    models = {}
    try:
        models['volume'] = joblib.load('model_volume.pkl')
        models['volume_features'] = joblib.load('model_volume_features.pkl')
        models['traveltime'] = joblib.load('model_traveltime_fixed.pkl')
        models['traveltime_features'] = joblib.load('model_traveltime_fixed_features.pkl')
        models['congestion'] = joblib.load('model_congestion.pkl')
        models['congestion_features'] = joblib.load('model_congestion_features.pkl')
        models['accident'] = joblib.load('model_accident.pkl')
        models['accident_scaler'] = joblib.load('model_accident_scaler.pkl')
        models['accident_features'] = joblib.load('model_accident_features.pkl')
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Make sure all .pkl files are in the same folder as app.py.")
    try:
        import tensorflow as tf
        models['lstm'] = tf.keras.models.load_model('model_volume_lstm.keras')
        models['lstm_scaler'] = joblib.load('model_volume_lstm_scaler.pkl')
    except Exception:
        models['lstm'] = None
    return models


@st.cache_data
def load_dashboard_data():
    with open('dashboard_data.json') as f:
        return json.load(f)


models = load_models()
data = load_dashboard_data()

CONGESTION_LABELS = ['Free-flow', 'Moderate', 'Heavy', 'Severe']
CONGESTION_COLORS = ['#34D399', '#F2B705', '#FB8B24', '#EF4444']


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("FlowCast")
st.sidebar.caption("AI Traffic Analytics Platform")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Traffic Volume",
    "Travel Time",
    "Congestion Level",
    "Accident Risk",
    "Deep Learning (LSTM)",
    "Live Predictor",
])
st.sidebar.markdown("---")
st.sidebar.caption("171,887 sensor readings | 25 roads | Jan-May 2025")


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page == "Overview":
    st.title("🚦 Traffic Analytics Platform - Model Overview")
    st.caption("Predicting traffic volume, travel time, congestion, and accident risk "
               "from sensor, weather, and calendar data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traffic Volume", f"R2 = {data['volume']['r2']:.3f}", "Random Forest")
    c2.metric("Travel Time", f"R2 = {data['traveltime']['r2']:.3f}", "Random Forest")
    c3.metric("Congestion Level", f"Acc = {data['congestion']['accuracy']:.3f}", "Random Forest")
    c4.metric("Accident Risk", f"AUC = {data['accident']['roc_auc']:.3f}", "Logistic Regression")

    st.markdown("---")
    st.subheader("Key methodology notes")
    st.info(
        "**Data leakage was identified and fixed** for Travel Time and Congestion Level: "
        "both were originally near-perfectly predictable (R2/Accuracy > 0.99) because they were "
        "mathematically derived from other columns in the raw data (travel_time = distance/avg_speed; "
        "congestion_level = a threshold on volume/capacity). Both models now use only **lagged "
        "(past) values** to represent a genuine forecasting task."
    )
    st.warning(
        "**Accident Risk is inherently hard to predict** from this data (ROC-AUC ~0.64, close to "
        "the ceiling reached even after adding weather/calendar features and SMOTE). This suggests "
        "accidents are driven substantially by factors outside macro-level sensor/weather monitoring."
    )

    st.markdown("---")
    st.subheader("All 5 models")
    overview_df = pd.DataFrame([
        {"Model": "Traffic Volume", "Algorithm": "Random Forest Regressor", "Metric": "R2", "Score": data['volume']['r2']},
        {"Model": "Travel Time", "Algorithm": "Random Forest Regressor", "Metric": "R2", "Score": data['traveltime']['r2']},
        {"Model": "Congestion Level", "Algorithm": "Random Forest Classifier", "Metric": "Accuracy", "Score": data['congestion']['accuracy']},
        {"Model": "Accident Risk", "Algorithm": "Logistic Regression", "Metric": "ROC-AUC", "Score": data['accident']['roc_auc']},
    ])
    st.dataframe(overview_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: TRAFFIC VOLUME
# =============================================================================
elif page == "Traffic Volume":
    st.title("Traffic Volume Prediction")
    st.caption(f"Random Forest Regressor | R2 = {data['volume']['r2']:.4f} | MAE = {data['volume']['mae']:.1f} vehicles")

    col1, col2 = st.columns([2, 1])
    with col1:
        scatter = pd.DataFrame(data['volume']['scatter'])
        fig = px.scatter(scatter, x='actual', y='predicted', opacity=0.5,
                          labels={'actual': 'Actual Volume', 'predicted': 'Predicted Volume'})
        max_val = max(scatter['actual'].max(), scatter['predicted'].max())
        fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color='red', dash='dash'))
        fig.update_layout(title="Actual vs Predicted", height=450)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        imp = pd.DataFrame(data['volume']['importance'])
        fig2 = px.bar(imp, x='value', y='feature', orientation='h',
                       labels={'value': 'Importance', 'feature': ''})
        fig2.update_layout(title="Feature Importance", height=450, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PAGE: TRAVEL TIME
# =============================================================================
elif page == "Travel Time":
    st.title("Travel Time Prediction")
    st.caption(f"Random Forest Regressor (leakage-free, lagged features) | "
               f"R2 = {data['traveltime']['r2']:.4f} | MAE = {data['traveltime']['mae']:.2f} min")

    col1, col2 = st.columns([2, 1])
    with col1:
        scatter = pd.DataFrame(data['traveltime']['scatter'])
        fig = px.scatter(scatter, x='actual', y='predicted', opacity=0.5,
                          labels={'actual': 'Actual Travel Time (min)', 'predicted': 'Predicted Travel Time (min)'})
        max_val = max(scatter['actual'].max(), scatter['predicted'].max())
        fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color='red', dash='dash'))
        fig.update_layout(title="Actual vs Predicted", height=450)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        imp = pd.DataFrame(data['traveltime']['importance'])
        fig2 = px.bar(imp, x='value', y='feature', orientation='h',
                       labels={'value': 'Importance', 'feature': ''})
        fig2.update_layout(title="Feature Importance", height=450, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PAGE: CONGESTION LEVEL
# =============================================================================
elif page == "Congestion Level":
    st.title("Congestion Level Prediction")
    st.caption(f"Random Forest Classifier (leakage-free, lagged features) | "
               f"Accuracy = {data['congestion']['accuracy']:.4f}")

    col1, col2 = st.columns([2, 1])
    with col1:
        cm = np.array(data['congestion']['confusion_matrix'])
        fig = px.imshow(cm, text_auto=True, x=CONGESTION_LABELS, y=CONGESTION_LABELS,
                         color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(title="Confusion Matrix", height=450)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        imp = pd.DataFrame(data['congestion']['importance'])
        fig2 = px.bar(imp, x='value', y='feature', orientation='h',
                       labels={'value': 'Importance', 'feature': ''})
        fig2.update_layout(title="Feature Importance", height=450, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Class distribution")
    dist = pd.DataFrame({'Class': CONGESTION_LABELS, 'Count': data['congestion']['class_distribution']})
    fig3 = px.bar(dist, x='Class', y='Count', color='Class',
                   color_discrete_sequence=CONGESTION_COLORS)
    st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# PAGE: ACCIDENT RISK
# =============================================================================
elif page == "Accident Risk":
    st.title("Accident Risk Prediction")
    st.caption(f"Logistic Regression (leakage-free, lagged features + weather/calendar) | "
               f"ROC-AUC = {data['accident']['roc_auc']:.4f}")
    st.warning(
        f"Only {data['accident']['accident_count']} accidents out of "
        f"{data['accident']['total_count']:,} records ({data['accident']['accident_count']/data['accident']['total_count']*100:.2f}%). "
        "Accuracy is a misleading metric here - ROC-AUC and recall are used instead."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        roc = pd.DataFrame(data['accident']['roc_curve'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc['fpr'], y=roc['tpr'], mode='lines', name=f"Model (AUC={data['accident']['roc_auc']:.3f})",
                                   line=dict(color='#EF4444', width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random guessing (AUC=0.5)',
                                   line=dict(color='gray', dash='dash')))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        coef = pd.DataFrame(data['accident']['coefficients'])
        colors = ['#EF4444' if v > 0 else '#3498DB' for v in coef['value']]
        fig2 = px.bar(coef, x='value', y='feature', orientation='h',
                       labels={'value': 'Coefficient (impact on risk)', 'feature': ''})
        fig2.update_traces(marker_color=colors)
        fig2.update_layout(title="Top Coefficients (red=raises risk, blue=lowers risk)", height=450,
                             yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)
    st.caption("Note: rain and fog conditions increase predicted accident risk - consistent with real-world expectation.")


# =============================================================================
# PAGE: DEEP LEARNING (LSTM)
# =============================================================================
elif page == "Deep Learning (LSTM)":
    st.title("LSTM vs Random Forest - Traffic Volume")
    st.caption("An LSTM sequence model was trained using only the past 6 readings (3 hours) of "
               "volume, hour-of-day, and weekend flag - no engineered features, no weather/calendar.")

    comp_df = pd.DataFrame([
        {"Model": "Random Forest (full features)", "R2": data['volume']['r2']},
        {"Model": "LSTM (sequence-only, 3hr window)", "R2": 0.8473},
    ])
    fig = px.bar(comp_df, x='Model', y='R2', color='Model', text='R2',
                  color_discrete_sequence=['#2ECC71', '#3498DB'])
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=450, showlegend=False, yaxis_range=[0,1.05])
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The Random Forest still outperforms the LSTM here, but the comparison isn't apples-to-apples: "
        "Random Forest has access to occupancy, avg_speed, road_capacity, and weather/calendar features, "
        "while the LSTM sees **only the raw volume sequence itself** and still reaches R2=0.847 using "
        "temporal pattern learning alone - a meaningful result for a from-scratch sequence model."
    )


# =============================================================================
# PAGE: LIVE PREDICTOR
# =============================================================================
elif page == "Live Predictor":
    st.title("Live Prediction Console")
    st.caption("Enter conditions below to get real predictions from the trained models.")

    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.slider("Hour of day", 0, 23, 8)
        day_of_week = st.selectbox("Day of week", options=list(range(7)),
                                     format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
        is_weekend = 1 if day_of_week >= 5 else 0
    with col2:
        volume_lag1 = st.slider("Traffic volume, 30 min ago", 0, 1500, 400)
        volume_lag2 = st.slider("Traffic volume, 60 min ago", 0, 1500, 380)
        signal_timing = st.slider("Signal timing (seconds)", 20, 70, 40)
    with col3:
        road_capacity = st.slider("Road capacity", 1200, 2400, 1800)
        weather_condition = st.selectbox("Weather condition", ['clear', 'cloudy', 'rain', 'fog', 'overcast'])
        temperature = st.slider("Temperature (C)", -5, 40, 20)

    rainfall = st.slider("Rainfall (mm)", 0.0, 20.0, 0.0)
    visibility = st.slider("Visibility (km)", 0.5, 15.0, 10.0)
    public_holiday = st.checkbox("Public holiday")
    event_flag = st.checkbox("Special event nearby")
    roadwork_flag = st.checkbox("Roadwork nearby")

    if st.button("Run Prediction", type="primary"):
        base_row = {
            'volume_lag1': volume_lag1, 'volume_lag2': volume_lag2,
            'signal_timing': signal_timing, 'road_capacity': road_capacity,
            'hour': hour, 'day_of_week': day_of_week, 'is_weekend': is_weekend,
            'latitude': 28.6, 'longitude': 77.2,
            'temperature': temperature, 'rainfall': rainfall, 'visibility': visibility,
            'public_holiday': int(public_holiday), 'event_flag': int(event_flag),
            'roadwork_flag': int(roadwork_flag),
        }
        for ws in ['WS-NORTH', 'WS-SOUTH']:
            base_row[f'weather_station_id_{ws}'] = 0
        for wc in ['cloudy', 'fog', 'overcast', 'rain']:
            base_row[f'weather_condition_{wc}'] = 1 if weather_condition == wc else 0

        def build_input(feature_list):
            row = {f: base_row.get(f, 0) for f in feature_list}
            return pd.DataFrame([row])[feature_list]

        result_cols = st.columns(4)

        try:
            X_c = build_input(models['congestion_features'])
            pred_c = models['congestion'].predict(X_c)[0]
            proba_c = models['congestion'].predict_proba(X_c)[0]
            result_cols[0].metric("Congestion Level", CONGESTION_LABELS[int(pred_c)])
            result_cols[0].caption(f"Confidence: {proba_c[int(pred_c)]*100:.0f}%")
        except Exception as e:
            result_cols[0].error(f"Congestion model error: {e}")

        try:
            X_a = build_input(models['accident_features'])
            X_a_scaled = models['accident_scaler'].transform(X_a)
            proba_a = models['accident'].predict_proba(X_a_scaled)[0][1]
            # Model uses class_weight='balanced' to handle ~1% accident rate, which shifts
            # raw probabilities away from the true base rate. We show a relative risk level
            # instead of the raw probability, to avoid implying a calibrated real-world chance.
            if proba_a < 0.35:
                risk_label, risk_color = "Low", "normal"
            elif proba_a < 0.55:
                risk_label, risk_color = "Elevated", "off"
            else:
                risk_label, risk_color = "High", "inverse"
            result_cols[1].metric("Accident Risk (relative)", risk_label)
            result_cols[1].caption(f"Model score: {proba_a:.2f} (rebalanced scale, not a real-world probability)")
        except Exception as e:
            result_cols[1].error(f"Accident model error: {e}")

        st.caption(
            "Note: Volume and Travel Time predictors require the full feature set from a live sensor "
            "feed (avg_speed, occupancy, current travel_time) and are best run against real sensor data "
            "rather than manual input - see the Traffic Volume / Travel Time pages for their validated "
            "performance on held-out test data. Accident Risk score reflects relative ranking (the "
            "model was rebalanced to detect the ~1% of records with accidents) rather than a calibrated "
            "real-world probability - use it to compare scenarios, not as an absolute chance."
        )
