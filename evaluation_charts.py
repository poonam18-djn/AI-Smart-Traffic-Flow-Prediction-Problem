"""
Model Evaluation Charts for Traffic Analytics Platform
- Actual vs Predicted scatter plots (Volume, Travel Time - regression)
- Confusion matrix (Congestion Level - multi-class classification)
- ROC curve (Accident Risk - binary classification)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score

traffic = pd.read_csv("traffic_merged.csv", low_memory=False)
traffic['date'] = pd.to_datetime(traffic['date'])
traffic['timestamp'] = pd.to_datetime(traffic['timestamp'])
traffic = traffic.sort_values(['road_id', 'timestamp']).reset_index(drop=True)
traffic['hour'] = traffic['timestamp'].dt.hour
traffic['day_of_week'] = traffic['timestamp'].dt.dayofweek
traffic['is_weekend'] = traffic['day_of_week'].isin([5, 6]).astype(int)
traffic['volume_lag1'] = traffic.groupby('road_id')['traffic_volume'].shift(1)
traffic['volume_lag2'] = traffic.groupby('road_id')['traffic_volume'].shift(2)
traffic['traveltime_lag1'] = traffic.groupby('road_id')['travel_time'].shift(1)
traffic['traveltime_lag2'] = traffic.groupby('road_id')['travel_time'].shift(2)
congestion_order = {'Free-flow': 0, 'Moderate': 1, 'Heavy': 2, 'Severe': 3}


# ===== MODEL 1: VOLUME =====
y1 = traffic['traffic_volume']
keep_cols_1 = ['latitude', 'longitude', 'avg_speed', 'occupancy', 'travel_time',
               'accident_count', 'signal_timing', 'road_capacity', 'hour', 'day_of_week', 'is_weekend',
               'weather_station_id', 'weather_condition', 'temperature', 'rainfall', 'visibility',
               'public_holiday', 'event_flag', 'roadwork_flag']
X1 = traffic[keep_cols_1].copy()
X1 = pd.get_dummies(X1, columns=['weather_station_id', 'weather_condition'], drop_first=True)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model1 = RandomForestRegressor(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
model1.fit(X1_train, y1_train)
pred1 = model1.predict(X1_test)
r2_1 = r2_score(y1_test, pred1)


# ===== MODEL 2: TRAVEL TIME (leakage-free, lagged features) =====
traffic_tt = traffic.dropna(subset=['volume_lag1','volume_lag2','traveltime_lag1','traveltime_lag2']).copy().reset_index(drop=True)
y2 = traffic_tt['travel_time']
keep_cols_2 = ['volume_lag1', 'volume_lag2', 'traveltime_lag1', 'traveltime_lag2',
               'signal_timing', 'road_capacity', 'hour', 'day_of_week', 'is_weekend',
               'latitude', 'longitude', 'weather_station_id', 'weather_condition',
               'temperature', 'rainfall', 'visibility', 'public_holiday', 'event_flag', 'roadwork_flag']
X2 = traffic_tt[keep_cols_2].copy()
X2 = pd.get_dummies(X2, columns=['weather_station_id', 'weather_condition'], drop_first=True)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = RandomForestRegressor(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)
r2_2 = r2_score(y2_test, pred2)


# ===== CHART 1: Actual vs Predicted (Volume + Travel Time) =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sample_idx = np.random.RandomState(42).choice(len(y1_test), min(3000, len(y1_test)), replace=False)
axes[0].scatter(y1_test.values[sample_idx], pred1[sample_idx], alpha=0.3, s=10, color='#3498DB')
lims = [min(y1_test.min(), pred1.min()), max(y1_test.max(), pred1.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
axes[0].set_xlabel('Actual Traffic Volume')
axes[0].set_ylabel('Predicted Traffic Volume')
axes[0].set_title(f'Traffic Volume: Actual vs Predicted (R2={r2_1:.3f})', fontweight='bold')
axes[0].legend()

sample_idx2 = np.random.RandomState(42).choice(len(y2_test), min(3000, len(y2_test)), replace=False)
axes[1].scatter(y2_test.values[sample_idx2], pred2[sample_idx2], alpha=0.3, s=10, color='#2ECC71')
lims2 = [min(y2_test.min(), pred2.min()), max(y2_test.max(), pred2.max())]
axes[1].plot(lims2, lims2, 'r--', linewidth=1.5, label='Perfect prediction')
axes[1].set_xlabel('Actual Travel Time (min)')
axes[1].set_ylabel('Predicted Travel Time (min)')
axes[1].set_title(f'Travel Time: Actual vs Predicted (R2={r2_2:.3f})', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()


# ===== MODEL 3: CONGESTION =====
traffic_c = traffic[traffic['congestion_level'] != 'Unknown'].copy().dropna(subset=['volume_lag1','volume_lag2']).reset_index(drop=True)
y3 = traffic_c['congestion_level'].map(congestion_order)
keep_cols_3 = ['volume_lag1', 'volume_lag2', 'signal_timing', 'road_capacity',
               'hour', 'day_of_week', 'is_weekend', 'latitude', 'longitude',
               'weather_station_id', 'weather_condition', 'temperature', 'rainfall', 'visibility',
               'public_holiday', 'event_flag', 'roadwork_flag']
X3 = traffic_c[keep_cols_3].copy()
X3 = pd.get_dummies(X3, columns=['weather_station_id', 'weather_condition'], drop_first=True)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)
model3 = RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
model3.fit(X3_train, y3_train)
pred3 = model3.predict(X3_test)
acc3 = accuracy_score(y3_test, pred3)
cm = confusion_matrix(y3_test, pred3)


# ===== MODEL 4: ACCIDENT RISK =====
traffic_a = traffic.dropna(subset=['volume_lag1', 'volume_lag2']).copy().reset_index(drop=True)
y4 = (traffic_a['accident_count'] > 0).astype(int)
keep_cols_4 = ['volume_lag1', 'volume_lag2', 'signal_timing', 'road_capacity',
               'hour', 'day_of_week', 'is_weekend', 'latitude', 'longitude',
               'weather_station_id', 'weather_condition', 'temperature', 'rainfall', 'visibility',
               'public_holiday', 'event_flag', 'roadwork_flag']
X4 = traffic_a[keep_cols_4].copy()
X4 = pd.get_dummies(X4, columns=['weather_station_id', 'weather_condition'], drop_first=True)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42, stratify=y4)
scaler = StandardScaler()
X4_train_s = scaler.fit_transform(X4_train)
X4_test_s = scaler.transform(X4_test)
model4 = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.01)
model4.fit(X4_train_s, y4_train)
proba4 = model4.predict_proba(X4_test_s)[:, 1]
auc4 = roc_auc_score(y4_test, proba4)
fpr, tpr, _ = roc_curve(y4_test, proba4)


# ===== CHART 2: Confusion Matrix + ROC Curve =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

labels = ['Free-flow', 'Moderate', 'Heavy', 'Severe']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Congestion Level: Confusion Matrix (Accuracy={acc3:.3f})', fontweight='bold')

axes[1].plot(fpr, tpr, color='#E74C3C', linewidth=2, label=f'Logistic Regression (AUC={auc4:.3f})')
axes[1].plot([0,1],[0,1], 'k--', linewidth=1, label='Random guessing (AUC=0.5)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Accident Risk: ROC Curve', fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#E74C3C')

plt.tight_layout()
plt.savefig('classification_eval.png', dpi=150, bbox_inches='tight')
plt.show()

print("All evaluation charts saved: actual_vs_predicted.png, classification_eval.png")
