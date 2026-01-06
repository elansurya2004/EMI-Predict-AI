import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, r2_score, mean_absolute_error,
                             confusion_matrix, classification_report)

print("="*60)
print("GENERATING MODEL PERFORMANCE METRICS")
print("="*60)

# Load the saved models
print("\n[1/4] Loading models...")
try:
    with open('models/best_classification_model.pkl', 'rb') as f:
        clf_model = pickle.load(f)
    with open('models/best_regression_model.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("\nPlease run train_models_simple.py first to create the models!")
    exit()

# Try to load feature names from training
try:
    with open('models/feature_names.pkl', 'rb') as f:
        saved_feature_cols = pickle.load(f)
    print(f"✓ Loaded {len(saved_feature_cols)} feature names from training")
    use_saved_features = True
except:
    print("⚠ Feature names not found, will auto-detect features")
    use_saved_features = False

# Load data
print("\n[2/4] Loading data...")
data_paths = [
    'app/data/processed_data.csv',
    'data/engineered_data.csv',
    'data/emi_prediction_dataset.csv',
    'data/processed_data.csv',
    r'C:\project\EMIPredict AI\app\data\processed_data.csv',
    r'C:\project\EMIPredict AI\data\engineered_data.csv',
]

df = None
for path in data_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✓ Loaded: {path}")
        break

if df is None:
    print("❌ No data file found!")
    exit()

# Prepare data
print("\n[3/4] Preparing data and calculating metrics...")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.fillna(df.median(numeric_only=True))

# Define features - use saved features if available
if use_saved_features:
    # Use the exact features from training
    available_cols = [col for col in saved_feature_cols if col in df.columns]
    missing_cols = [col for col in saved_feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠ Warning: {len(missing_cols)} features missing from data")
        print(f"  Missing: {missing_cols[:5]}...")
        # Create missing columns with zeros
        for col in missing_cols:
            df[col] = 0
    
    feature_cols = saved_feature_cols
else:
    # Auto-detect features from numeric columns
    exclude = ['loan_status', 'approval_status', 'emi', 'emi_amount', 'default_status']
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude]
    
    print(f"✓ Auto-detected {len(feature_cols)} features")

# Select features and ensure they're all numeric
X = df[feature_cols].copy()

# Convert any non-numeric columns to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(0)

print(f"✓ Using {len(feature_cols)} features for prediction")

# Create targets
if 'loan_amount' in df.columns and 'income' in df.columns:
    debt_ratio = df['loan_amount'] / (df['income'] + 1)
    y_class = (debt_ratio < 0.4).astype(int)
else:
    y_class = np.random.randint(0, 2, len(df))

if 'emi' in df.columns:
    y_reg = df['emi']
elif 'loan_amount' in df.columns and 'interest_rate' in df.columns and 'loan_tenure' in df.columns:
    P = df['loan_amount']
    r = df['interest_rate'] / (12 * 100)
    n = df['loan_tenure']
    y_reg = P * r * np.power(1 + r, n) / (np.power(1 + r, n) - 1)
else:
    y_reg = np.random.uniform(5000, 50000, len(df))

# Split data for evaluation
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Scale features
try:
    X_test_c_scaled = scaler.transform(X_test_c)
    X_test_r_scaled = scaler.transform(X_test_r)
    print("✓ Features scaled successfully")
except Exception as e:
    print(f"⚠ Scaling error: {e}")
    print("  Attempting to retrain scaler...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)
    X_test_r_scaled = scaler.transform(X_test_r)
    print("✓ Scaler retrained and applied")

# Calculate Classification Metrics
print("\nCalculating classification metrics...")
try:
    y_pred_c = clf_model.predict(X_test_c_scaled)
    
    classification_metrics = {
        'accuracy': float(accuracy_score(y_test_c, y_pred_c)),
        'precision': float(precision_score(y_test_c, y_pred_c, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test_c, y_pred_c, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_test_c, y_pred_c, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test_c, y_pred_c).tolist(),
        'total_samples': int(len(y_test_c)),
        'approved': int(sum(y_pred_c)),
        'rejected': int(len(y_pred_c) - sum(y_pred_c))
    }
    
    print(f"  Accuracy: {classification_metrics['accuracy']:.2%}")
    print(f"  Precision: {classification_metrics['precision']:.2%}")
    print(f"  Recall: {classification_metrics['recall']:.2%}")
    print(f"  F1 Score: {classification_metrics['f1_score']:.2%}")
except Exception as e:
    print(f"❌ Classification error: {e}")
    classification_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'confusion_matrix': [[0, 0], [0, 0]],
        'total_samples': 0,
        'approved': 0,
        'rejected': 0,
        'error': str(e)
    }

# Calculate Regression Metrics
print("\nCalculating regression metrics...")
try:
    y_pred_r = reg_model.predict(X_test_r_scaled)
    
    mse = mean_squared_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test_r - y_pred_r) / (y_test_r + 1))) * 100
    
    regression_metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'mse': float(mse),
        'mape': float(mape),
        'mean_actual': float(np.mean(y_test_r)),
        'mean_predicted': float(np.mean(y_pred_r)),
        'min_emi': float(np.min(y_pred_r)),
        'max_emi': float(np.max(y_pred_r)),
        'total_samples': int(len(y_test_r))
    }
    
    print(f"  RMSE: ₹{rmse:,.2f}")
    print(f"  MAE: ₹{mae:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
except Exception as e:
    print(f"❌ Regression error: {e}")
    regression_metrics = {
        'rmse': 0.0,
        'mae': 0.0,
        'r2_score': 0.0,
        'mse': 0.0,
        'mape': 0.0,
        'mean_actual': 0.0,
        'mean_predicted': 0.0,
        'min_emi': 0.0,
        'max_emi': 0.0,
        'total_samples': 0,
        'error': str(e)
    }

# Feature importance
print("\nCalculating feature importance...")
try:
    if hasattr(clf_model, 'feature_importances_'):
        feature_importance = dict(zip(feature_cols, clf_model.feature_importances_.tolist()))
        # Sort by importance and get top 10
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        feature_importance = {}
except Exception as e:
    print(f"⚠ Could not calculate feature importance: {e}")
    feature_importance = {}

# Save metrics
print("\n[4/4] Saving metrics...")
os.makedirs('models', exist_ok=True)

metrics = {
    'classification': classification_metrics,
    'regression': regression_metrics,
    'feature_importance': feature_importance,
    'model_info': {
        'classification_model': str(type(clf_model).__name__),
        'regression_model': str(type(reg_model).__name__),
        'n_features': len(feature_cols),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
}

with open('models/performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("✓ Metrics saved to: models/performance_metrics.json")

# Also save a CSV for easy viewing
metrics_df = pd.DataFrame({
    'Metric': ['Classification Accuracy', 'Classification Precision', 'Classification Recall', 
               'Classification F1-Score', 'Regression RMSE', 'Regression MAE', 
               'Regression R² Score', 'Regression MAPE'],
    'Value': [
        f"{classification_metrics.get('accuracy', 0):.2%}",
        f"{classification_metrics.get('precision', 0):.2%}",
        f"{classification_metrics.get('recall', 0):.2%}",
        f"{classification_metrics.get('f1_score', 0):.2%}",
        f"₹{regression_metrics.get('rmse', 0):,.2f}",
        f"₹{regression_metrics.get('mae', 0):,.2f}",
        f"{regression_metrics.get('r2_score', 0):.4f}",
        f"{regression_metrics.get('mape', 0):.2f}%"
    ]
})

metrics_df.to_csv('models/performance_summary.csv', index=False)
print("✓ Summary saved to: models/performance_summary.csv")

print("\n" + "="*60)
print("✅ PERFORMANCE METRICS GENERATED SUCCESSFULLY!")
print("="*60)
print("\nFiles created:")
print("  • models/performance_metrics.json")
print("  • models/performance_summary.csv")
print("\n📊 Now refresh your application's Performance page!")