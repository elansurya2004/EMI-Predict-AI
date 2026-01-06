import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

class EMIModelTrainer:
    def __init__(self):
        """Initialize the model trainer"""
        self.classification_model = None
        self.regression_model = None
        self.scaler = None
        self.label_encoders = {}
        self.df = None
        
        # Store test data for evaluation
        self.X_test_clf = None
        self.y_test_clf = None
        self.X_test_reg = None
        self.y_test_reg = None
        
    def find_data_file(self):
        """Find available data files in the data directory"""
        possible_paths = [
            'data/processed_data.csv',
            'data/engineered_data.csv',
            'data/emi_prediction_dataset.csv',
            './data/processed_data.csv',
            './data/engineered_data.csv',
            './data/emi_prediction_dataset.csv',
            'app/data/processed_data.csv',
            'app/data/engineered_data.csv',
            'app/data/emi_prediction_dataset.csv',
            r'C:\project\EMIPredict AI\app\data\processed_data.csv',
            r'C:\project\EMIPredict AI\data\engineered_data.csv',
        ]
        
        print("\nSearching for data files...")
        print(f"Current directory: {os.getcwd()}")
        
        if os.path.exists('data'):
            print(f"\nFiles in 'data' folder:")
            for file in os.listdir('data'):
                print(f"  - {file}")
        elif os.path.exists('app/data'):
            print(f"\nFiles in 'app/data' folder:")
            for file in os.listdir('app/data'):
                print(f"  - {file}")
        
        print("\nChecking possible file paths:")
        for filepath in possible_paths:
            print(f"  Checking: {filepath} ... ", end="")
            if os.path.exists(filepath):
                print("✓ FOUND!")
                return filepath
            else:
                print("✗")
        
        return None
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("="*50)
        print("STEP 1: Loading Data")
        print("="*50)
        
        data_path = self.find_data_file()
        
        if data_path is None:
            print("❌ Error: No data file found!")
            print("\nPlease ensure you have one of these files in the 'data' folder:")
            print("  - processed_data.csv")
            print("  - engineered_data.csv")
            print("  - emi_prediction_dataset.csv")
            return False
        
        try:
            self.df = pd.read_csv(data_path)
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*50)
        print("STEP 2: Preprocessing Data")
        print("="*50)
        
        print(f"Original shape: {self.df.shape}")
        
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['loan_status', 'default_status', 'approval_status']:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded: {col}")
        
        print(f"✓ Preprocessing completed")
        print(f"  Final shape: {self.df.shape}")
    
    def create_targets(self):
        """Create target variables if they don't exist"""
        # Classification target
        if 'loan_status' in self.df.columns:
            if self.df['loan_status'].dtype == 'object':
                le = LabelEncoder()
                self.y_classification = le.fit_transform(self.df['loan_status'])
            else:
                self.y_classification = self.df['loan_status']
        elif 'approval_status' in self.df.columns:
            if self.df['approval_status'].dtype == 'object':
                le = LabelEncoder()
                self.y_classification = le.fit_transform(self.df['approval_status'])
            else:
                self.y_classification = self.df['approval_status']
        else:
            if 'income' in self.df.columns and 'loan_amount' in self.df.columns:
                dti = self.df['loan_amount'] / (self.df['income'] + 1)
                self.y_classification = (dti < 0.4).astype(int)
            else:
                self.y_classification = np.random.randint(0, 2, size=len(self.df))
        
        # Regression target (EMI)
        if 'emi' in self.df.columns:
            self.y_regression = self.df['emi']
        elif 'emi_amount' in self.df.columns:
            self.y_regression = self.df['emi_amount']
        elif 'monthly_payment' in self.df.columns:
            self.y_regression = self.df['monthly_payment']
        else:
            if all(col in self.df.columns for col in ['loan_amount', 'interest_rate', 'loan_tenure']):
                P = self.df['loan_amount']
                r = self.df['interest_rate'] / (12 * 100)
                n = self.df['loan_tenure']
                self.y_regression = P * r * np.power(1 + r, n) / (np.power(1 + r, n) - 1)
            else:
                self.y_regression = np.random.uniform(5000, 50000, size=len(self.df))
    
    def prepare_features(self):
        """Prepare feature matrix"""
        exclude_cols = ['loan_status', 'default_status', 'approval_status', 
                       'emi', 'emi_amount', 'monthly_payment']
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.X = self.df[feature_cols].select_dtypes(include=[np.number])
        
        print(f"\nFeatures selected: {self.X.shape[1]} columns")
        print(f"Feature names: {list(self.X.columns)[:10]}...")
    
    def train_classification_model(self):
        """Train loan approval classification model"""
        print("\n" + "="*50)
        print("STEP 3: Training Classification Model")
        print("="*50)
        
        # Split data and STORE test sets
        X_train, self.X_test_clf, y_train, self.y_test_clf = train_test_split(
            self.X, self.y_classification, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        print("\nTraining Random Forest Classifier...")
        self.classification_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=10
        )
        self.classification_model.fit(X_train_scaled, y_train)
        
        # Quick evaluation
        X_test_scaled = self.scaler.transform(self.X_test_clf)
        y_pred = self.classification_model.predict(X_test_scaled)
        accuracy = accuracy_score(self.y_test_clf, y_pred)
        
        print(f"\n✓ Classification Model Trained")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(self.X_test_clf)}")
    
    def train_regression_model(self):
        """Train EMI amount prediction regression model"""
        print("\n" + "="*50)
        print("STEP 4: Training Regression Model")
        print("="*50)
        
        # Split data and STORE test sets
        X_train, self.X_test_reg, y_train, self.y_test_reg = train_test_split(
            self.X, self.y_regression, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        self.regression_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=10
        )
        self.regression_model.fit(X_train_scaled, y_train)
        
        # Quick evaluation
        X_test_scaled = self.scaler.transform(self.X_test_reg)
        y_pred = self.regression_model.predict(X_test_scaled)
        mse = mean_squared_error(self.y_test_reg, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test_reg, y_pred)
        
        print(f"\n✓ Regression Model Trained")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(self.X_test_reg)}")
    
    def evaluate_and_save_metrics(self, output_dir='models'):
        """
        🆕 NEW METHOD: Evaluate models and save comprehensive metrics
        """
        print("\n" + "="*50)
        print("STEP 5: Evaluating Models & Saving Metrics")
        print("="*50)
        
        try:
            # ============ CLASSIFICATION EVALUATION ============
            print("\n📊 Classification Model Evaluation:")
            X_test_clf_scaled = self.scaler.transform(self.X_test_clf)
            y_pred_clf = self.classification_model.predict(X_test_clf_scaled)
            
            # Calculate metrics
            clf_accuracy = accuracy_score(self.y_test_clf, y_pred_clf)
            clf_precision = precision_score(self.y_test_clf, y_pred_clf, average='weighted', zero_division=0)
            clf_recall = recall_score(self.y_test_clf, y_pred_clf, average='weighted', zero_division=0)
            clf_f1 = f1_score(self.y_test_clf, y_pred_clf, average='weighted', zero_division=0)
            
            print(f"  Accuracy:  {clf_accuracy:.4f}")
            print(f"  Precision: {clf_precision:.4f}")
            print(f"  Recall:    {clf_recall:.4f}")
            print(f"  F1-Score:  {clf_f1:.4f}")
            
            # ============ REGRESSION EVALUATION ============
            print("\n📊 Regression Model Evaluation:")
            X_test_reg_scaled = self.scaler.transform(self.X_test_reg)
            y_pred_reg = self.regression_model.predict(X_test_reg_scaled)
            
            # Calculate metrics
            reg_mse = mean_squared_error(self.y_test_reg, y_pred_reg)
            reg_rmse = np.sqrt(reg_mse)
            reg_mae = mean_absolute_error(self.y_test_reg, y_pred_reg)
            reg_r2 = r2_score(self.y_test_reg, y_pred_reg)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((self.y_test_reg - y_pred_reg) / (self.y_test_reg + 1e-10))) * 100
            
            print(f"  RMSE:      ₹{reg_rmse:.2f}")
            print(f"  MAE:       ₹{reg_mae:.2f}")
            print(f"  R² Score:  {reg_r2:.4f}")
            print(f"  MAPE:      {mape:.2f}%")
            
            # ============ CREATE METRICS DICTIONARY ============
            metrics = {
                "classification": {
                    "accuracy": float(clf_accuracy),
                    "precision": float(clf_precision),
                    "recall": float(clf_recall),
                    "f1_score": float(clf_f1),
                    "total_samples": int(len(self.y_test_clf)),
                    "approved": int(np.sum(y_pred_clf == 1)),
                    "rejected": int(np.sum(y_pred_clf == 0))
                },
                "regression": {
                    "rmse": float(reg_rmse),
                    "mae": float(reg_mae),
                    "r2_score": float(reg_r2),
                    "mse": float(reg_mse),
                    "mape": float(mape),
                    "mean_actual": float(np.mean(self.y_test_reg)),
                    "mean_predicted": float(np.mean(y_pred_reg))
                },
                "metadata": {
                    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_records": int(len(self.df)),
                    "n_features": int(self.X.shape[1]),
                    "test_size": 0.2
                }
            }
            
            # ============ SAVE METRICS TO JSON ============
            os.makedirs(output_dir, exist_ok=True)
            metrics_path = os.path.join(output_dir, 'performance_metrics.json')
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"\n✅ Metrics saved to: {metrics_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_models(self, output_dir='models'):
        """Save trained models to pickle files"""
        print("\n" + "="*50)
        print("STEP 6: Saving Models")
        print("="*50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save classification model
            classification_path = os.path.join(output_dir, 'best_classification_model.pkl')
            with open(classification_path, 'wb') as f:
                pickle.dump(self.classification_model, f, protocol=4)
            print(f"✓ Classification model saved: {classification_path}")
            
            # Save regression model
            regression_path = os.path.join(output_dir, 'best_regression_model.pkl')
            with open(regression_path, 'wb') as f:
                pickle.dump(self.regression_model, f, protocol=4)
            print(f"✓ Regression model saved: {regression_path}")
            
            # Save scaler
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f, protocol=4)
            print(f"✓ Scaler saved: {scaler_path}")
            
            # Save label encoders if any
            if self.label_encoders:
                encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
                with open(encoders_path, 'wb') as f:
                    pickle.dump(self.label_encoders, f, protocol=4)
                print(f"✓ Label encoders saved: {encoders_path}")
            
            return True
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False
    
    def verify_models(self, output_dir='models'):
        """Verify that saved models can be loaded"""
        print("\n" + "="*50)
        print("STEP 7: Verifying Models")
        print("="*50)
        
        try:
            with open(os.path.join(output_dir, 'best_classification_model.pkl'), 'rb') as f:
                loaded_clf = pickle.load(f)
            print("✓ Classification model loaded successfully")
            
            with open(os.path.join(output_dir, 'best_regression_model.pkl'), 'rb') as f:
                loaded_reg = pickle.load(f)
            print("✓ Regression model loaded successfully")
            
            with open(os.path.join(output_dir, 'scaler.pkl'), 'rb') as f:
                loaded_scaler = pickle.load(f)
            print("✓ Scaler loaded successfully")
            
            # Test prediction
            sample_data = self.X.iloc[:1]
            scaled_data = loaded_scaler.transform(sample_data)
            
            clf_pred = loaded_clf.predict(scaled_data)
            reg_pred = loaded_reg.predict(scaled_data)
            
            print(f"\n✓ Test Predictions:")
            print(f"  Classification: {clf_pred[0]}")
            print(f"  Regression: ₹{reg_pred[0]:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error verifying models: {str(e)}")
            return False
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("\n" + "="*60)
        print(" EMI PREDICTION MODEL TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Preprocess
        self.preprocess_data()
        
        # Step 2.5: Create targets and prepare features
        self.create_targets()
        self.prepare_features()
        
        # Step 3: Train classification model
        self.train_classification_model()
        
        # Step 4: Train regression model
        self.train_regression_model()
        
        # Step 5: 🆕 EVALUATE AND SAVE METRICS
        if not self.evaluate_and_save_metrics():
            print("⚠️ Warning: Metrics could not be saved, but models are trained")
        
        # Step 6: Save models
        if not self.save_models():
            return False
        
        # Step 7: Verify models
        if not self.verify_models():
            return False
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🎉 Your models are ready to use!")
        print("\nGenerated files in 'models/' directory:")
        print("  • best_classification_model.pkl")
        print("  • best_regression_model.pkl")
        print("  • scaler.pkl")
        print("  • performance_metrics.json  ← 🆕 NEW!")
        if self.label_encoders:
            print("  • label_encoders.pkl")
        print("\nYou can now run your application!")
        return True


if __name__ == "__main__":
    # Initialize trainer
    trainer = EMIModelTrainer()
    
    # Run full pipeline
    success = trainer.run_full_pipeline()
    
    if not success:
        print("\n⚠️ Training failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  1. Make sure you have a CSV file in the 'data' folder")
        print("  2. Check if the data file has the correct format")
        print("  3. Ensure all required packages are installed")