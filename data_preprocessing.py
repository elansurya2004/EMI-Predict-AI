import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    def load_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    # -----------------------------
    # CONVERT NUMERIC COLUMNS
    # -----------------------------
    def convert_numeric_columns(self):
        print("\nConverting numeric columns...")

        numeric_cols = [
            'age','monthly_salary','years_of_employment','monthly_rent',
            'family_size','dependents','school_fees','college_fees',
            'travel_expenses','groceries_utilities','other_monthly_expenses',
            'current_emi_amount','credit_score','bank_balance',
            'emergency_fund','requested_amount','requested_tenure',
            'max_monthly_emi'
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        print("Numeric conversion done")
        return self.df

    # -----------------------------
    # DATA QUALITY CHECK
    # -----------------------------
    def data_quality_check(self):
        print("\n=== Data Quality Assessment ===")

        print("\nMissing Values:")
        print(self.df.isnull().sum()[self.df.isnull().sum() > 0])

        print(f"\nDuplicate Rows: {self.df.duplicated().sum()}")

        print("\nData Types:")
        print(self.df.dtypes)

        print("\nBasic Statistics:")
        print(self.df.describe())

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    def handle_missing_values(self):
        print("\nHandling missing values...")

        for col in self.df.select_dtypes(include=["float64","int64"]).columns:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        print("Missing values handled")
        return self.df

    # -----------------------------
    # REMOVE DUPLICATES
    # -----------------------------
    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {before - len(self.df)} duplicate rows")
        return self.df

    # -----------------------------
    # ENCODE CATEGORICAL FEATURES
    # -----------------------------
    def encode_categorical_features(self):
        print("\nEncoding categorical features...")

        categorical_cols = [
            'gender','marital_status','education','employment_type',
            'company_type','house_type','existing_loans','emi_scenario'
        ]

        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + "_encoded"] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

        return self.df

    # -----------------------------
    # ENCODE TARGET
    # -----------------------------
    def encode_target(self):
        if 'emi_eligibility' in self.df.columns:
            le = LabelEncoder()
            self.df['emi_eligibility_encoded'] = le.fit_transform(self.df['emi_eligibility'])
            self.label_encoders['emi_eligibility'] = le
        return self.df

    # -----------------------------
    # SPLIT DATA
    # -----------------------------
    def split_data(self):
        print("\nSplitting data...")

        feature_cols = [c for c in self.df.columns if c.endswith("_encoded") and c != "emi_eligibility_encoded"]

        numeric_features = [
            'age','monthly_salary','years_of_employment','monthly_rent',
            'family_size','dependents','school_fees','college_fees',
            'travel_expenses','groceries_utilities','other_monthly_expenses',
            'current_emi_amount','credit_score','bank_balance',
            'emergency_fund','requested_amount','requested_tenure'
        ]

        feature_cols.extend([c for c in numeric_features if c in self.df.columns])

        X = self.df[feature_cols]
        y_class = self.df['emi_eligibility_encoded']
        y_reg = self.df['max_monthly_emi']

        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )

        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test

    # -----------------------------
    # SCALE FEATURES
    # -----------------------------
    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    # -----------------------------
    # FULL PIPELINE
    # -----------------------------
    def preprocess_pipeline(self):
        self.load_data()
        self.convert_numeric_columns()
        self.data_quality_check()
        self.handle_missing_values()
        self.remove_duplicates()
        self.encode_categorical_features()
        self.encode_target()

        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = self.split_data()
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("\n✅ Preprocessing completed successfully")

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_class_train": y_class_train,
            "y_class_test": y_class_test,
            "y_reg_train": y_reg_train,
            "y_reg_test": y_reg_test
        }, self.df, self.label_encoders, self.scaler


# ==================================================
# MAIN EXECUTION
# ==================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "emi_prediction_dataset.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")

    preprocessor = DataPreprocessor(DATA_PATH)
    data_splits, processed_df, encoders, scaler = preprocessor.preprocess_pipeline()

    processed_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nProcessed data saved at: {OUTPUT_PATH}")
