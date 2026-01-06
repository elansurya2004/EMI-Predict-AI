import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        """Initialize Feature Engineer"""
        self.df = df.copy()
    
    def create_financial_ratios(self):
        """Create derived financial ratio features"""
        print("Creating financial ratio features...")
        
        # 1. Debt-to-Income Ratio
        self.df['debt_to_income_ratio'] = (
            self.df['current_emi_amount'] / self.df['monthly_salary']
        )
        
        # 2. Expense-to-Income Ratio
        total_expenses = (
            self.df['monthly_rent'] + 
            self.df['school_fees'] + 
            self.df['college_fees'] + 
            self.df['travel_expenses'] + 
            self.df['groceries_utilities'] + 
            self.df['other_monthly_expenses']
        )
        self.df['expense_to_income_ratio'] = total_expenses / self.df['monthly_salary']
        
        # 3. Affordability Ratio
        disposable_income = self.df['monthly_salary'] - total_expenses - self.df['current_emi_amount']
        self.df['affordability_ratio'] = disposable_income / self.df['monthly_salary']
        
        # 4. Savings Rate
        self.df['savings_rate'] = (
            (self.df['monthly_salary'] - total_expenses - self.df['current_emi_amount']) / 
            self.df['monthly_salary']
        )
        
        # 5. Emergency Fund Coverage (months)
        self.df['emergency_fund_months'] = (
            self.df['emergency_fund'] / 
            (total_expenses + self.df['current_emi_amount'] + 1)  # +1 to avoid division by zero
        )
        
        # 6. Liquidity Ratio
        self.df['liquidity_ratio'] = (
            (self.df['bank_balance'] + self.df['emergency_fund']) / 
            self.df['monthly_salary']
        )
        
        print(f"Created 6 financial ratio features")
        return self.df
    
    def create_risk_scoring_features(self):
        """Create risk scoring features"""
        print("Creating risk scoring features...")
        
        # 1. Credit Score Category
        self.df['credit_score_category'] = pd.cut(
            self.df['credit_score'],
            bins=[0, 550, 650, 750, 850],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # 2. Employment Stability Score
        self.df['employment_stability'] = (
            self.df['years_of_employment'] / self.df['age']
        )
        
        # 3. Dependency Burden
        self.df['dependency_burden'] = (
            self.df['dependents'] / self.df['family_size']
        )
        
        # 4. Loan-to-Income Ratio for Requested Amount
        self.df['requested_loan_to_income'] = (
            self.df['requested_amount'] / 
            (self.df['monthly_salary'] * self.df['requested_tenure'])
        )
        
        # 5. Available Income After Expenses
        total_expenses = (
            self.df['monthly_rent'] + 
            self.df['school_fees'] + 
            self.df['college_fees'] + 
            self.df['travel_expenses'] + 
            self.df['groceries_utilities'] + 
            self.df['other_monthly_expenses']
        )
        self.df['available_income'] = (
            self.df['monthly_salary'] - total_expenses - self.df['current_emi_amount']
        )
        
        # 6. Risk Score (composite)
        self.df['composite_risk_score'] = (
            (self.df['credit_score'] / 850) * 0.4 +  # 40% weight
            (1 - self.df['debt_to_income_ratio']) * 0.3 +  # 30% weight
            (self.df['employment_stability']) * 0.2 +  # 20% weight
            (self.df['liquidity_ratio'] / 10) * 0.1  # 10% weight
        )
        
        print(f"Created 6 risk scoring features")
        return self.df
    
    def create_interaction_features(self):
        """Create interaction features between key variables"""
        print("Creating interaction features...")
        
        # 1. Salary × Credit Score
        self.df['salary_credit_interaction'] = (
            self.df['monthly_salary'] * self.df['credit_score'] / 100000
        )
        
        # 2. Age × Employment Years
        self.df['age_employment_interaction'] = (
            self.df['age'] * self.df['years_of_employment']
        )
        
        # 3. Requested Amount × Tenure
        self.df['amount_tenure_interaction'] = (
            self.df['requested_amount'] * self.df['requested_tenure'] / 100000
        )
        
        # 4. Income × Affordability
        self.df['income_affordability_interaction'] = (
            self.df['monthly_salary'] * self.df['affordability_ratio']
        )
        
        # 5. Credit Score × Liquidity
        self.df['credit_liquidity_interaction'] = (
            self.df['credit_score'] * self.df['liquidity_ratio']
        )
        
        print(f"Created 5 interaction features")
        return self.df
    
    def create_categorical_derived_features(self):
        """Create derived categorical features"""
        print("Creating derived categorical features...")
        
        # 1. Age Group
        self.df['age_group'] = pd.cut(
            self.df['age'],
            bins=[0, 30, 40, 50, 100],
            labels=['Young', 'Mid-Career', 'Senior', 'Mature']
        )
        
        # 2. Income Group
        self.df['income_group'] = pd.cut(
            self.df['monthly_salary'],
            bins=[0, 30000, 60000, 100000, 300000],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # 3. Loan Size Category
        self.df['loan_size_category'] = pd.cut(
            self.df['requested_amount'],
            bins=[0, 50000, 200000, 500000, 2000000],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        # 4. Has Dependents
        self.df['has_dependents'] = (self.df['dependents'] > 0).astype(int)
        
        # 5. High Credit Score
        self.df['high_credit_score'] = (self.df['credit_score'] >= 700).astype(int)
        
        # 6. Financial Stress Indicator
        self.df['financial_stress'] = (
            (self.df['debt_to_income_ratio'] > 0.4) | 
            (self.df['expense_to_income_ratio'] > 0.7)
        ).astype(int)
        
        print(f"Created 6 derived categorical features")
        return self.df
    
    def create_monthly_obligation_features(self):
        """Create features related to monthly obligations"""
        print("Creating monthly obligation features...")
        
        # 1. Total Monthly Obligations
        self.df['total_monthly_obligations'] = (
            self.df['monthly_rent'] + 
            self.df['school_fees'] + 
            self.df['college_fees'] + 
            self.df['travel_expenses'] + 
            self.df['groceries_utilities'] + 
            self.df['other_monthly_expenses'] +
            self.df['current_emi_amount']
        )
        
        # 2. Essential vs Discretionary Expenses
        self.df['essential_expenses'] = (
            self.df['monthly_rent'] + 
            self.df['groceries_utilities']
        )
        
        self.df['discretionary_expenses'] = (
            self.df['travel_expenses'] + 
            self.df['other_monthly_expenses']
        )
        
        # 3. Education Expenses Total
        self.df['education_expenses_total'] = (
            self.df['school_fees'] + self.df['college_fees']
        )
        
        # 4. Fixed Obligations Ratio
        self.df['fixed_obligations_ratio'] = (
            (self.df['monthly_rent'] + self.df['current_emi_amount']) / 
            self.df['monthly_salary']
        )
        
        print(f"Created 5 monthly obligation features")
        return self.df
    
    def handle_infinite_values(self):
        """Handle infinite and NaN values created during feature engineering"""
        print("Handling infinite and NaN values...")
        
        # Replace infinite values with NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values in ratio columns with 0
        ratio_cols = [col for col in self.df.columns if 'ratio' in col.lower()]
        for col in ratio_cols:
            self.df[col].fillna(0, inplace=True)
        
        # Fill other NaN values with median
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        print("Infinite and NaN values handled")
        return self.df
    
    def get_feature_importance_ranking(self):
        """Display created features summary"""
        print("\n=== Feature Engineering Summary ===")
        
        new_features = [
            'Financial Ratios (6)',
            'Risk Scoring Features (6)',
            'Interaction Features (5)',
            'Categorical Derived Features (6)',
            'Monthly Obligation Features (5)'
        ]
        
        print("\nCreated Feature Categories:")
        for i, feature in enumerate(new_features, 1):
            print(f"{i}. {feature}")
        
        total_features = self.df.shape[1]
        print(f"\nTotal Features: {total_features}")
        
        return self.df
    
    def apply_all_feature_engineering(self):
        """Apply all feature engineering steps"""
        print("Starting Feature Engineering Pipeline...\n")
        
        self.create_financial_ratios()
        self.create_risk_scoring_features()
        self.create_interaction_features()
        self.create_categorical_derived_features()
        self.create_monthly_obligation_features()
        self.handle_infinite_values()
        self.get_feature_importance_ranking()
        
        print("\n✓ Feature Engineering completed successfully!")
        
        return self.df


# =====================================================
# MAIN EXECUTION (WINDOWS SAFE)
# =====================================================
if __name__ == "__main__":

    INPUT_PATH = r"C:\project\EMIPredict AI\data\processed_data.csv"
    OUTPUT_PATH = r"C:\project\EMIPredict AI\data\engineered_data.csv"

    df = pd.read_csv(INPUT_PATH)

    feature_engineer = FeatureEngineer(df)
    engineered_df = feature_engineer.apply_all_feature_engineering()

    engineered_df.to_csv(OUTPUT_PATH, index=False)

    print("\n📁 Engineered data saved successfully")
    print("📍 Location:", OUTPUT_PATH)