import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EDAAnalyzer:
    def __init__(self, df):
        """Initialize EDA Analyzer"""
        self.df = df
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("=== Summary Statistics ===\n")
        
        # Numerical summary
        print("Numerical Features Summary:")
        print(self.df.describe())
        
        # Categorical summary
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print("\n\nCategorical Features Summary:")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
    
    def analyze_target_distribution(self):
        """Analyze target variable distributions"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('EMI Eligibility Distribution', 
                          'Max Monthly EMI Distribution')
        )
        
        # Classification target
        if 'emi_eligibility' in self.df.columns:
            eligibility_counts = self.df['emi_eligibility'].value_counts()
            fig.add_trace(
                go.Bar(x=eligibility_counts.index, y=eligibility_counts.values,
                      name='EMI Eligibility', marker_color='indianred'),
                row=1, col=1
            )
        
        # Regression target
        if 'max_monthly_emi' in self.df.columns:
            fig.add_trace(
                go.Histogram(x=self.df['max_monthly_emi'], 
                           name='Max Monthly EMI', marker_color='lightseagreen'),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=False, 
                         title_text="Target Variables Analysis")
        fig.show()
        
        # Statistics
        print("\n=== Target Variable Statistics ===")
        print(f"\nEMI Eligibility Distribution:")
        print(self.df['emi_eligibility'].value_counts(normalize=True) * 100)
        print(f"\nMax Monthly EMI - Mean: {self.df['max_monthly_emi'].mean():.2f}")
        print(f"Max Monthly EMI - Median: {self.df['max_monthly_emi'].median():.2f}")
    
    def analyze_emi_scenarios(self):
        """Analyze different EMI scenarios"""
        if 'emi_scenario' not in self.df.columns:
            return
        
        fig = px.histogram(self.df, x='emi_scenario', 
                          color='emi_eligibility',
                          title='EMI Eligibility by Scenario',
                          barmode='group',
                          height=500)
        fig.show()
        
        # Statistics by scenario
        print("\n=== EMI Scenario Analysis ===")
        scenario_stats = self.df.groupby('emi_scenario').agg({
            'requested_amount': ['mean', 'median'],
            'requested_tenure': ['mean', 'median'],
            'max_monthly_emi': ['mean', 'median']
        }).round(2)
        print(scenario_stats)
    
    def analyze_financial_variables(self):
        """Analyze key financial variables"""
        financial_vars = ['monthly_salary', 'current_emi_amount', 
                         'credit_score', 'bank_balance', 'emergency_fund']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=financial_vars
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for i, var in enumerate(financial_vars):
            if var in self.df.columns:
                row, col = positions[i]
                fig.add_trace(
                    go.Box(y=self.df[var], name=var, marker_color='lightblue'),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, showlegend=False,
                         title_text="Financial Variables Distribution")
        fig.show()
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        # Compute correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=1)
        plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Top correlations with target
        if 'max_monthly_emi' in corr_matrix.columns:
            print("\n=== Top Correlations with Max Monthly EMI ===")
            target_corr = corr_matrix['max_monthly_emi'].sort_values(ascending=False)
            print(target_corr.head(10))
    
    def analyze_demographics(self):
        """Analyze demographic patterns"""
        demographic_vars = ['age', 'gender', 'marital_status', 'education']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=demographic_vars
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=self.df['age'], name='Age', marker_color='salmon'),
            row=1, col=1
        )
        
        # Gender distribution
        if 'gender' in self.df.columns:
            gender_counts = self.df['gender'].value_counts()
            fig.add_trace(
                go.Bar(x=gender_counts.index, y=gender_counts.values,
                      name='Gender', marker_color='lightgreen'),
                row=1, col=2
            )
        
        # Marital status
        if 'marital_status' in self.df.columns:
            marital_counts = self.df['marital_status'].value_counts()
            fig.add_trace(
                go.Bar(x=marital_counts.index, y=marital_counts.values,
                      name='Marital Status', marker_color='lightcoral'),
                row=2, col=1
            )
        
        # Education
        if 'education' in self.df.columns:
            edu_counts = self.df['education'].value_counts()
            fig.add_trace(
                go.Bar(x=edu_counts.index, y=edu_counts.values,
                      name='Education', marker_color='skyblue'),
                row=2, col=2
            )
        
        fig.update_layout(height=700, showlegend=False,
                         title_text="Demographic Analysis")
        fig.show()
    
    def income_vs_emi_analysis(self):
        """Analyze relationship between income and EMI"""
        fig = px.scatter(self.df, 
                        x='monthly_salary', 
                        y='max_monthly_emi',
                        color='emi_eligibility',
                        title='Monthly Salary vs Max EMI',
                        labels={'monthly_salary': 'Monthly Salary (INR)',
                               'max_monthly_emi': 'Max Monthly EMI (INR)'},
                        height=500)
        fig.show()
        
        # Calculate EMI to Income ratio
        self.df['emi_to_income_ratio'] = (
            self.df['max_monthly_emi'] / self.df['monthly_salary'] * 100
        )
        
        print("\n=== EMI to Income Ratio Analysis ===")
        print(f"Mean Ratio: {self.df['emi_to_income_ratio'].mean():.2f}%")
        print(f"Median Ratio: {self.df['emi_to_income_ratio'].median():.2f}%")
        
        by_eligibility = self.df.groupby('emi_eligibility')['emi_to_income_ratio'].mean()
        print("\nBy Eligibility:")
        print(by_eligibility)
    
    def credit_score_analysis(self):
        """Analyze credit score patterns"""
        if 'credit_score' not in self.df.columns:
            return
        
        fig = px.box(self.df, 
                    x='emi_eligibility', 
                    y='credit_score',
                    color='emi_eligibility',
                    title='Credit Score Distribution by EMI Eligibility',
                    height=500)
        fig.show()
        
        print("\n=== Credit Score by Eligibility ===")
        credit_stats = self.df.groupby('emi_eligibility')['credit_score'].describe()
        print(credit_stats)
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Insight 1: Approval rates
        approval_rate = (
            self.df['emi_eligibility'].value_counts(normalize=True) * 100
        )
        print(f"\n1. APPROVAL RATES:")
        print(approval_rate)
        
        # Insight 2: Average loan amounts by scenario
        print(f"\n2. AVERAGE LOAN AMOUNTS BY SCENARIO:")
        avg_amounts = self.df.groupby('emi_scenario')['requested_amount'].mean()
        print(avg_amounts.sort_values(ascending=False))
        
        # Insight 3: High-risk segments
        print(f"\n3. HIGH-RISK INDICATORS:")
        high_risk = self.df[self.df['emi_eligibility'] == 'High_Risk']
        print(f"Average Credit Score: {high_risk['credit_score'].mean():.2f}")
        print(f"Average EMI to Income Ratio: {high_risk['emi_to_income_ratio'].mean():.2f}%")
        
        # Insight 4: Employment patterns
        print(f"\n4. EMPLOYMENT TYPE DISTRIBUTION:")
        emp_dist = self.df.groupby('employment_type')['emi_eligibility'].value_counts(normalize=True)
        print(emp_dist * 100)
    
    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        print("Starting Comprehensive EDA...\n")
        
        self.generate_summary_statistics()
        self.analyze_target_distribution()
        self.analyze_emi_scenarios()
        self.analyze_financial_variables()
        self.correlation_analysis()
        self.analyze_demographics()
        self.income_vs_emi_analysis()
        self.credit_score_analysis()
        self.generate_business_insights()
        
        print("\n✓ EDA completed successfully!")


# Usage Example
if __name__ == "__main__":
    df = pd.read_csv('C:\project\EMIPredict AI\data\processed_data.csv')
    eda = EDAAnalyzer(df)
    eda.run_complete_eda()