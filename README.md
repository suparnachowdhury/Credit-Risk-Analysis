# Credit Risk Prediction & Analytics Dashboard

#### A comprehensive machine learning project that predicts loan default risk using ensemble methods and visualizes portfolio risk through an interactive Power BI dashboard.


### Project Overview
Financial institutions face significant losses from loan defaults. This project builds a predictive risk assessment system that:

- Classifies loan applications into risk categories (High/Medium/Low)
- Identifies key default drivers through feature importance analysis
- Provides actionable insights through an interactive dashboard
- Achieves 93% accuracy with optimized XGBoost

### Business Impact:

Analyzed 32,581 loan applications worth $312.4M
Identified 7,108 defaults ($77.1M exposure)
21.8% overall default rate with risk-based segmentation


### Key Features
#### Machine Learning Pipeline

- Three-Model Comparison: Logistic Regression, Random Forest, XGBoost
- Hyperparameter Tuning: GridSearchCV with 5-fold cross-validation
- Production-Ready: Sklearn pipelines with preprocessing transformations
- Feature Engineering: Log transformations, binning, one-hot encoding

#### Power BI Dashboard

- 4 Interactive Pages: Overview, Client Insights, Loan Portfolio, Risk Analysis
- Dynamic Filtering: By country, loan grade, risk category
- Risk Segmentation: High (67.1%), Medium (44.1%), Low (12.5%) default rates
- Geographic Analysis: Top 10 cities by loan volume and default risk


###  Dataset
- Source: Credit Risk Dataset (32,581 records, 29 features)
#### Features:

- Demographic: Age, gender, marital status, education, location
- Financial: Income, employment length, debt-to-income ratio
- Loan Details: Amount, grade (A-G), interest rate, term, purpose
- Credit History: Credit history length, default history, past delinquencies, open accounts
- Target Variable: loan_status (0 = Non-default, 1 = Default)

### Exploratory Data Analysis Highlights
#### Key Findings
**1. Income Drives Default Risk**

<$25K income: 53.3% default rate
$100K+: 6.5% default rate
Strong inverse correlation between income and default

**2. Loan Grade is Highly Predictive**

Grade A: 10.8% default
Grade G: 97.1% default
Clear risk stratification across grades

**3. Employment Stability Matters**

0-1 years employment: 28.2% default
10+ years: 16.3% default

**4. Loan Purpose Varies**

Medical loans: highest default rates
Education loans: most popular (6,453 loans)

**5. Geographic Patterns**

Vancouver, BC: 24.2% default (highest)
Edinburgh, Scotland: 23.5%


###  Machine Learning Models
#### Model Performance Comparison
### Model Performance Comparison

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC |
|-------|----------|---------------------|------------------|-------------------|---------|
| Logistic Regression | 87% | 0.77 | 0.57 | 0.65 | - |
| Random Forest | **93%** | **0.97** | 0.71 | 0.82 | Best |
| XGBoost | **93%** | 0.95 | **0.73** | **0.83** | 0.83 |

**Winner: XGBoost** ✅ (Best F1-score and balanced precision-recall trade-off)

---

**Performance Breakdown:**

- **Logistic Regression:** Baseline model with 87% accuracy but poor recall (misses 43% of defaults)
- **Random Forest:** Excellent precision (0.97) but lower recall—prioritizes minimizing false positives
- **XGBoost:** Best overall performance with highest F1-score (0.83) and balanced precision-recall trade-off

**Why XGBoost Won:**
- Highest recall (0.73) → Catches 73% of actual defaults
- Strong precision (0.95) → Only 5% false positive rate
- Best F1-score (0.83) → Optimal balance for business use case
- Handles class imbalance well through gradient boosting

**Best Model Configuration**
pythonXGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.2,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=42
)

### Top 10 Predictive Features

- interest_rate - Strongest predictor of default
- loan_grade_D - High-risk grade indicator
- loan_grade_E - Very high-risk grade
- loan_amount - Larger loans = higher risk
- credit_history_length - Shorter history = higher risk
- default_history_Y - Past defaults predict future defaults
- debt_to_income_ratio - Higher DTI = higher risk
- loan_grade_F - Extremely high-risk grade
- age - Younger borrowers default more
- log_income - Lower income = higher risk


### Power BI Dashboard
Live Report: https://app.powerbi.com/view?r=eyJrIjoiYTQwODJhOWMtYjlhZS00NmM4LWJhYTQtNThhYTU5ZGFhODk3IiwidCI6ImRiMjMwNmZkLWFmMjUtNGUyOS05Y2NiLWJmMjg2YWY2MjFjMCJ9
#### Dashboard Pages
**1. Overview**

Portfolio KPIs: Total loans, default rate, average interest
Loan distribution by purpose, country, risk category
Loan size vs. default rate analysis


**2. Client Insights**

Age group segmentation (18-25 highest risk at ~31%)
Income vs. default correlation
Home ownership impact (Renters: 31.4% default)
Employment length analysis

**3. Loan Portfolio**

Loan grade distribution (Grade A: 10,777 loans)
Debt-to-income ratio vs. defaults
Top 10 cities by loan volume
Geographic default patterns

**4. Risk Analysis**

Risk category breakdown: High (1,193 loans), Medium (7,518), Low (23,870)
Default rate by loan grade and risk category
Loan purpose risk segmentation
Portfolio metrics by risk tier

#### Interactive Features

- Slicers: Country, loan grade, risk category, date range
- Drill-Through: From overview to detailed client/loan analysis
- Tooltips: Contextual metrics on hover
- Cross-Filtering: Click any chart to filter entire dashboard


### Tech Stack
#### Programming & ML:

- Python 3.8+
- scikit-learn (pipelines, preprocessing, models)
- XGBoost (gradient boosting)
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)

#### Business Intelligence:

- Power BI Desktop
- DAX (calculated measures)
- Power Query (data transformation)

#### Development Tools:

- Jupyter Notebook
- Git/GitHub


### Getting Started
Prerequisites
bashPython 3.8+
Jupyter Notebook
Power BI Desktop (for dashboard)

### Installation


1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Credit-Risk-Analysis.git
cd Credit-Risk-Analysis
```

2. **Run the Jupyter Notebook**
```bash
jupyter notebook credit_risk_analysis.ipynb
```

3. **Open Power BI Dashboard**
- Open `credit_risk_dashboard.pbix` in Power BI Desktop
- Refresh data connections if needed

### Project Structure
```
credit-risk-prediction/
│
├── data/
│   └── Credit_Risk_Dataset.xlsx          # Raw dataset
│
├── notebooks/
│   └── credit_risk_analysis.ipynb        # Main analysis notebook

│
├── Report Pages/
│   ├── dashboard_overview.png
│   ├── client_insights.png
│   └── risk_analysis.png
               # Saved model (optional)
│
└── README.md                              # This file



```

## Key Insights & Recommendations

### Business Recommendations

**1. Risk-Based Pricing**
- Increase interest rates for Grade D+ loans (default rate >50%)
- Offer lower rates to high-income ($100K+) borrowers

**2. Targeted Underwriting**
- Require additional verification for <1 year employment
- Scrutinize renters more carefully (31.4% default vs. 7-12% for owners)
- Flag applicants with DTI >50%

**3. Geographic Focus**
- Monitor Vancouver, Glasgow, Edinburgh portfolios (>23% default)
- Expand in low-risk markets (Victoria: 20.8%)

**4. Product Strategy**
- Offer financial counseling for medical loan applicants
- Create income-verified education loan products
- Limit loan amounts for Grade E-G borrowers

**5. Portfolio Rebalancing**
- Reduce exposure to high-risk segment (currently $16.6M)
- Increase low-risk lending (only 12.5% default, $217M portfolio)

---

## Model Limitations & Future Work

### Current Limitations
- Imbalanced dataset (78% non-default, 22% default)
- Missing macroeconomic features (unemployment rate, GDP)
- Static model (doesn't adapt to market changes)

### Future Enhancements

**1. Advanced Techniques**
- SMOTE for handling class imbalance
- Ensemble stacking (combine RF + XGBoost)
- Neural networks for non-linear patterns
- Time-series analysis for economic trends

**2. Feature Engineering**
- Loan-to-value ratio for home buyers
- Credit utilization trends (not just snapshot)
- Industry-specific risk (employment sector)
- Behavioral features (payment patterns)

**3. Model Deployment**
- REST API for real-time predictions
- A/B testing framework for model updates
- Monitoring dashboard for model drift
- Explainable AI (SHAP values for regulatory compliance)

**4. Business Integration**
- Automated approval/rejection workflow
- Risk-based loan amount recommendations
- Early warning system for portfolio deterioration

---

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### 👤 Author
Suparna Chowdhury

GitHub:[ @suparnachowdhury](https://github.com/suparnachowdhury)
LinkedIn: www.linkedin.com/in/suparna-chowdhury
Portfolio: [yourportfolio.com](https://suparnachowdhury.github.io/home)


### Acknowledgments

Dataset: [Onyx Data - September 2025 Challenge]
Inspiration: Real-world credit risk management practices
Tools: scikit-learn, XGBoost, Power BI communities


### Contact
For questions or collaboration opportunities:

Email: suparna.chowdhury.data@gmail.com


⭐ If you found this project helpful, please give it a star!
