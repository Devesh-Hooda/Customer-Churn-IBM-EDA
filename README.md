# Telco Customer Churn Analysis

## Project Overview
This project creates a data analytics story from customer churn analysis and builds a Tableau dashboard to visualize key insights. Combining Python-based exploratory data analysis (EDA) with machine learning predictions, the project translates data insights into actionable business strategies for revenue protection and growth.

## Data Analytics Story: The Churn Crisis Narrative

### Act 1: Setting the Scene
A telecommunications company faces a **26.5% customer churn rate** across 7,043 customers, representing a **$139,130 monthly revenue loss**. The story examines who is leaving, why, and the financial impact.

### Act 2: Character Development
Customers are segmented by value and tenure: Low-value (≤$50/month), Medium-value ($50-$100), High-value (>$100). Loyal customers (37+ months) prove most profitable, generating **47.4% of total profits**.

### Act 3: The Conflict
Key churn drivers emerge: month-to-month contracts, high monthly charges, and short tenure. Logistic regression model (84.0% ROC-AUC) identifies tenure as the strongest predictor.

### Act 4: Resolution
Strategic interventions focus on early retention, contract upgrades, and premium service adoption to reduce churn and protect revenue.

## Churn Overview

### Python Insights
- Dataset: 7,043 customers, 21 features
- Churn rate: 26.5% (1,869 churned vs 5,174 retained)
- Average tenure: 32.4 months
- Average CLV: $2,280 (calculated from TotalCharges)

### Tableau Insights
- Pie chart shows 26.54% churn vs 73.46% retention
- KPI cards display total customers (7,043), average tenure (32.37 months), average CLV ($4,400)
- Business insight: 1 in 4 customers churn, highlighting scale of retention opportunity

## Revenue Impact Analysis

### Python Insights
- Monthly revenue loss: $139,130 (30.5% of total MRR)
- CLV at risk: $2.86 million from churned customers
- High-value customers (>$100/month) churn at 28% rate
- Loyal customers generate $43,219 monthly profit

### Tableau Insights
- Horizontal bar chart compares MRR: Retained ($351K+) vs Churned ($139K)
- Revenue retained significantly outweighs revenue lost
- Business insight: Churn is a revenue leakage problem requiring immediate attention

## Key Churn Drivers

### Python Insights
- Top predictors: Tenure, MonthlyCharges, Contract type
- Month-to-month contracts drive 60% of churn
- Churn probability decreases with tenure (logistic regression)
- Premium services correlate with lower churn rates

### Tableau Insights
- Stacked bar chart: Month-to-month (highest churn), 1-year (moderate), 2-year (lowest)
- Line chart: Churn probability drops from 54% (0-6 months) to minimal after 5 years
- Business insight: First 6-12 months critical retention window; contract length major lever

## Final Dashboard

![](https://github.com/Devesh-Hooda/Customer-Churn-IBM-EDA/blob/main/Tableau_workbook/Churn%20Dashboard.png)

## Business Recommendations

### Python-Based Actions
- High-value customer retention: Executive outreach, custom discounts (15-20% off)
- Service monetization: Bundle premium services (OnlineSecurity + DeviceProtection) at 20% discount
- Contract conversion: 10-15% discounts for 1-2 year commitments
- Predictive deployment: ML-driven intervention triggers (30%+ probability → email, 50%+ → phone call)

### Tableau-Supported Strategies
- Early engagement focus: Target customers in first year with onboarding support
- Contract incentives: Aggressive month-to-month to annual conversions
- Revenue protection: Prioritize low-tenure, high-value segments
- Visual monitoring: Use dashboard for ongoing churn tracking and intervention

## Technical Implementation

### Python Components
- `IBM_Churn_EDA.py`: EDA with KPI calculations, customer segmentation, profit analysis
- `logistic_regression_model.py`: Churn prediction model (79.9% accuracy, 84.0% ROC-AUC)
- `funnel_analysis.py`: Customer retention funnel visualization
- Assumptions: 20% profit margin on services, tier thresholds ($50/$100 monthly charges)

### Tableau Components
- Interactive dashboard with 4 key visualizations (pie, bars, line charts)
- Workbook file: `Tableau_workbook/ChurnFinal.twbx`
- Dashboard image: `Tableau_workbook/Churn Dashboard.png`
- Focus: Visual storytelling of churn patterns and business impact

## Final Takeaway
This integrated approach combines Python's analytical depth with Tableau's visual clarity to create a compelling churn reduction strategy. The data story reveals that early intervention, contract stability, and premium service adoption are key to protecting revenue and building customer loyalty.

## Project Files
- `IBM_Churn_EDA.py`: Python EDA script
- `logistic_regression_model.py`: ML prediction model
- `funnel_analysis.py`: Funnel analysis script
- `requirements.txt`: Python dependencies
- `Tableau_workbook/Churn Dashboard.png`: Dashboard screenshot
- `Tableau_workbook/ChurnFinal.twbx`: Interactive Tableau workbook
- `README.md`: Project documentation
- **Key Business Insight**: First 6-12 months represent the critical retention window.
