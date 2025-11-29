# Telco Customer Churn Analysis 

## Project Overview

This project combines **data analytics** with **business strategy** to tackle customer churn in the telecommunications industry. While technical analysis forms the foundation, the primary focus is on translating data insights into actionable business decisions that protect revenue and drive growth.

## Technical Implementation 

### Python EDA & Analysis
- **Data Exploration**: Comprehensive analysis of 7,043 customer records with 21 features
- **Funnel Analysis**: Advanced customer retention funnel visualization revealing critical drop-off points across the customer journey (New → Engaged → Loyal → Retained)
- **Business KPIs**: Developed financial metrics focusing on revenue impact and customer lifetime value
- **Customer Segmentation**: Created value tiers, tenure segments, and risk categories
- **Visual Analytics**: Interactive dashboards for stakeholder communication
- **Environment Setup**: Virtual environment with requirements.txt for reproducible analysis

### Machine Learning Model

#### Logistic Regression Model (`logistic_regression_model.py`)
- **Algorithm**: Logistic Regression with L2 regularization for interpretable churn prediction
- **Performance Metrics**:
  - ROC-AUC Score: 0.8403 (excellent discriminatory power)
  - Accuracy: 79.9%
  - Precision (Churn): 64.3%
  - Recall (Churn): 54.8%
  - F1-Score (Churn): 59.2%

- **Top Predictors of Churn** (Feature Importance Ranking):
  1. **Tenure** - Longer customer tenure significantly reduces churn risk
  2. **MonthlyCharges** - Higher monthly charges increase churn probability
  3. **Contract** - Contract type is a major churn driver (month-to-month contracts)
  4. **TotalCharges** - Cumulative charges paid by customer
  5. **PhoneService** - Whether customer has phone service
  6. **OnlineSecurity** - Security service subscription status
  7. **TechSupport** - Technical support service status
  8. **PaperlessBilling** - Billing method preference
  9. **InternetService** - Internet service type
  10. **OnlineBackup** - Backup service subscription

---

## CRITICAL BUSINESS INSIGHTS

### (There are some assumptions in the Python Code - 20% profit margin on services & assignment of tier on 50/100 margin (this is subject to change dependin on the preconditions). ###

### The Revenue Crisis
- **30.5% of monthly revenue** is lost to churn ($139,130 monthly)
- **High-value customers** are leaving at 28% rate - double the acceptable threshold
- **253 high-value customers** at immediate risk of churn
- **$2.86 million** in customer lifetime value currently at risk

### The Profit Opportunity
- **Loyal customers** (37+ months) generate **47.4% of total profit** ($43,219 monthly)
- **Premium services** show strong profitability:
  - Device Protection: $16.96 monthly profit per customer
  - Online Backup: $16.62 monthly profit per customer  
  - Online Security: $15.77 monthly profit per customer

## BUSINESS DECISIONS & ACTION PLAN

#### 1. **High-Value Customer Retention**
```python
Actions:
✓ Senior executive personal outreach to top 50 customers
✓ Custom retention offers: 15-20% discount for 6-month commitment
✓ Priority service routing and dedicated support lines
✓ Monthly business reviews for enterprise accounts
```

#### 2. **Service Monetization Acceleration**
```python
# FOCUS: Premium service adoption
# STRATEGY: Bundle pricing and targeted upsells

Launch Immediately:
• "Essential Security Bundle" (OnlineSecurity + DeviceProtection) - 20% off
• "Complete Protection Suite" (All 3 premium services) - 25% off  
• Sales team incentives: Double commission on premium service sales
```
**Target**: Increase services per customer from 2.1 to 2.8 (Metric found from EDA code)

#### 3. **Contract Stabilization Initiative**
```python
# PROBLEM: Month-to-month contracts driving 60% of churn
# SOLUTION: Aggressive conversion to annual contracts

Incentive Structure:
• 10% discount for 1-year contracts
• 15% discount for 2-year contracts  
• $50 bonus for sales team per conversion
```
**Goal**: Convert targeted month-to-month customers into Year(s) contraact

#### 4. **Payment Method Optimization**
```python
FINDING: Electronic check users have 45% higher churn risk

Auto-Pay Transformation:
• $5 monthly discount for credit card auto-pay
• $8 monthly discount for bank auto-pay  
• One-time $25 bonus for switching to auto-pay
```
**Goal**: Reduce electronic check usage 

#### 5. **Predictive Retention Deployment**
```python
# DEPLOY: ML-driven early warning system
# RESOURCES: 3 FTE retention team + CRM integration

Intervention Triggers:
• 30%+ churn probability → Automated email campaign
• 50%+ churn probability → Personal phone outreach
• 70%+ churn probability → Manager escalation + special offer
```
**Goal**: Reduction in overall churn

#### 6. **Logistic Regression Model Insights & Actions**
```python
# MODEL VALIDATION: Logistic Regression confirms key churn drivers
# CONFIDENCE LEVEL: High (ROC-AUC: 84.0%, Accuracy: 79.9%)

Priority Actions Based on ML Insights:
• **Tenure Building**: Reward customer loyalty milestones (6, 12, 24 months)
• **Contract Conversion**: Target month-to-month customers with personalized offers
• **Service Optimization**: Promote OnlineSecurity and TechSupport adoption
• **Price Sensitivity Management**: Monitor MonthlyCharges impact on churn risk
• **Paperless Billing Incentives**: Encourage electronic billing adoption
```
**Goal**: Data-driven retention strategies validated by ML model

---

***FINAL TAKEAWAY***
- This comprehensive analysis combines EDA-driven insights with ML-powered predictions to create a data-driven churn reduction strategy
- **Logistic Regression Model** provides interpretable insights (79.9% accuracy, 84.0% ROC-AUC) identifying tenure, contract type, and monthly charges as key churn drivers
- Business decisions prioritize high-value customer retention while building long-term loyalty through service optimization and contract stabilization
- The integrated approach ensures both strategic understanding and operational execution for sustainable customer retention

## Project Files
- `IBM_Churn_EDA.py`: Comprehensive exploratory data analysis script with KPI calculations and visualizations
- `logistic_regression_model.py`: Logistic regression model for churn prediction with detailed evaluation and business insights
- `funnel_analysis.py`: Custom customer retention funnel analysis with advanced visualizations
- `requirements.txt`: Python dependencies for reproducible environment setup
- `README.md`: Comprehensive project documentation with business insights and action plans

---
