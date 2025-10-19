# Telco Customer Churn Analysis 

## Project Overview

This project combines **data analytics** with **business strategy** to tackle customer churn in the telecommunications industry. While technical analysis forms the foundation, the primary focus is on translating data insights into actionable business decisions that protect revenue and drive growth.

## Technical Implementation 

### Python EDA & Analysis
- **Data Exploration**: Comprehensive analysis of 7,043 customer records with 21 features
- **Profit-Centric KPIs**: Developed financial metrics focusing on revenue impact and customer lifetime value
- **Customer Segmentation**: Created value tiers, tenure segments, and risk categories
- **Visual Analytics**: Interactive dashboards for stakeholder communication

### Machine Learning Model
- **Algorithm**: LightGBM for efficient churn prediction
- **Performance**: ROC-AUC score: 0.8417 ; Accuracy 0.7984
- **Feature Importance**: Identified key drivers of churn - 

---

## CRITICAL BUSINESS INSIGHTS

# There are some assumptions in the Python Code - 20% profit margin on services & assignment of tier on 50/100 margin (this is subject to change dependin on the preconditions).

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

---

***FINAL TAKEAWAY***
- The business decisions are based on the EDA mostly and the ML is my own attempt at using LightGBM, the decisions made are mostly to retain the most impactful section of the revenue generation stream (High Value customers) while also keeping the broader scope of development and incentivising customers to be associated with the company for longer durations since those show promise in terms of retention. 




---

