import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print(" LOGISTIC REGRESSION MODEL FOR CUSTOMER CHURN PREDICTION")
print("=" * 60)

# Step 1: Load Dataset
print("\nStep 1: Loading Dataset...")

try:
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        main_file = csv_files[0]
        file_path = os.path.join(path, main_file)
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully from {main_file}")
        print(f"✓ Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        print("✗ No CSV files found!")
        exit()
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit()

# Step 2: Data Preprocessing
print("\nStep 2: Data Preprocessing...")

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Remove customerID as it's not useful for prediction
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop('Churn') if 'Churn' in categorical_cols else categorical_cols

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"✓ Data preprocessing completed")
print(f"✓ Encoded {len(categorical_cols)} categorical variables")

# Step 3: Feature Selection and Train-Test Split
print("\nStep 3: Feature Selection and Train-Test Split...")

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"✓ Features: {X.shape[1]}")

# Step 4: Model Training
print("\nStep 4: Training Logistic Regression Model...")

# Initialize and train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("✓ Model trained successfully")

# Step 5: Model Evaluation
print("\nStep 5: Model Evaluation...")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"✓ Accuracy: {accuracy:.4f}")
print(f"✓ ROC-AUC Score: {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
print("\nConfusion Matrix:")
print("-" * 50)
cm = confusion_matrix(y_test, y_pred)
print("Predicted →    No Churn    Churn")
print(f"Actual ↓   {cm[0][0]:8d}  {cm[0][1]:8d}")
print(f"         {cm[1][0]:8d}  {cm[1][1]:8d}")

# Step 6: Feature Importance
print("\nStep 6: Feature Importance Analysis...")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print("-" * 40)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:20s}: {row['importance']:.4f}")

# Step 7: ROC Curve Visualization
print("\nStep 7: Generating ROC Curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression for Churn Prediction')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Step 8: Model Insights
print("\nStep 8: Model Insights and Recommendations...")

print("MODEL PERFORMANCE SUMMARY:")
print("-" * 30)
print(f"• Accuracy: {accuracy:.1%}")
print(f"• ROC-AUC: {roc_auc:.1%}")
print(f"• Precision (Churn): {classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.1%}")
print(f"• Recall (Churn): {classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.1%}")
print(f"• F1-Score (Churn): {classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.1%}")

print("\nTOP PREDICTORS OF CHURN:")
print("-" * 25)
for idx, row in feature_importance.head(5).iterrows():
    print(f"• {row['feature']}")

print("\nBUSINESS RECOMMENDATIONS:")
print("-" * 30)
print("• Focus retention efforts on customers with high values in top predictors")
print("• Implement targeted interventions based on model predictions")
print("• Monitor model performance regularly and retrain as needed")

print("\n" + "=" * 60)
print(" MODEL EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 60)
