import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading dataset...")
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path)

# Data preparation
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Premium services
premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['Premium_Service_Count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Custom Funnel Analysis
print("\n" + "="*60)
print("CUSTOMER RETENTION FUNNEL ANALYSIS")
print("="*60)

# Define funnel stages
stages = [
    'Total Customers',
    'With Internet Service',
    'With Premium Services',
    'Long Tenure (>12 months)',
    'Retained Customers'
]

# Calculate counts (cumulative)
total_customers = len(df)
internet_customers = len(df[df['InternetService'] != 'No'])
premium_customers = len(df[(df['InternetService'] != 'No') & (df['Premium_Service_Count'] > 0)])
long_tenure_customers = len(df[(df['InternetService'] != 'No') & (df['Premium_Service_Count'] > 0) & (df['tenure'] > 12)])
retained_customers = len(df[(df['InternetService'] != 'No') & (df['Premium_Service_Count'] > 0) & (df['tenure'] > 12) & (df['Churn'] == 'No')])

counts = [total_customers, internet_customers, premium_customers, long_tenure_customers, retained_customers]

# Calculate conversion rates
conversion_rates = []
for i in range(len(counts)):
    if i == 0:
        conversion_rates.append(100.0)
    else:
        rate = (counts[i] / counts[i-1]) * 100
        conversion_rates.append(rate)

# Calculate drop-off rates
drop_off_rates = []
for i in range(1, len(counts)):
    drop = ((counts[i-1] - counts[i]) / counts[i-1]) * 100
    drop_off_rates.append(drop)

# Print funnel analysis
print("\nFunnel Stages and Conversion Rates:")
print("-" * 50)
for i, stage in enumerate(stages):
    print(f"{stage}: {counts[i]:,} customers ({conversion_rates[i]:.1f}%)")
    if i > 0:
        print(f"  Drop-off from previous: {drop_off_rates[i-1]:.1f}%")

# Key insights
print("\nKey Insights:")
print(f"• Overall retention rate: {(retained_customers/total_customers)*100:.1f}%")
print(f"• Biggest drop-off: Between '{stages[drop_off_rates.index(max(drop_off_rates))+1]}' and '{stages[drop_off_rates.index(max(drop_off_rates))+2]}' ({max(drop_off_rates):.1f}%)")
print(f"• Internet service adoption: {(internet_customers/total_customers)*100:.1f}%")
print(f"• Premium service uptake: {(premium_customers/total_customers)*100:.1f}%")

# Visualize funnel
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Custom Funnel Chart
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
max_count = max(counts)

# Reverse for top-to-bottom funnel
stages_rev = stages[::-1]
counts_rev = counts[::-1]
colors_rev = colors[::-1]

for i, (stage, count, color) in enumerate(zip(stages_rev, counts_rev, colors_rev)):
    # Calculate width as proportion of max
    width = count / max_count
    # Center the bar
    left = (1 - width) / 2
    ax1.barh(i, width, height=0.6, left=left, color=color, edgecolor='black', linewidth=1)
    # Add count label in center
    ax1.text(0.5, i, f'{count:,}\n({count/max_count*100:.1f}%)', ha='center', va='center', fontweight='bold', fontsize=10)

ax1.set_title('Customer Retention Funnel', fontsize=16, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.5, len(stages)-0.5)
ax1.set_yticks(range(len(stages)))
ax1.set_yticklabels(stages_rev)
ax1.set_xlabel('Proportion of Customers')
ax1.grid(True, alpha=0.3)

# Conversion rates with drop-off arrows
conversion_rev = conversion_rates[::-1]
drop_off_rev = drop_off_rates[::-1] + [0]  # Add 0 for first stage

bars = ax2.barh(range(len(stages_rev)), conversion_rev, color=colors_rev, height=0.6)
ax2.set_title('Conversion Rates & Drop-off Analysis', fontsize=16, fontweight='bold')
ax2.set_xlabel('Conversion Rate (%)')
ax2.set_xlim(0, 110)
ax2.set_yticks(range(len(stages_rev)))
ax2.set_yticklabels(stages_rev)

# Add percentage labels
for i, (bar, rate) in enumerate(zip(bars, conversion_rev)):
    ax2.text(rate + 2, bar.get_y() + bar.get_height()/2, f'{rate:.1f}%', va='center', fontweight='bold')

# Add drop-off annotations
for i in range(len(drop_off_rev)-1):
    drop = drop_off_rev[i]
    if drop > 0:
        ax2.annotate(f'Drop-off: {drop:.1f}%', xy=(conversion_rev[i], i), xytext=(110, i+0.3),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

plt.tight_layout()
plt.show()

# Additional funnel visualization: Simple decreasing bars
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(range(len(stages)), counts, color=colors, width=0.8, edgecolor='black')
ax.set_title('Retention Funnel - Customer Counts', fontsize=16, fontweight='bold')
ax.set_ylabel('Number of Customers')
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, rotation=45, ha='right')

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{count:,}', ha='center', va='bottom', fontweight='bold')

# Add conversion rate annotations
for i in range(1, len(stages)):
    prev_count = counts[i-1]
    curr_count = counts[i]
    conversion = (curr_count / prev_count) * 100
    ax.annotate(f'{conversion:.1f}%', xy=((i-1 + i)/2, (prev_count + curr_count)/2),
               xytext=((i-1 + i)/2, prev_count + 200), ha='center',
               arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10, color='blue')

plt.tight_layout()
plt.show()

# Additional analysis: Churn by funnel stage
print("\nChurn Analysis by Funnel Stage:")
print("-" * 40)

for i, stage in enumerate(stages[:-1]):  # Exclude retained
    if stage == 'Total Customers':
        stage_df = df
    elif stage == 'With Internet Service':
        stage_df = df[df['InternetService'] != 'No']
    elif stage == 'With Premium Services':
        stage_df = df[df['Premium_Service_Count'] > 0]
    elif stage == 'Long Tenure (>12 months)':
        stage_df = df[df['tenure'] > 12]

    churn_rate = (stage_df['Churn'] == 'Yes').mean() * 100
    print(f"{stage}: Churn rate = {churn_rate:.1f}% ({len(stage_df[stage_df['Churn'] == 'Yes'])} customers)")
