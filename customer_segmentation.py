# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('mall_customer.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values if any
df = df.dropna()

# Identify ID column (if exists) and exclude it
id_columns = [col for col in df.columns if 'id' in col.lower() or 'cust' in col.lower()]
print("\nIdentified ID columns to exclude:", id_columns)

# Define numeric and categorical features
numeric_features = ['Age', 'IncomeLevel', 'ElectronicsSpending', 'ClothingSpending',
                   'GrocerySpending', 'HomeSpending', 'Visits', 'PurchaseFrequency',
                   'OnlineActivity', 'EmailOpens', 'AppUsage', 'LoyaltyPoints']

categorical_features = ['Gender', 'MembershipLevel']

# Verify all feature columns exist in the dataset
numeric_features = [col for col in numeric_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]

print("\nNumeric features found:", numeric_features)
print("Categorical features found:", categorical_features)

# Data Preprocessing
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Create a copy for preprocessing, excluding ID columns
df_for_clustering = df.drop(columns=id_columns, errors='ignore')

# One-hot encode categorical features
df_encoded = pd.get_dummies(df_for_clustering, columns=categorical_features, drop_first=True)
print("\nAfter one-hot encoding, shape:", df_encoded.shape)

# Check data types and identify any remaining non-numeric columns
print("\nData types after encoding:")
print(df_encoded.dtypes)

# Keep only numeric columns for clustering
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
df_encoded = df_encoded[numeric_cols]
print("\nNumeric columns selected for clustering:", numeric_cols)
print("Final shape:", df_encoded.shape)

# Standardize all numeric features
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_encoded),
    columns=df_encoded.columns,
    index=df_encoded.index
)

print("\nStandardization completed!")
print("Sample of standardized data:")
print(df_scaled.head())

# Prepare features for clustering (use all columns from df_scaled)
X = df_scaled.values
print(f"\nFeature matrix shape for clustering: {X.shape}")
print(f"Data type of X: {X.dtype}")

# K-Means Clustering for different k values
print("\n" + "="*80)
print("K-MEANS CLUSTERING")
print("="*80)

k_values = [2, 3, 4]
results = {}

for k in k_values:
    print(f"\n{'='*50}")
    print(f"CLUSTERING WITH K={k}")
    print(f"{'='*50}")

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Add cluster labels to original dataframe
    df[f'Cluster_k{k}'] = clusters

    # Calculate inertia (within-cluster sum of squares)
    inertia = kmeans.inertia_
    print(f"\nInertia (WCSS): {inertia:.2f}")

    # Store results
    results[k] = {
        'kmeans': kmeans,
        'clusters': clusters,
        'inertia': inertia
    }

    # Analyze each cluster
    print(f"\n{'='*50}")
    print(f"CLUSTER ANALYSIS (k={k})")
    print(f"{'='*50}")

    for cluster_id in range(k):
        cluster_data = df[df[f'Cluster_k{k}'] == cluster_id]
        n_customers = len(cluster_data)

        print(f"\n--- CLUSTER {cluster_id} ---")
        print(f"Number of Customers: {n_customers} ({n_customers/len(df)*100:.1f}%)")

        # Average spending across product categories
        print("\nAverage Spending by Category:")
        print(f"  Electronics: ${cluster_data['ElectronicsSpending'].mean():.2f}")
        print(f"  Clothing: ${cluster_data['ClothingSpending'].mean():.2f}")
        print(f"  Grocery: ${cluster_data['GrocerySpending'].mean():.2f}")
        print(f"  Home: ${cluster_data['HomeSpending'].mean():.2f}")
        total_spending = (cluster_data['ElectronicsSpending'] +
                         cluster_data['ClothingSpending'] +
                         cluster_data['GrocerySpending'] +
                         cluster_data['HomeSpending']).mean()
        print(f"  Total Average Spending: ${total_spending:.2f}")

        # Average income level
        print(f"\nAverage Income Level: ${cluster_data['IncomeLevel'].mean():.2f}")

        # Average loyalty points
        print(f"Average Loyalty Points: {cluster_data['LoyaltyPoints'].mean():.2f}")

        # Additional insights
        print(f"\nAdditional Metrics:")
        print(f"  Average Age: {cluster_data['Age'].mean():.1f} years")
        print(f"  Average Visits: {cluster_data['Visits'].mean():.1f}")
        print(f"  Average Purchase Frequency: {cluster_data['PurchaseFrequency'].mean():.2f}")
        print(f"  Average Online Activity: {cluster_data['OnlineActivity'].mean():.1f}")
        print(f"  Average App Usage: {cluster_data['AppUsage'].mean():.1f}")

        # Gender distribution
        if 'Gender' in cluster_data.columns:
            print(f"\nGender Distribution:")
            gender_dist = cluster_data['Gender'].value_counts()
            for gender, count in gender_dist.items():
                print(f"  {gender}: {count} ({count/n_customers*100:.1f}%)")

        # Membership level distribution
        if 'MembershipLevel' in cluster_data.columns:
            print(f"\nMembership Level Distribution:")
            membership_dist = cluster_data['MembershipLevel'].value_counts()
            for level, count in membership_dist.items():
                print(f"  {level}: {count} ({count/n_customers*100:.1f}%)")

# Business Insights Summary
print("\n" + "="*80)
print("BUSINESS INSIGHTS SUMMARY")
print("="*80)

print("\nElbow Method - Inertia values for different k:")
for k in k_values:
    print(f"  k={k}: {results[k]['inertia']:.2f}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
Based on the clustering analysis:

1. Optimal Number of Clusters:
   - Use the elbow method to determine the optimal k
   - Look for the 'elbow' point where inertia decrease slows down
   - Consider business interpretability alongside statistical metrics

2. Customer Segment Strategies:
   - High-value segments: Target with premium products and loyalty rewards
   - Medium-value segments: Focus on upselling and cross-selling
   - Low-engagement segments: Design re-engagement campaigns

3. Personalization Opportunities:
   - Tailor marketing messages based on spending patterns
   - Customize product recommendations by segment
   - Adjust communication channels based on online activity

4. Resource Allocation:
   - Allocate marketing budget proportional to segment value
   - Design segment-specific loyalty programs
   - Optimize inventory based on segment preferences
""")

# Visualization
print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Elbow plot
axes[0, 0].plot(k_values, [results[k]['inertia'] for k in k_values], 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0, 0].set_ylabel('Inertia (WCSS)', fontsize=12)
axes[0, 0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Cluster size comparison for k=3
k_chosen = 3
cluster_counts = df[f'Cluster_k{k_chosen}'].value_counts().sort_index()
axes[0, 1].bar(cluster_counts.index, cluster_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 1].set_xlabel('Cluster', fontsize=12)
axes[0, 1].set_ylabel('Number of Customers', fontsize=12)
axes[0, 1].set_title(f'Customer Distribution (k={k_chosen})', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Average spending by cluster
spending_categories = ['ElectronicsSpending', 'ClothingSpending', 'GrocerySpending', 'HomeSpending']
cluster_spending = df.groupby(f'Cluster_k{k_chosen}')[spending_categories].mean()
cluster_spending.plot(kind='bar', ax=axes[1, 0], width=0.8)
axes[1, 0].set_xlabel('Cluster', fontsize=12)
axes[1, 0].set_ylabel('Average Spending ($)', fontsize=12)
axes[1, 0].set_title('Average Spending by Category and Cluster', fontsize=14, fontweight='bold')
axes[1, 0].legend(title='Category', labels=['Electronics', 'Clothing', 'Grocery', 'Home'])
axes[1, 0].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=0)

# 4. Income vs Loyalty Points by cluster
for cluster_id in range(k_chosen):
    cluster_data = df[df[f'Cluster_k{k_chosen}'] == cluster_id]
    axes[1, 1].scatter(cluster_data['IncomeLevel'], cluster_data['LoyaltyPoints'],
                      label=f'Cluster {cluster_id}', alpha=0.6, s=50)
axes[1, 1].set_xlabel('Income Level ($)', fontsize=12)
axes[1, 1].set_ylabel('Loyalty Points', fontsize=12)
axes[1, 1].set_title('Income vs Loyalty Points by Cluster', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'customer_segmentation_analysis.png'")
plt.show()

# Save results to CSV
for k in k_values:
    output_file = f'customer_segments_k{k}.csv'
    df[[col for col in df.columns if not col.startswith('Cluster_') or col == f'Cluster_k{k}']].to_csv(output_file, index=False)
    print(f"\nClustering results for k={k} saved to '{output_file}'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)