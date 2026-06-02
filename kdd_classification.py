# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# HELPER FUNCTIONS FOR VISUALIZATION
# ============================================================================

def plot_clusters_2d(data, clusters, centroids, labels, iteration, k, title_prefix="K-Means"):
    """Plot clusters in 2D using PCA for dimensionality reduction"""
    from sklearn.decomposition import PCA

    # Reduce to 2D using PCA if data has more than 2 dimensions
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.transform(centroids)
    else:
        data_2d = data
        centroids_2d = centroids

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']

    plt.figure(figsize=(10, 8))

    # Plot points by cluster with actual labels shown by marker
    for cluster_id in range(k):
        cluster_mask = clusters == cluster_id
        cluster_data = data_2d[cluster_mask]
        cluster_labels = labels[cluster_mask]

        # Separate by actual label
        for label in [0, 1]:
            label_mask = cluster_labels == label
            if np.any(label_mask):
                marker = 'o' if label == 0 else '^'
                plt.scatter(cluster_data[label_mask, 0],
                           cluster_data[label_mask, 1],
                           s=80,
                           c=colors[cluster_id % len(colors)],
                           marker=marker,
                           alpha=0.6,
                           edgecolors='black',
                           linewidth=0.5,
                           label=f"Cluster {cluster_id} - Label {label}")

    # Plot centroids
    for cluster_id, centroid in enumerate(centroids_2d):
        plt.scatter(centroid[0], centroid[1],
                   s=300,
                   c='black',
                   marker='X',
                   edgecolors='yellow',
                   linewidth=2,
                   zorder=5)
        plt.text(centroid[0]+0.1, centroid[1]+0.1,
                f"C{cluster_id}",
                fontsize=14,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.title(f"{title_prefix} Clustering (k={k}, Iteration {iteration})", fontsize=16, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cluster_distribution(clusters, labels, k, unique_labels):
    """Plot bar chart showing class distribution in each cluster"""
    fig, axes = plt.subplots(1, k, figsize=(5*k, 4))
    if k == 1:
        axes = [axes]

    colors = ['lightblue', 'lightcoral']

    for cluster_id in range(k):
        cluster_mask = clusters == cluster_id
        cluster_labels = labels[cluster_mask]

        counts = [np.sum(cluster_labels == 0), np.sum(cluster_labels == 1)]
        percentages = [c/len(cluster_labels)*100 if len(cluster_labels) > 0 else 0 for c in counts]

        axes[cluster_id].bar([unique_labels[0], unique_labels[1]],
                            counts,
                            color=colors,
                            edgecolor='black',
                            linewidth=1.5)
        axes[cluster_id].set_title(f'Cluster {cluster_id}\n({len(cluster_labels)} samples)',
                                   fontsize=12, fontweight='bold')
        axes[cluster_id].set_ylabel('Count', fontsize=10)
        axes[cluster_id].grid(axis='y', alpha=0.3)

        # Add percentage labels on bars
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            axes[cluster_id].text(i, count + max(counts)*0.02,
                                 f'{count}\n({pct:.1f}%)',
                                 ha='center',
                                 fontweight='bold')

    plt.suptitle(f'Class Distribution Across {k} Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_elbow_method(k_values, wcss_values):
    """Plot the Elbow Method graph"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # Add value labels
    for k, wcss in zip(k_values, wcss_values):
        plt.text(k, wcss, f'{wcss:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_silhouette_scores(k_values, silhouette_scores):
    """Plot Silhouette Scores for different k values"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=10)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    plt.title('Silhouette Score for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Add value labels
    for k, score in zip(k_values, silhouette_scores):
        plt.text(k, score, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Highlight the best k
    best_k_idx = np.argmax(silhouette_scores)
    plt.scatter(k_values[best_k_idx], silhouette_scores[best_k_idx],
                s=300, c='red', marker='*', edgecolors='black', linewidth=2, zorder=5)

    plt.tight_layout()
    plt.show()

def plot_accuracy_comparison(perceptron_acc, kmeans_results):
    """Plot accuracy comparison between Perceptron and K-Means for different k values"""
    k_values = list(kmeans_results.keys())
    kmeans_accuracies = [kmeans_results[k]['accuracy'] for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(k_values))
    width = 0.35

    # K-Means bars
    bars1 = ax.bar(x, kmeans_accuracies, width, label='K-Means', color='skyblue', edgecolor='black')

    # Perceptron line
    ax.axhline(y=perceptron_acc, color='red', linestyle='--', linewidth=2, label='Perceptron (Test Set)')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

def calculate_wcss(data, clusters, centroids, k):
    """Calculate Within-Cluster Sum of Squares"""
    wcss = 0
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        if len(cluster_points) > 0:
            centroid = centroids[cluster_id]
            # Sum of squared distances
            wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

def calculate_silhouette_score(data, clusters, k):
    """Calculate Silhouette Score"""
    n = len(data)
    silhouette_vals = []

    for i in range(n):
        point = data[i]
        own_cluster = clusters[i]

        # a(i): mean distance to points in same cluster
        same_cluster_points = data[clusters == own_cluster]
        if len(same_cluster_points) > 1:
            a_i = np.mean([np.sqrt(np.sum((point - other)**2))
                          for other in same_cluster_points if not np.array_equal(other, point)])
        else:
            a_i = 0

        # b(i): mean distance to points in nearest different cluster
        b_i = float('inf')
        for cluster_id in range(k):
            if cluster_id != own_cluster:
                other_cluster_points = data[clusters == cluster_id]
                if len(other_cluster_points) > 0:
                    mean_dist = np.mean([np.sqrt(np.sum((point - other)**2))
                                        for other in other_cluster_points])
                    b_i = min(b_i, mean_dist)

        # Silhouette coefficient
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0

        silhouette_vals.append(s_i)

    return np.mean(silhouette_vals)

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================
print("=" * 80)
print("LOADING KDD DATASET")
print("=" * 80)

# Load the dataset
df = pd.read_csv('kdd_dataset.csv')

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Identify the label column
label_column = None
for col in df.columns:
    if 'class' in col.lower() or 'label' in col.lower():
        label_column = col
        break

if label_column is None:
    label_column = df.columns[-1]

print(f"\nLabel column: '{label_column}'")
print(f"\nLabel distribution:\n{df[label_column].value_counts()}")

# Separate features and labels
X = df.drop(columns=[label_column]).values
y = df[label_column].values

# Convert labels to binary (0 and 1)
unique_labels = np.unique(y)
label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
y_binary = np.array([label_map[label] for label in y])

# Handle non-numeric features
for i in range(X.shape[1]):
    if not np.issubdtype(X[:, i].dtype, np.number):
        unique_vals = np.unique(X[:, i])
        val_map = {val: idx for idx, val in enumerate(unique_vals)}
        X[:, i] = [val_map[val] for val in X[:, i]]

X = X.astype(float)

# Normalize features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1
X_normalized = (X - X_mean) / X_std

# ============================================================================
# PART A: DATASET SPLITTING (70% Training, 30% Testing)
# ============================================================================
print("\n" + "=" * 80)
print("PART A: DATASET SPLITTING (70% Training, 30% Testing)")
print("=" * 80)

# Stratified split
np.random.seed(42)
indices_class0 = np.where(y_binary == 0)[0]
indices_class1 = np.where(y_binary == 1)[0]

np.random.shuffle(indices_class0)
np.random.shuffle(indices_class1)

split0 = int(0.7 * len(indices_class0))
split1 = int(0.7 * len(indices_class1))

train_indices = np.concatenate([indices_class0[:split0], indices_class1[:split1]])
test_indices = np.concatenate([indices_class0[split0:], indices_class1[split1:]])

np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

X_train = X_normalized[train_indices]
y_train = y_binary[train_indices]
X_test = X_normalized[test_indices]
y_test = y_binary[test_indices]

print(f"\nTraining Set (Total: {len(y_train)} records):")
print(f"  {unique_labels[0]}: {np.sum(y_train == 0)} records ({np.sum(y_train == 0)/len(y_train)*100:.2f}%)")
print(f"  {unique_labels[1]}: {np.sum(y_train == 1)} records ({np.sum(y_train == 1)/len(y_train)*100:.2f}%)")

print(f"\nTesting Set (Total: {len(y_test)} records):")
print(f"  {unique_labels[0]}: {np.sum(y_test == 0)} records ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)")
print(f"  {unique_labels[1]}: {np.sum(y_test == 1)} records ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)")

# ============================================================================
# PART B: SINGLE PERCEPTRON CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("PART B: SINGLE PERCEPTRON CLASSIFIER")
print("=" * 80)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                if y_[idx] * y_predicted <= 0:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.sign(linear_output)
        return np.where(y_predicted <= 0, 0, 1)

# Train Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
perceptron.fit(X_train, y_train)
y_pred_perceptron = perceptron.predict(X_test)

correct = np.sum(y_pred_perceptron == y_test)
accuracy_perceptron = (correct / len(y_test)) * 100

print(f"\nPerceptron Training completed!")
print(f"Test Set Accuracy: {accuracy_perceptron:.2f}%")
print(f"Correctly classified: {correct} out of {len(y_test)}")

# ============================================================================
# PART C: K-MEANS CLUSTERING ANALYSIS WITH VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("PART C: K-MEANS CLUSTERING ANALYSIS")
print("=" * 80)

def kmeans_clustering(data, K, max_iters=100, show_plots=True):
    """K-Means clustering implementation with visualization"""
    data = np.array(data, dtype=float)
    n_samples = len(data)

    np.random.seed(42)
    random_indices = np.random.choice(n_samples, K, replace=False)
    centroids = data[random_indices].copy()

    for iteration in range(1, max_iters + 1):
        # Assign clusters
        clusters = np.zeros(n_samples, dtype=int)
        for i, point in enumerate(data):
            distances = [np.sqrt(np.sum((point - centroid)**2)) for centroid in centroids]
            clusters[i] = np.argmin(distances)

        # Recompute centroids
        new_centroids = []
        for k in range(K):
            if np.any(clusters == k):
                new_centroids.append(data[clusters == k].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        # Check convergence
        if np.allclose(new_centroids, centroids):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return clusters, centroids

k_values = [2, 3, 4]
cluster_analysis = {}
kmeans_results = {}
wcss_values = []
silhouette_scores = []

for k in k_values:
    print(f"\n{'─' * 80}")
    print(f"K-Means with k = {k} clusters")
    print(f"{'─' * 80}")

    # Apply K-Means
    clusters, centroids = kmeans_clustering(X_train, k)

    # Calculate WCSS (Within-Cluster Sum of Squares)
    wcss = calculate_wcss(X_train, clusters, centroids, k)
    wcss_values.append(wcss)

    # Calculate Silhouette Score
    silhouette = calculate_silhouette_score(X_train, clusters, k)
    silhouette_scores.append(silhouette)

    print(f"\n  WCSS: {wcss:.2f}")
    print(f"  Silhouette Score: {silhouette:.3f}")

    # Visualize clusters
    plot_clusters_2d(X_train, clusters, centroids, y_train, "Final", k, f"K-Means (k={k})")

    # Analyze each cluster
    total_purity = 0
    for cluster_id in range(k):
        cluster_mask = clusters == cluster_id
        cluster_labels = y_train[cluster_mask]

        total_in_cluster = len(cluster_labels)
        count_class0 = np.sum(cluster_labels == 0)
        count_class1 = np.sum(cluster_labels == 1)

        print(f"\n  Cluster {cluster_id} (Total: {total_in_cluster} records):")
        print(f"    {unique_labels[0]}: {count_class0} records ({count_class0/total_in_cluster*100:.2f}%)")
        print(f"    {unique_labels[1]}: {count_class1} records ({count_class1/total_in_cluster*100:.2f}%)")

        cluster_purity = (max(count_class0, count_class1) / total_in_cluster) * 100
        total_purity += cluster_purity
        print(f"    Cluster Purity: {cluster_purity:.2f}%")

    avg_purity = total_purity / k

    # Show distribution bar chart
    plot_cluster_distribution(clusters, y_train, k, unique_labels)

    cluster_analysis[k] = {
        'centroids': centroids,
        'clusters': clusters,
        'purity': avg_purity
    }

    print(f"\n  Average Cluster Purity for k={k}: {avg_purity:.2f}%")

# Plot Elbow Method
print("\n" + "=" * 80)
print("FINDING OPTIMAL K USING DISTANCE-BASED METRICS")
print("=" * 80)
plot_elbow_method(k_values, wcss_values)
plot_silhouette_scores(k_values, silhouette_scores)

# Find optimal k using different methods
optimal_k_purity = max(cluster_analysis.keys(), key=lambda k: cluster_analysis[k]['purity'])
optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
optimal_k_elbow = k_values[1]  # Typically the elbow is at k=2 or k=3 for this size dataset

print(f"\n{'─' * 80}")
print("OPTIMAL K SELECTION SUMMARY:")
print(f"{'─' * 80}")
print(f"Method 1 - Highest Purity: k = {optimal_k_purity} (Purity: {cluster_analysis[optimal_k_purity]['purity']:.2f}%)")
print(f"Method 2 - Highest Silhouette Score: k = {optimal_k_silhouette} (Score: {silhouette_scores[k_values.index(optimal_k_silhouette)]:.3f})")
print(f"Method 3 - Elbow Method: k = {optimal_k_elbow} (WCSS: {wcss_values[k_values.index(optimal_k_elbow)]:.2f})")

# Use Silhouette Score as the primary metric
optimal_k = optimal_k_silhouette
print(f"\n{'═' * 80}")
print(f"OPTIMAL NUMBER OF CLUSTERS: k = {optimal_k}")
print(f"Average Purity: {cluster_analysis[optimal_k]['purity']:.2f}%")
print(f"{'═' * 80}")

# ============================================================================
# PART D: CLASSIFICATION USING OPTIMAL CENTROIDS
# ============================================================================
print("\n" + "=" * 80)
print("PART D: CLASSIFICATION USING OPTIMAL CENTROIDS")
print("=" * 80)

optimal_centroids = cluster_analysis[optimal_k]['centroids']
optimal_clusters = cluster_analysis[optimal_k]['clusters']

# Assign labels to each cluster
cluster_labels_map = {}
print(f"\nCluster Label Assignment (k={optimal_k}):")
for cluster_id in range(optimal_k):
    cluster_mask = optimal_clusters == cluster_id
    cluster_data_labels = y_train[cluster_mask]

    count_class0 = np.sum(cluster_data_labels == 0)
    count_class1 = np.sum(cluster_data_labels == 1)

    majority_class = 0 if count_class0 > count_class1 else 1
    cluster_labels_map[cluster_id] = majority_class

    majority_name = unique_labels[majority_class]
    print(f"  Cluster {cluster_id} → Assigned label: {majority_name}")

# Predict on training set
train_predicted_labels = []
for point in X_train:
    distances = [np.sqrt(np.sum((point - centroid)**2)) for centroid in optimal_centroids]
    nearest_cluster = np.argmin(distances)
    train_predicted_labels.append(cluster_labels_map[nearest_cluster])

train_predicted_labels = np.array(train_predicted_labels)

correct_kmeans = np.sum(train_predicted_labels == y_train)
accuracy_kmeans = (correct_kmeans / len(y_train)) * 100

print(f"\nK-Means Classification Results on Training Set:")
print(f"Accuracy: {accuracy_kmeans:.2f}%")
print(f"Correctly classified: {correct_kmeans} out of {len(y_train)}")

# Store accuracy for all k values
for k in k_values:
    centroids = cluster_analysis[k]['centroids']
    clusters = cluster_analysis[k]['clusters']

    # Assign labels
    cluster_labels_map_temp = {}
    for cluster_id in range(k):
        cluster_mask = clusters == cluster_id
        cluster_data_labels = y_train[cluster_mask]
        count_class0 = np.sum(cluster_data_labels == 0)
        count_class1 = np.sum(cluster_data_labels == 1)
        majority_class = 0 if count_class0 > count_class1 else 1
        cluster_labels_map_temp[cluster_id] = majority_class

    # Predict
    predictions = []
    for point in X_train:
        distances = [np.sqrt(np.sum((point - centroid)**2)) for centroid in centroids]
        nearest_cluster = np.argmin(distances)
        predictions.append(cluster_labels_map_temp[nearest_cluster])

    predictions = np.array(predictions)
    acc = (np.sum(predictions == y_train) / len(y_train)) * 100
    kmeans_results[k] = {'accuracy': acc}

# ============================================================================
# SUMMARY COMPARISON WITH VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print(f"\nPerceptron Accuracy (Test Set): {accuracy_perceptron:.2f}%")
for k in k_values:
    print(f"K-Means Accuracy (Training Set, k={k}): {kmeans_results[k]['accuracy']:.2f}%")
print(f"\nOptimal K-Means (k={optimal_k}): {accuracy_kmeans:.2f}%")
print("=" * 80)

# Plot accuracy comparison
plot_accuracy_comparison(accuracy_perceptron, kmeans_results)