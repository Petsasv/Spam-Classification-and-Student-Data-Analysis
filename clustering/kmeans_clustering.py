# Set environment variables to prevent KMeans memory leak on Windows
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Import warning filter to suppress KMeans warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import load_and_preprocess_data, visualize_clusters, analyze_clusters

def elbow_method(X, max_k=10):
    """
    Perform elbow method to determine optimal number of clusters for K-means.
    
    Args:
        X: Input data
        max_k: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    distortions = []
    K = range(1, max_k + 1)
    
    # Calculate distortions for different k values
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Calculate the rate of change of distortion
    rate_of_change = np.diff(distortions)
    rate_of_change = np.append(rate_of_change, rate_of_change[-1])  # Pad with last value
    
    # Find the elbow point (where rate of change starts to level off)
    optimal_k = np.argmax(np.abs(np.diff(rate_of_change))) + 1
    
    print(f"\nElbow method suggests optimal k = {optimal_k}")
    return optimal_k

def get_kmeans_model(n_clusters=3):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    return model

def perform_kmeans(X, n_clusters, feature_names=None, original_df=None):
    """
    Perform K-means clustering
    
    Args:
        X: Input data
        n_clusters: Number of clusters
        feature_names: Names of features
        original_df: Original dataframe
    
    Returns:
        KMeans model, cluster labels
    """
    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")
    
    # Create and fit K-means model
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=10
    )
    kmeans.fit(X)
    labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # Visualize clusters
    visualize_clusters(X, labels, feature_names, title='K-means Clustering', method_name=f'k={n_clusters}')
    
    # Analyze cluster properties
    if original_df is not None:
        analyze_clusters(original_df, labels, method_name=f'K-means (k={n_clusters})')
    
    return kmeans, labels

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Load and preprocess data
    print("Running K-means Clustering Analysis...")
    data = load_and_preprocess_data()
    X_scaled = data['X_scaled']
    X_scaled_df = data['X_scaled_df']
    feature_names = data['feature_names']
    original_df = data['original_df']
    
    # Check data shape and feature types
    print(f"\nData shape: {X_scaled.shape}")
    print(f"Feature names: {feature_names}")
    
    # Perform elbow method to find optimal k
    optimal_k = elbow_method(X_scaled)
    
    # If optimal_k is 1, use a reasonable default (3 clusters)
    if optimal_k == 1:
        print("\nWarning: Elbow method suggested k=1, which is not suitable for clustering.")
        print("Using k=3 as a reasonable default.")
        optimal_k = 3
    
    # Perform K-means with optimal k
    kmeans, kmeans_labels = perform_kmeans(X_scaled, optimal_k, feature_names, original_df)
    
    # Print cluster sizes
    print("\nK-means cluster sizes:")
    unique_kmeans, counts_kmeans = np.unique(kmeans_labels, return_counts=True)
    for cluster, count in zip(unique_kmeans, counts_kmeans):
        print(f"  Cluster {cluster}: {count} samples ({count/len(kmeans_labels)*100:.2f}%)")
    
    print("\nK-means Clustering Analysis Completed!") 