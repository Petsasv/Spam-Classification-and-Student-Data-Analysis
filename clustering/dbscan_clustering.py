import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from utils import load_and_preprocess_data, visualize_clusters, analyze_clusters

def plot_kdistance(X, k=5):
    """
    Plot the k-distance graph to help determine eps parameter for DBSCAN.
    """
    # Calculate k-distance
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances)
    plt.title(f'K-Distance Graph (k={k})')
    plt.xlabel('Points')
    plt.ylabel(f'{k}-Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return distances

def get_dbscan_model(eps=0.5, min_samples=5):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean'
    )
    return model

def perform_dbscan(X, eps, min_samples, feature_names=None, original_df=None):
    """
    Perform DBSCAN clustering
    
    Args:
        X: Input data
        eps: Maximum distance between samples
        min_samples: Minimum number of samples in a neighborhood
        feature_names: Names of features
        original_df: Original dataframe
    
    Returns:
        DBSCAN model, cluster labels
    """
    print(f"\nPerforming DBSCAN clustering with eps={eps:.3f}, min_samples={min_samples}...")
    
    # Create and fit DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Count number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    
    # Calculate silhouette score if there are at least 2 clusters and not all points are noise
    if n_clusters >= 2 and n_noise < len(labels):
        # Calculate silhouette score (ignore noise points)
        mask = labels != -1
        if np.sum(mask) > 1:  # Need at least 2 points for silhouette score
            silhouette = silhouette_score(X[mask], labels[mask])
            print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
    elif n_clusters < 2:
        print("Warning: Less than 2 clusters found. Try a different eps value.")
    
    # Visualize clusters
    visualize_clusters(X, labels, feature_names, title='DBSCAN Clustering', 
                       method_name=f'eps={eps:.3f}, min_samples={min_samples}')
    
    # Analyze cluster properties
    if original_df is not None and n_clusters >= 1:
        analyze_clusters(original_df, labels, method_name=f'DBSCAN (eps={eps:.3f}, min_samples={min_samples})')
    
    return dbscan, labels

def try_multiple_dbscan_parameters(X, feature_names=None, original_df=None):
    """
    Try multiple DBSCAN parameter combinations to find the best one
    
    Args:
        X: Input data
        feature_names: Names of features
        original_df: Original dataframe
    
    Returns:
        Best DBSCAN model, best labels, best parameters
    """
    # Get suggested eps from k-distance graph
    distances = plot_kdistance(X)
    
    # Define multiple parameter combinations to try
    eps_values = [
        np.percentile(distances, 25),
        np.percentile(distances, 50),
        np.percentile(distances, 75)
    ]
    min_samples_values = [5, 10, 15]
    
    best_silhouette = -1
    best_model = None
    best_labels = None
    best_params = None
    
    # Try different parameter combinations
    print("\nTrying different DBSCAN parameter combinations...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Count number of clusters and noise points
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Skip if less than 2 clusters or too many noise points
            if n_clusters < 2 or n_noise >= len(labels) * 0.5:
                print(f"  eps={eps:.3f}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points (skipping)")
                continue
            
            # Calculate silhouette score (ignore noise points)
            mask = labels != -1
            if np.sum(mask) > 1:
                silhouette = silhouette_score(X[mask], labels[mask])
                print(f"  eps={eps:.3f}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points, silhouette={silhouette:.4f}")
                
                # Update best if better
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_model = dbscan
                    best_labels = labels
                    best_params = {'eps': eps, 'min_samples': min_samples}
    
    # If no good combination found, use default
    if best_model is None:
        print("\nNo good parameter combination found. Using suggested eps with min_samples=5.")
        best_model, best_labels = perform_dbscan(X, np.percentile(distances, 25), 5, feature_names, original_df)
        best_params = {'eps': np.percentile(distances, 25), 'min_samples': 5}
    else:
        print(f"\nBest DBSCAN parameters: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}")
        # Run the best combination with visualization and analysis
        best_model, best_labels = perform_dbscan(X, best_params['eps'], best_params['min_samples'], 
                                                feature_names, original_df)
    
    return best_model, best_labels, best_params

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Load and preprocess data
    print("Running DBSCAN Clustering Analysis...")
    data = load_and_preprocess_data()
    X_scaled = data['X_scaled']
    X_scaled_df = data['X_scaled_df']
    feature_names = data['feature_names']
    original_df = data['original_df']
    
    # Check data shape and feature types
    print(f"\nData shape: {X_scaled.shape}")
    print(f"Feature names: {feature_names}")
    
    # Try multiple DBSCAN parameter combinations
    dbscan, dbscan_labels, best_params = try_multiple_dbscan_parameters(
        X_scaled, feature_names, original_df
    )
    
    # Print cluster sizes
    print("\nDBSCAN cluster sizes:")
    unique_dbscan, counts_dbscan = np.unique(dbscan_labels, return_counts=True)
    for cluster, count in zip(unique_dbscan, counts_dbscan):
        cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        print(f"  {cluster_name}: {count} samples ({count/len(dbscan_labels)*100:.2f}%)")
    
    print("\nDBSCAN Clustering Analysis Completed!") 