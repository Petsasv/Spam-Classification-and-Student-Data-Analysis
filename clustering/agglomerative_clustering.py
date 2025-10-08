import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from utils import load_and_preprocess_data, visualize_clusters, analyze_clusters

def plot_dendrogram(X):
    """Plot the dendrogram to help determine the number of clusters."""
    linkage_matrix = linkage(X, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    return linkage_matrix

def determine_clusters_from_dendrogram(linked, threshold=None):
    if threshold is None:
        heights = linked[:, 2]
        height_diffs = np.diff(heights)
        largest_gap_idx = np.argmax(height_diffs)
        n_clusters = len(height_diffs) - largest_gap_idx
        n_clusters = max(2, min(n_clusters, 10))
    else:
        clusters = []
        n = len(linked) + 1
        for i in range(n - 1):
            if linked[i, 2] > threshold:
                clusters.append(linked[i, :2].astype(int))
        n_clusters = len(set(np.concatenate(clusters)))
    
    print(f"Suggested number of clusters from dendrogram analysis: {n_clusters}")
    return n_clusters

def perform_agglomerative(X, n_clusters, feature_names=None, original_df=None, linkage_method='ward'):
    print(f"\nPerforming Agglomerative clustering with {n_clusters} clusters using {linkage_method} linkage...")
    
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage_method,
        compute_distances=True
    )
    labels = agg_clustering.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    visualize_clusters(X, labels, feature_names, title='Agglomerative Clustering', 
                       method_name=f'k={n_clusters}, linkage={linkage_method}')
    
    if original_df is not None:
        analyze_clusters(original_df, labels, method_name=f'Agglomerative (k={n_clusters}, linkage={linkage_method})')
    
    return agg_clustering, labels

def get_agglomerative_model(n_clusters=3):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    return model

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Running Agglomerative Clustering Analysis...")
    data = load_and_preprocess_data()
    X_scaled = data['X_scaled']
    X_scaled_df = data['X_scaled_df']
    feature_names = data['feature_names']
    original_df = data['original_df']
    
    print(f"\nData shape: {X_scaled.shape}")
    print(f"Feature names: {feature_names}")
    
    linkage_matrix = plot_dendrogram(X_scaled)
    n_clusters = determine_clusters_from_dendrogram(linkage_matrix)
    
    agg_clustering, agg_labels = perform_agglomerative(
        X_scaled, n_clusters, feature_names, original_df
    )
    
    print("\nAgglomerative clustering cluster sizes:")
    unique_agg, counts_agg = np.unique(agg_labels, return_counts=True)
    for cluster, count in zip(unique_agg, counts_agg):
        print(f"  Cluster {cluster}: {count} samples ({count/len(agg_labels)*100:.2f}%)")
    
    print("\nAgglomerative Clustering Analysis Completed!")

    model = get_agglomerative_model()
    model.fit(X_scaled)
    print(f"\nCluster sizes: {np.bincount(model.labels_)}")
    print(f"Silhouette score: {silhouette_score(X_scaled, model.labels_):.3f}") 