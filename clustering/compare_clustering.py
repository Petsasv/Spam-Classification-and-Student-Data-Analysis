# Set environment variables to prevent KMeans memory leak on Windows
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Set to 1 instead of 4
os.environ['MKL_NUM_THREADS'] = '1'  # Add MKL threads setting
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Add OpenBLAS threads setting

# Import warning filter to suppress KMeans warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage
from utils import load_and_preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Run all clustering algorithms and show the comparison results with visualizations"""
    # Disable matplotlib interactive plotting
    plt.ioff()
    
    # Set the style for plots
    sns.set(style="whitegrid")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    X = data['X_scaled']
    original_df = data['original_df']
    
    print("\nRunning all clustering algorithms. Please wait...")
    
    # Dictionary to store clustering results
    labels = {}
    silhouette_scores = {}
    num_clusters = {}
    
    # 1. K-means - find best k
    print("Running K-means clustering...")
    distortions = []
    sil_scores = []
    best_sil = -1
    best_k = 2
    
    # Modify KMeans initialization to use fewer threads
    for k in range(2, 11):
        kmeans = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init=10
        )
        kmeans_labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        distortions.append(kmeans.inertia_)
        sil = silhouette_score(X, kmeans_labels)
        sil_scores.append(sil)
        
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    # Run with best k
    kmeans = KMeans(
        n_clusters=best_k, 
        random_state=42, 
        n_init=10
    )
    kmeans_labels = kmeans.fit_predict(X)
    
    # Store results
    labels['K-means'] = kmeans_labels
    silhouette_scores['K-means'] = silhouette_score(X, kmeans_labels)
    num_clusters['K-means'] = best_k
    
    # 2. Agglomerative Clustering
    print("Running Agglomerative clustering...")
    # Calculate linkage matrix
    linked = linkage(X, method='ward')
    
    # Determine number of clusters based on largest gap in linkage distances
    heights = linked[:, 2]
    height_diffs = np.diff(heights)
    largest_gap_idx = np.argmax(height_diffs)
    n_clusters = max(2, min(len(height_diffs) - largest_gap_idx, 10))
    
    # Run with best number of clusters
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X)
    
    # Store results
    labels['Agglomerative'] = agg_labels
    silhouette_scores['Agglomerative'] = silhouette_score(X, agg_labels)
    num_clusters['Agglomerative'] = n_clusters
    
    # 3. DBSCAN
    print("Running DBSCAN clustering...")
    # Determine eps using k-distance graph
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, 4])  # Get distances to 5th neighbor
    
    # Find elbow point for eps
    diff1 = np.diff(distances)
    diff2 = np.diff(diff1)
    elbow_idx = np.argmax(diff2) + 2
    eps = distances[elbow_idx]
    
    print(f"DBSCAN parameters: eps={eps:.4f}")
    
    # Try different min_samples values
    best_dbscan_sil = -1
    best_dbscan_labels = None
    best_min_samples = 5
    dbscan_results = []
    
    for min_samples in [5, 10, 15]:
        print(f"  Trying min_samples={min_samples}...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = dbscan.fit_predict(X)
        
        # Count clusters and noise points
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = list(db_labels).count(-1)
        
        # Store results for this attempt
        dbscan_results.append({
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'labels': db_labels
        })
        
        print(f"    Found {n_clusters} clusters and {n_noise} noise points")
        
        # Only calculate silhouette if we have at least 2 clusters and not all noise
        if n_clusters >= 2 and n_noise < len(db_labels):
            mask = db_labels != -1
            if np.sum(mask) > 1:
                try:
                    sil = silhouette_score(X[mask], db_labels[mask])
                    print(f"    Silhouette score: {sil:.4f}")
                    if sil > best_dbscan_sil:
                        best_dbscan_sil = sil
                        best_dbscan_labels = db_labels
                        best_min_samples = min_samples
                except Exception as e:
                    print(f"    Could not calculate silhouette score: {str(e)}")
    
    # If we found a good DBSCAN clustering
    if best_dbscan_labels is not None:
        labels['DBSCAN'] = best_dbscan_labels
        mask = best_dbscan_labels != -1
        silhouette_scores['DBSCAN'] = best_dbscan_sil
        num_clusters['DBSCAN'] = len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)
        print(f"\nDBSCAN best result:")
        print(f"  min_samples={best_min_samples}")
        print(f"  clusters={num_clusters['DBSCAN']}")
        print(f"  silhouette={best_dbscan_sil:.4f}")
    else:
        print("\nDBSCAN could not find a valid clustering solution.")
        print("Summary of DBSCAN attempts:")
        for result in dbscan_results:
            print(f"  min_samples={result['min_samples']}: {result['n_clusters']} clusters, {result['n_noise']} noise points")
        
        # Try one more time with more lenient parameters
        print("\nTrying DBSCAN with more lenient parameters...")
        eps = np.percentile(distances, 75)  # Use 75th percentile instead of elbow
        min_samples = 3  # Try with smaller min_samples
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = list(db_labels).count(-1)
        
        print(f"  eps={eps:.4f}, min_samples={min_samples}")
        print(f"  Found {n_clusters} clusters and {n_noise} noise points")
        
        if n_clusters >= 2 and n_noise < len(db_labels):
            mask = db_labels != -1
            if np.sum(mask) > 1:
                try:
                    sil = silhouette_score(X[mask], db_labels[mask])
                    print(f"  Silhouette score: {sil:.4f}")
                    labels['DBSCAN'] = db_labels
                    silhouette_scores['DBSCAN'] = sil
                    num_clusters['DBSCAN'] = n_clusters
                except Exception as e:
                    print(f"  Could not calculate silhouette score: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING COMPARISON RESULTS")
    print("="*60)
    
    # Create results table
    results = []
    for algo, algo_labels in labels.items():
        # Calculate basic stats
        n_cls = num_clusters[algo]
        sil = silhouette_scores[algo]
        noise = algo_labels.tolist().count(-1) if -1 in algo_labels else 0
        
        # Calculate cluster sizes
        sizes = pd.Series(algo_labels).value_counts().sort_index()
        if -1 in sizes:
            sizes = sizes.drop(-1)  # Remove noise points from sizes
        
        # Calculate size balance (coefficient of variation - lower is better)
        size_std = sizes.std() if len(sizes) > 1 else 0
        size_mean = sizes.mean()
        balance = size_std / size_mean if size_mean > 0 else float('inf')
        
        results.append({
            'Algorithm': algo,
            'Clusters': n_cls,
            'Noise Points': noise,
            'Silhouette Score': sil,
            'Size Balance': balance  # Lower is better (more balanced)
        })
    
    # Convert to DataFrame for display
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Algorithm')
    
    # Print the results
    print("\nClustering Performance Metrics:")
    print(results_df)
    
    # Find the best algorithm
    best_algo = results_df['Silhouette Score'].idxmax()
    
    print("\n" + "="*60)
    print(f"RECOMMENDATION: {best_algo}")
    print("="*60)
    
    # Print key insights about each algorithm
    print("\nAlgorithm Characteristics:")
    for algo in labels.keys():
        if algo == 'K-means':
            print(f"- K-means (k={num_clusters[algo]}): Tends to create clusters of similar size and shape.")
        elif algo == 'Agglomerative':
            print(f"- Agglomerative (clusters={num_clusters[algo]}): Creates a hierarchical structure of clusters.")
        elif algo == 'DBSCAN':
            print(f"- DBSCAN (clusters={num_clusters[algo]}): Identifies dense regions and marks outliers as noise.")
    
    # Visualize the results
    # Clear any existing figures before starting
    plt.close('all')
    
    # 1. Bar chart for silhouette scores
    fig1 = plt.figure(figsize=(10, 6))
    algo_names = list(silhouette_scores.keys())
    sil_values = list(silhouette_scores.values())
    
    bars = plt.bar(algo_names, sil_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Silhouette Score Comparison', fontsize=16)
    plt.xlabel('Clustering Algorithm', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.ylim(0, max(sil_values) * 1.2)
    
    # Add value labels on top of bars
    for bar, score in zip(bars, sil_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig1)
    
    # 2. Cluster size distribution
    fig2 = plt.figure(figsize=(15, 5))
    
    for i, (algo, algo_labels) in enumerate(labels.items()):
        plt.subplot(1, len(labels), i+1)
        
        # Count cluster sizes (excluding noise points)
        cluster_sizes = pd.Series(algo_labels)
        cluster_sizes = cluster_sizes[cluster_sizes != -1].value_counts().sort_index()
        
        # Plot
        sns.barplot(x=cluster_sizes.index.astype(str), y=cluster_sizes.values)
        plt.title(f'{algo} Cluster Sizes')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Points')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig2)
    
    # 3. Noise points comparison (if applicable)
    if any(-1 in algo_labels for algo_labels in labels.values()):
        fig3 = plt.figure(figsize=(10, 6))
        
        noise_counts = []
        algo_names = []
        
        for algo, algo_labels in labels.items():
            noise = list(algo_labels).count(-1)
            if noise > 0 or algo == 'DBSCAN':  # Always include DBSCAN
                noise_counts.append(noise)
                algo_names.append(algo)
        
        if noise_counts:
            bars = plt.bar(algo_names, noise_counts, color='#e74c3c')
            plt.title('Noise Points Comparison', fontsize=16)
            plt.xlabel('Clustering Algorithm', fontsize=12)
            plt.ylabel('Number of Noise Points', fontsize=12)
            
            # Add value labels
            for bar, count in zip(bars, noise_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.show()
            plt.close(fig3)
    
    # 4. K-means elbow method visualization
    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), distortions, 'bo-')
    plt.title('K-means Elbow Method', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Distortion (Inertia)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.annotate(f'Selected k={best_k}', 
                xy=(best_k, distortions[best_k-2]), 
                xytext=(best_k+1, distortions[best_k-2]*1.1),
                arrowprops=dict(arrowstyle='->'))
    plt.tight_layout()
    plt.show()
    plt.close(fig4)
    
    # 5. K-means silhouette score visualization
    fig5 = plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), sil_scores, 'ro-')
    plt.title('K-means Silhouette Scores', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.annotate(f'Selected k={best_k}', 
                xy=(best_k, sil_scores[best_k-2]), 
                xytext=(best_k+1, sil_scores[best_k-2]*0.95),
                arrowprops=dict(arrowstyle='->'))
    plt.tight_layout()
    plt.show()
    plt.close(fig5)
    
    # Final cleanup
    plt.close('all')
    
    print("\nClustering comparison visualization complete!")
    
if __name__ == "__main__":
    main() 