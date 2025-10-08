import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import os

def clean_dataset(df):
    """
    Clean the StudentsPerformance dataset by checking for missing values,
    fixing data types, and removing duplicates if any.
    
    Args:
        df: Original DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print("\nNo missing values found.")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\nFound {duplicates} duplicate rows")
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    else:
        print("\nNo duplicate rows found.")
    
    print("\nData types:")
    print(df.dtypes)
    
    score_cols = ['math score', 'reading score', 'writing score']
    for col in score_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    cleaned_file = 'data/students_performance_cleaned.csv'
    try:
        df.to_csv(cleaned_file, index=False)
        print(f"\nSaved cleaned dataset to {cleaned_file}")
    except Exception as e:
        print(f"\nCould not save cleaned dataset: {str(e)}")
        
        try:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_cleaned_file = os.path.join(parent_dir, 'data', 'students_performance_cleaned.csv')
            df.to_csv(alt_cleaned_file, index=False)
            print(f"Saved cleaned dataset to {alt_cleaned_file}")
        except Exception as e2:
            print(f"Could not save cleaned dataset to alternative path: {str(e2)}")
    
    return df

def load_and_preprocess_data():
    """
    Load and preprocess the StudentsPerformance dataset.
    Returns preprocessed data suitable for clustering.
    """
    # Load the dataset
    print("Loading student performance dataset...")
    try:
        try:
            df = pd.read_csv('data/students_performance_cleaned.csv')
            print("Loaded cleaned dataset.")
        except FileNotFoundError:
            # Try alternative path for cleaned file
            try:
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                cleaned_file = os.path.join(parent_dir, 'data', 'students_performance_cleaned.csv')
                df = pd.read_csv(cleaned_file)
                print("Loaded cleaned dataset from alternative path.")
            except FileNotFoundError:
                # If cleaned version doesn't exist, load the original
                try:
                    # Try relative path from clustering directory
                    df = pd.read_csv('../data/StudentsPerformance.csv')
                    print("Loaded original dataset.")
                    # Clean the dataset
                    df = clean_dataset(df)
                except FileNotFoundError:
                    # Try alternative paths
                    try:
                        df = pd.read_csv('data/StudentsPerformance.csv')
                        print("Loaded original dataset from current directory.")
                        # Clean the dataset
                        df = clean_dataset(df)
                    except FileNotFoundError:
                        # Try absolute path to project root
                        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                              'data', 'StudentsPerformance.csv')
                        df = pd.read_csv(file_path)
                        print(f"Loaded original dataset from {file_path}")
                        # Clean the dataset
                        df = clean_dataset(df)
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    encoded_df = df.copy()
    
    education_order = {
        'some high school': 1,
        'high school': 2,
        'some college': 3,
        'associate\'s degree': 4,
        'bachelor\'s degree': 5,
        'master\'s degree': 6
    }
    encoded_df['parental level of education'] = encoded_df['parental level of education'].map(education_order)
    
    encoded_df['test preparation course'] = encoded_df['test preparation course'].map({'none': 0, 'completed': 1})
    
    encoded_df['lunch'] = encoded_df['lunch'].map({'standard': 1, 'free/reduced': 0})
    
    encoded_df = pd.get_dummies(encoded_df, columns=['gender', 'race/ethnicity'], drop_first=True)
    
    if encoded_df.isnull().sum().sum() > 0:
        print("\nWarning: Dataset contains missing values!")
        print(encoded_df.isnull().sum())
        encoded_df = encoded_df.dropna()
    
    X = encoded_df[['math score', 'reading score', 'writing score']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("\nPreprocessed data shape:", X_scaled_df.shape)
    
    return {
        'original_df': df,
        'encoded_df': encoded_df,
        'X_scaled': X_scaled,
        'X_scaled_df': X_scaled_df,
        'feature_names': X.columns
    }

def visualize_clusters(data, labels, feature_names=None, title='Cluster Visualization', method_name='', n_components=2):
    """
    Visualize clusters using PCA for dimension reduction if needed.
    """
    plt.close('all')
    
    X = data.copy()
    
    if X.shape[1] > 2:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        fig1 = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'{title} - {method_name}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig1)
        
        if n_components >= 3:
            fig2 = plt.figure(figsize=(10, 8))
            ax = fig2.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Cluster')
            ax.set_title(f'3D {title} - {method_name}')
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_zlabel('PCA Component 3')
            plt.tight_layout()
            plt.show()
            plt.close(fig2)
    else:
        fig = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'{title} - {method_name}')
        
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

def analyze_clusters(df, labels, features=None, method_name=''):
    """
    Analyze clusters to understand their properties.
    """
    plt.close('all')  # Close all existing figures
    
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"\n{'='*50}")
    print(f"Cluster Analysis for {method_name} with {n_clusters} clusters")
    print(f"{'='*50}")
    
    print("\nCluster Distribution:")
    cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
    print(cluster_counts)
    
    # Plot cluster distribution
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title(f'Cluster Distribution - {method_name}')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Students')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print("\nCluster Properties (Mean Values):")

    if features is not None:
        features_to_analyze = features
    else:
        features_to_analyze = ['math score', 'reading score', 'writing score']

    cluster_means = df_with_clusters.groupby('Cluster')[features_to_analyze].mean()
    print(cluster_means)
    
    # Plot average scores by cluster
    plt.figure(figsize=(12, 8))
    cluster_means.plot(kind='bar')
    plt.title(f'Average Scores by Cluster - {method_name}')
    plt.xlabel('Cluster')
    plt.ylabel('Average Score')
    plt.legend(title='Subject')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title(f'Cluster Means Heatmap - {method_name}')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return df_with_clusters, cluster_means 