import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import time
import seaborn as sns

from decision_tree import get_model as get_decision_tree_model
from knn import get_model as get_knn_model
from naive_bayes import get_model as get_naive_bayes_model
from random_forest import get_model as get_random_forest_model
from utils import load_and_preprocess_data

def compare_models(show_plots=True):
    """
    Compare all classification models and show results with visualizations.
    """
    plt.ioff()
    
    print("Loading and preprocessing data...")
    # Load the data
    data = load_and_preprocess_data()
    X_train = data['X_train_balanced']
    X_test = data['X_test_scaled']
    y_train = data['y_train_balanced']
    y_test = data['y_test']
    
    # Initialize models
    models = {
        'Decision Tree': get_decision_tree_model(),
        'KNN': get_knn_model(),
        'Naive Bayes': get_naive_bayes_model(),
        'Random Forest': get_random_forest_model()
    }
    
    print("Running all classification algorithms. Please wait...")
    
    # Dictionary to store results
    results = []
    roc_data = []
    confusion_matrices = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        # Train model and measure time
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Get best model if GridSearchCV
        if hasattr(model, 'best_estimator_'):
            best_model = model.best_estimator_
            if hasattr(model, 'best_params_'):
                print(f"Best parameters for {model_name}: {model.best_params_}")
        else:
            best_model = model
        
        # Predict and measure time
        start_pred = time.time()
        y_pred = best_model.predict(X_test)
        pred_time = time.time() - start_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='spam')
        recall = recall_score(y_test, y_pred, pos_label='spam')
        f1 = f1_score(y_test, y_pred, pos_label='spam')
        
        # Store confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm
        
        # ROC curve if possible
        roc_auc = None
        try:
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label='spam')
            roc_auc = auc(fpr, tpr)
            roc_data.append((model_name, fpr, tpr, roc_auc))
        except:
            pass
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall, 
            'F1 Score': f1,
            'AUC': roc_auc if roc_auc is not None else np.nan,
            'Training Time (s)': train_time,
            'Prediction Time (s)': pred_time
        })
    
    # Create DataFrame for results
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION COMPARISON RESULTS")
    print("="*60)
    print("\nPerformance Metrics:")
    print(results_df.to_string(index=False))
    
    # Find best model for each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    recommendation = {}
    
    for metric in metrics:
        if metric == 'AUC' and results_df['AUC'].isnull().any():
            continue
            
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        recommendation[metric] = (best_model, best_score)
    
    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    for metric, (model, score) in recommendation.items():
        print(f"Best model for {metric}: {model} ({score:.4f})")
    
    # Overall recommendation (based on F1 score)
    if 'F1 Score' in recommendation:
        best_model, _ = recommendation['F1 Score']
        print("\nOVERALL RECOMMENDATION:")
        print(f"Based on F1 Score, {best_model} is recommended for this spam classification task.")
    
    # Show plots if requested
    if show_plots:
        # Clear any existing figures before starting
        plt.close('all')
        
        # Set the style for plots
        sns.set(style="whitegrid")
        
        # 1. Bar chart for performance metrics
        fig1 = plt.figure(figsize=(12, 8))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Reshape data for plotting
        plot_data = results_df.melt(id_vars=['Model'], 
                                    value_vars=metrics_to_plot,
                                    var_name='Metric', 
                                    value_name='Score')
        
        # Create grouped bar chart
        sns.barplot(x='Model', y='Score', hue='Metric', data=plot_data)
        plt.title('Classification Performance Metrics Comparison', fontsize=16)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close(fig1)
        
        # 2. Plot ROC curves
        if roc_data:
            fig2 = plt.figure(figsize=(10, 8))
            for model_name, fpr, tpr, roc_auc in roc_data:
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves', fontsize=16)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.show()
            plt.close(fig2)
        
        # 3. Confusion Matrices
        n_models = len(confusion_matrices)
        fig3, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
            
        for i, (model_name, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
        plt.tight_layout()
        plt.show()
        plt.close(fig3)
        
        # 4. Training and prediction time comparison
        fig4 = plt.figure(figsize=(10, 6))
        time_data = results_df.melt(id_vars=['Model'], 
                                   value_vars=['Training Time (s)', 'Prediction Time (s)'],
                                   var_name='Time Type', 
                                   value_name='Seconds')
        
        sns.barplot(x='Model', y='Seconds', hue='Time Type', data=time_data)
        plt.title('Time Performance Comparison', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close(fig4)
        
        # Final cleanup
        plt.close('all')
    
    print("\nClassification comparison completed!")
    return results_df

if __name__ == "__main__":
    # Run comparison with plots
    results = compare_models(show_plots=True) 