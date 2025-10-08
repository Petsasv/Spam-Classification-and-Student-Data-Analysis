import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def clean_dataset(df):
    """Clean the dataset by fixing column names and data types."""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Fix column names
    df.rename(columns={
        'cap_ave numeric': 'cap_ave',
        'ooo': '000'  # Fix the column name for three zeros
    }, inplace=True)
    
    # Convert class to category type and fix the 'emai' category
    df['class'] = df['class'].astype('category')
    if 'emai' in df['class'].cat.categories:
        df['class'] = df['class'].cat.rename_categories({'emai': 'email'})
    
    # Print information about categorical columns
    categorical_cols = ['remove', '000', 'money', 'free', 'our', 'char_$', 'char_!']
    print("\nCategorical columns unique values:")
    for col in categorical_cols:
        print(f"{col} unique values: {df[col].unique()}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    if duplicates > 0:
        print("\nExample of duplicate rows:")
        print(df[df.duplicated()].head())
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    # Save the cleaned dataset to a permanent location
    cleaned_file = 'data/spam_cleaned_permanent.csv'
    df.to_csv(cleaned_file, index=False)
    print(f"\nSaved cleaned dataset to {cleaned_file}")
    
    return df

def load_and_preprocess_data():
    """Load and preprocess the spam dataset."""
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('data/spam.csv')
    
    # Clean the dataset
    print("\nCleaning dataset...")
    df = clean_dataset(df)
    
    # Display basic information
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("\nClass distribution:")
    print(df['class'].value_counts(normalize=True))
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Keep original test data
    X_test_original = X_test.copy()
    y_test_original = y_test.copy()
    
    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to training set
    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print("Training set shape after SMOTE:", X_train_balanced.shape)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts(normalize=True))
    
    return {
        'X_train_balanced': X_train_balanced,
        'X_test_scaled': X_test_scaled,
        'y_train_balanced': y_train_balanced,
        'y_test': y_test,
        'X_test_original': X_test_original,
        'y_test_original': y_test_original
    }

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, X_test_original=None, y_test_original=None, show_plots=True):
    if show_plots:
        plt.close('all')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    if X_test_original is not None and y_test_original is not None:
        print("\nExample predictions from test set:")
        print("First 5 test cases:")
        indices = np.random.choice(len(X_test), 5, replace=False)
        for i, idx in enumerate(indices):
            print(f"\nTest case {i+1}:")
            print(f"Features: {X_test_original.iloc[idx].to_dict()}")
            print(f"Actual class: {y_test_original.iloc[idx]}")
            print(f"Predicted class: {y_pred[idx]}")
            print(f"Prediction confidence: {y_pred_proba[idx]:.2%}")
            print(f"Correct prediction: {'Yes' if y_pred[idx] == y_test_original.iloc[idx] else 'No'}")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if show_plots:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label='spam')
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()
    if show_plots:
        plt.close('all')
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 