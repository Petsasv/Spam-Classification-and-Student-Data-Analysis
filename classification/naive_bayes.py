import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from utils import load_and_preprocess_data, evaluate_model

def get_model():
    """Get a Naive Bayes model."""
    
    model = GaussianNB()
    f1_scorer = make_scorer(f1_score, pos_label='spam')
    grid_search = GridSearchCV(
        model,
        param_grid={},
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1
    )
    
    return grid_search

if __name__ == "__main__":
    # Load and preprocess data
    print("Running Naive Bayes Model...")
    data = load_and_preprocess_data()
    
    # Get the model
    model = get_model()
    
    try:
        # Fit the model
        model.fit(data['X_train_balanced'], data['y_train_balanced'])
        best_model = model.best_estimator_
        
        # Evaluate the model
        evaluate_model(
            model=best_model,
            X_train=data['X_train_balanced'],
            X_test=data['X_test_scaled'],
            y_train=data['y_train_balanced'],
            y_test=data['y_test'],
            model_name='Naive Bayes',
            X_test_original=data['X_test_original'],
            y_test_original=data['y_test_original']
        )
        
    except Exception as e:
        print(f"Error with Naive Bayes: {str(e)}") 