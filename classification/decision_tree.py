from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from utils import load_and_preprocess_data, evaluate_model

def get_model():
    """Get a Decision Tree model with optimized hyperparameters."""
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    f1_scorer = make_scorer(f1_score, pos_label='spam')
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1,
        error_score='raise'
    )
    return grid_search

if __name__ == "__main__":
    print("Running Decision Tree Model...")
    data = load_and_preprocess_data()
    model = get_model()
    
    try:
        model.fit(data['X_train_balanced'], data['y_train_balanced'])
        print(f"Best parameters: {model.best_params_}")
        best_model = model.best_estimator_
        evaluate_model(
            model=best_model,
            X_train=data['X_train_balanced'],
            X_test=data['X_test_scaled'],
            y_train=data['y_train_balanced'],
            y_test=data['y_test'],
            model_name='Decision Tree',
            X_test_original=data['X_test_original'],
            y_test_original=data['y_test_original']
        )
    except Exception as e:
        print(f"Error with Decision Tree: {str(e)}")