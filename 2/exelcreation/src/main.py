from . import load_data, split_and_scale, train_and_evaluate_model

def run_model_pipeline(data_path: str, target_column: str, model_library: str, 
                      model_function: str, model_params: dict = None):
    """
    Complete ML pipeline from data loading to model evaluation
    """
    # Load and preprocess data
    X, y = load_data(data_path, target_column)
    x_train, x_test, y_train, y_test = split_and_scale(X, y)
    
    # Train and evaluate model
    results = train_and_evaluate_model(
        library=model_library,
        function=model_function,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        attributes=model_params
    )
    
    
    return results
