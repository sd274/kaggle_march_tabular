import pandas as pd
import pipeline_tools as pt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import src
import os


def run_experiment(model_config, save=False, testing=True):
    print('Loading config...')
    config = getattr(src, model_config)()

    print('Loading training data...')
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(current_file_dir, 'tabular-playground-series-mar-2021' , 'train.csv')
    if testing:
        df = pd.read_csv(train_dir).head(1000)
    else:
        df = pd.read_csv(train_dir)
    cat_features = [f'cat{i}' for i in range(19)]
    num_features = [f'cont{i}' for i in range(11)]
    target = 'target'
    X = df[cat_features + num_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('Fitting the model...')
    config.fit_model(X_train, y_train, save=save, fit_notes='train test split state of 42')

    print('Validating the model...')
    config.validate_model(X_test, y_test, validation_name='train test split state of 42')
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run experiments from the command line.')
    parser.add_argument("--config", required=True, type=str, help="The name of the class that contains the config for your experiment.")
    parser.add_argument("-save", help="Save the model",
        action='store_true'
    )
    parser.add_argument("-testing", help="Run with small amount of data to test scripts.",
        action='store_true'
    )
    
    args = parser.parse_args()
    cofing_name = args.config
    save = args.save
    testing = args.testing
    
    run_experiment(cofing_name, save=save, testing=testing)