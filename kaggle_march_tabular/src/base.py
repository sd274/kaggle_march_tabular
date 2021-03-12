import os
from datetime import datetime
from joblib import dump, load
from sklearn import metrics
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from sklearn.exceptions import NotFittedError


class TabularModelConfigBase:

    def __init__(self, load_model=False):
        current_file_path = os.path.abspath(__file__)

        try:
            self.model_save_location = os.path.join(
                os.path.dirname(os.path.dirname(current_file_path)),
                'models',
                self.name
            )
            self.model_logs_location = os.path.join(
                os.path.dirname(os.path.dirname(current_file_path)),
                'logs',
                self.name
            )
            for directory in [self.model_logs_location, self.model_save_location]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
        except:
            raise Exception('Model name needs to be defined')

        if load_model:
            saved_models = os.path.join(self.model_logs_location, 'saved_models.csv')
            df = pd.read_csv(saved_models)
            latest_model_name = df.sort_values('timestamp', ascending=False).iloc[0]['model_name']
            model_location = os.path.join(self.model_save_location, latest_model_name)
            print(f'Loading model from {model_location}...')
            try:
                self.model = load(os.path.join(model_location))
            except:
                raise Exception(
                    'Model has not yet been saved. You need to fit a model with the save flag set to true.')
        else:
            self.model = self.set_model()

    def set_model(self):
        raise Exception('No model has been implimented')

    def get_model(self):
        return self.model

    def save_model(self):
        timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
        model_savename = os.path.join(
                self.model_save_location, f'model_{timestamp}.joblib')
        dump(self.model, model_savename)
        log_data = pd.DataFrame([{
            'timestamp': timestamp,
            'model_name': f'model_{timestamp}.joblib'
        }])
        logfile = os.path.join(self.model_logs_location, 'saved_models.csv')
        with open(logfile, 'a') as f:
            log_data.to_csv(f, mode='a', header=not f.tell(), index=False)
        return f'model_{timestamp}.joblib'

    def log_validation_data(self,X, y, notes='', save_name='fit.csv', meta_data={}):
        prediction = self.model.predict_proba(X)[:, 1]
        auc_score = metrics.roc_auc_score(y, prediction)

        log_data = pd.DataFrame([{
            'timestamp': datetime.today(),
            'auc_score': auc_score,
            'fit_notes': notes,
            **meta_data
        }])
        logfile = os.path.join(self.model_logs_location, save_name)
        with open(logfile, 'a') as f:
            log_data.to_csv(f, mode='a', header=not f.tell(), index=False)
        return log_data

    def fit_model(self, X, y, fit_notes='', save=False, log_data=True):
        self.model.fit(X, y)
        save_name = ''
        if save:
            save_name = self.save_model()
        if log_data:
            self.log_validation_data(X, y, notes=fit_notes, save_name='fit.csv', meta_data={
                'save': save,
                'save_name': save_name
            })
            

        

    def validate_model(self, X, y, validation_name, log_validation=True):
        try: 
        
            test_prediction = self.model.predict_proba(X)[:, 1]

            predictions = self.model.predict(X)
        except NotFittedError:
            raise Exception('Model has not been fitted yet please run fit_model.')

        score = metrics.roc_auc_score(y, test_prediction)
        accuracy = metrics.accuracy_score(y, predictions)

        results = pd.DataFrame([
            {
                'validation_name': validation_name,
                'timestamp': datetime.today(),
                'accuracy': accuracy,
                'score': score
            }
        ])
        validation_logfile = os.path.join(
            self.model_logs_location, 'validation.csv')
        with open(validation_logfile, 'a') as f:
            results.to_csv(f, mode='a', header=not f.tell(), index=False)
        return results