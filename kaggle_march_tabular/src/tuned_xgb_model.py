from . import TabularModelConfigBase
import pipeline_tools as pt
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn import metrics

class TunedXGBModelConfig(TabularModelConfigBase):
    name = 'tuned_xgb_model'

    def set_model(self):
        cat_features = [f'cat{i}' for i in range(19)]
        num_features = [f'cont{i}' for i in range(11)]
        target = 'target'
        pre_pipe = pt.standard_preprocessing_pipe(
            num_features=num_features, cat_features=cat_features)
        model = Pipeline([
            ('pre_pipe', pre_pipe),
            ('learn', xgb.XGBClassifier())
        ])
        return model

    def fit_model(self, X, y, fit_notes='', save=False, log_data=True):
        model = self.model

        def objective(trial, data=X, target=y):
            train_x, test_x, train_y, test_y = train_test_split(
                data, target, test_size=0.15, random_state=42)
            param = {
                'learn__lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'learn__alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'learn__colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'learn__subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learn__learning_rate': trial.suggest_categorical('learning_rate', [0.003, 0.03, 0.3, 0.6]),
                'learn__n_estimators':trial.suggest_int('n_estimators', 100, 300),
                'learn__max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
                'learn__random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
                'learn__min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            }
            model.set_params(**param)

            model.fit(train_x, train_y)

            preds = self.model.predict_proba(test_x)[:, 1]

            score = metrics.roc_auc_score(test_y, preds)

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_trial.params
        self.model.set_params(
            **{f'learn__{key}': value for key, value in best_params.items()}
        )

        if save:
            save_name = self.save_model()

        if log_data:
            validation_data = self.log_validation_data(X, y, notes=fit_notes, save_name='fit.csv', meta_data={
                'save': save,
                'save_name': save_name,
                'best_params': str(best_params.items())
            })
            return validation_data
        return {}
