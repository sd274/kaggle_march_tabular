from . import TabularModelConfigBase
import pipeline_tools as pt
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np

class TunedLightGbmConfig(TabularModelConfigBase):
    name = 'tuned_lightgbm_model'

    def set_model(self):
        cat_features = [f'cat{i}' for i in range(19)]
        num_features = [f'cont{i}' for i in range(11)]
        target = 'target'
        pre_pipe = pt.standard_preprocessing_pipe(
            num_features=num_features, cat_features=cat_features)
        model = Pipeline([
            ('pre_pipe', pre_pipe),
            ('learn', LGBMClassifier())
        ])
        return model

    def fit_model(self, X, y, fit_notes='', save=False, log_data=True, N_SPLITS=5):
        model = self.model

        def objective(trial, data = X, target=y, cv=StratifiedKFold(N_SPLITS, shuffle = True, random_state = 29)):
            model = self.model

            param = {
                "learn__random_state": trial.suggest_int("random_state", 1, 100),
                "learn__objective": "binary",
                "learn__metric": "binary_logloss",
                "learn__verbosity": -1,
                "learn__boosting_type": "gbdt",
                "learn__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "learn__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "learn__max_depth": trial.suggest_int("max_depth", -1, 10),
                "learn__n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learn__num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "learn__colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "learn__subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "learn__subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                "learn__min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
            
            model.set_params(**param)
            
            val_aucs = []
            aucs = []
            
            for train_idx, val_idx in cv.split(data, target):
                
                model.fit(data.iloc[train_idx], target.iloc[train_idx])
                val_true = target.iloc[val_idx]
                
                preds = model.predict_proba(data.iloc[val_idx])[:, 1]
                
                auc = metrics.roc_auc_score(val_true, preds)
                
                aucs.append(auc)
            
            return np.average(aucs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_trial.params
        self.model.set_params(
            **{f'learn__{key}': value for key, value in best_params.items()}
        )

        self.model.fit(X, y)
        save_name = '' 
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
