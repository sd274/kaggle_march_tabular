from . import TabularModelConfigBase
import pipeline_tools as pt
from sklearn.pipeline import Pipeline
import xgboost as xgb

class BaseLineModelConfig(TabularModelConfigBase):
    name = 'baseline_model'
    
    def set_model(self):
        cat_features = [f'cat{i}' for i in range(19)]
        num_features = [f'cont{i}' for i in range(11)]
        target = 'target'
        pre_pipe = pt.standard_preprocessing_pipe(num_features=num_features, cat_features=cat_features)
        model = Pipeline([
            ('pre_pipe', pre_pipe),
            ('learn', xgb.XGBClassifier())
        ])
        return model