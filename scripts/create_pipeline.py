from preprocessor import preprocessor
from sklearn.pipeline import Pipeline

def create_pipeline(estimator): # return new pipeline with scalar preprocessor and specified model
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])