import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from create_pipeline import create_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

def train_test(pipeline, model_id): # find best parameters, fit pipeline, and test
    df = pd.read_csv('data/dataframes/raw_numerical.csv') # get dataset
    X = df.drop(columns='finishing_pos') # get unlabeled version
    y = df['finishing_pos'] # extract labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25) # split features and labels with 80-20 split

    models = { # collection of models and their parameter grids (as tuples) to chosen from wtih model_id
        'randomforest': (RandomForestRegressor(random_state=25), {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth':    [None, 10, 20],
            'model__min_samples_leaf': [1, 3, 5]
        }),
        'ridge': (Ridge(random_state=25), {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth':    [None, 10, 20],
            'model__min_samples_leaf': [1, 3, 5]
        }),
        'svr': (SVR(), {
            'model__C': [0.1, 1.0, 10.0, 100.0],
            'model__gamma':   ['scale', 'auto', 0.001, 0.01, 0.1],
            'model__epsilon': [0.01, 0.1, 0.2, 0.5]
        })
    }

    estimator, param_grid = models[model_id] # get chosen model and its parameter grid

    pipeline = create_pipeline(estimator) # create pipeline

    grid_search = GridSearchCV( # initialize grid search with 10 folds and MAE scoring
        pipeline,
        param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        refit=True,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train) # run the grid search on the training set

    best_model = grid_search.best_estimator_ # get the model with best performing parameters

    y_pred = best_model.predict(X_test) # predict on testing set 
    #todo: save predictions as CSV
    print("MAE:", mean_absolute_error(y_test, y_pred)) 