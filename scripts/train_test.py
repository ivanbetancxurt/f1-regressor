import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.create_pipeline import create_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib

def train_test(model_id): # find best parameters, fit pipeline, and test
    df = pd.read_csv('data/dataframes/raw_numerical.csv') # get dataset
    X = df.drop(columns='finishing_pos') # extract features
    y = df['finishing_pos'] # extract labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25) # split features and labels with 80-20 split

    models = { # collection of models and their parameter grids (as tuples) to chosen from wtih model_id
        'randomforest': (RandomForestRegressor(random_state=25), {
            'model__n_estimators': [700],
            'model__max_depth': [23],
        }),
        'ridge': (Ridge(random_state=25), {
            'model__alpha': [2, 3, 5, 7, 9]
        }),
        'svr': (SVR(), {
            'model__C': [0.1, 1.0, 10.0, 100.0],
            'model__gamma':   [0.1, 0.5, 1],
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

    # print message if any of the best parameters happen to be at an edge of the parameter grid
    for param, vals in param_grid.items():
        if grid_search.best_params_[param] in (vals[0], vals[-1]):
            print(f'⚠️ {param} = {grid_search.best_params_[param]} at grid edge {vals}!')

    best_model = grid_search.best_estimator_ # get the model with best performing parameters
    best_params_str = '_'.join(f"{param.split('__')[-1]}-{value}" for param, value in grid_search.best_params_.items()) # build string showing best model's parameters
    joblib.dump(best_model, f'models/{model_id}_{best_params_str}.joblib') # serialize and save the model for future use

    y_pred = best_model.predict(X_test) # predict on testing set 
    results = pd.DataFrame({ # make dataframe showing results against actual labels
        'actual': y_test,
        'predicted': y_pred
    })  

    results.to_csv(f'data/results/{model_id}_{best_params_str}.csv', index=False) # save resutls

    return { # return model and MAE on testing set
        'model_id': model_id,
        'best_params': grid_search.best_params_,
        'best_params_str': best_params_str,
        'cv_MAE': -grid_search.best_score_,
        'test_MAE': mean_absolute_error(y_test, y_pred),
        'X_test': X_test,
        'y_test': y_test
    }