HOW TO RUN:

There is no need to run the generate_data.py and clean_data.py files; The JSONs and generate and manipulate are ready to go in data/jsons.

The dataset with one hot encoded values (raw_numerical.csv) is also ready to go.

train_test.py is the meat of this project. At the time of this submission, we only consider three models, Random Forect Regressor, Ridge, and SVR. In order to
train and tune one of these models, feel free to edit notebook.ipynb or create your own notebook with the following import statement:

  from train_test import train_test

After that, you can call train_test like a regular method. It takes one string argument which is the "ID" of the model you want to train. The IDs are the following:

  'randomforest'
  'ridge'
  'svr'

Choose whichever one you'd like to train and print the returned value to see results and information of training. You can play around with the parameter grid for your model within
train_test.py. Each call to train_test will generate a csv file in data/results showing the predicted values against the ground truth. 

If you'd like to plot your results, follow the patters shown in plots.ipynb.
