##################
## Author: Vitor B. C. Silva
## Description: This script executes a grid search over the parameter space defined in the dissertation to find the best Random Forest (RF) model according to the MCC variable. 
# The script outputs the best model and its training results to the trained_models folder.
# The script expects a csv file containing the training feature set as defined in the dissertation.
# Besides all features in the columns, a column named target is expected to be present, holding the expected outcome of an entry.
# Rows in this file are the entries describing user-window pairs. The training feature set is expected to be in the data directory, and named as train_feature_set.csv.
##

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier 
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
from joblib import dump

#Setting the training feature set filename and loading it.
training_feature_set_filename = "data\\train_feature_set.csv"
try:
    train_set = pd.read_csv(training_feature_set_filename, index_col='index')
except FileNotFoundError:
    print("Expected file defined in the training_feature_set_filename variable not found. Please fix it and run again.")
    exit(0)
except pd.errors.EmptyDataError:
    print("Training feature set file is empty. Please fix it and run again.")
    exit(0)

#Separating the target information from the features before training the models.
try:
    y_train = train_set.target
except AttributeError:
    print("Training feature set file does not contain target column. Please fix it and run again.")
    exit(0)

#Removing the target from the pandas data object.
train_set.drop(['target'], axis=1, inplace=True)

#Outlining the numerical and categorical features for different pre-processing procedures.
numerical_cols = []
categorical_cols = []
for column_name in train_set.columns:
    #If a column is an integer or floating point it is numerical. Otherwise, it is categorical.
    if train_set[column_name].dtype in ['int64', 'float64']:
        numerical_cols.append(column_name)
    else:
        categorical_cols.append(column_name)

#Defining the preprocessing procedure for missing numerical data with zeros.
numerical_transformer = SimpleImputer(strategy='constant', fill_value = 0)

#Defining the preprocessing procedures for categorical data.
#Missing data is replaced by the most common element and then the categorical features go through one-hot encoding.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Preprocessing for numerical and categorical features are merged to be added to the pipeline.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#Creating a base random forest model, the parameters are supplied later during the grid search procedure.
model = RandomForestClassifier()

#Given the presence of class imbalance, we apply Random Oversampling of the minority class to mitigate the issue.
#Creating the random over sampler object.
minority_ros = RandomOverSampler(random_state=0)

#Combining the preprocessing transformer with the random oversampling and the base model in a pipeline.
imba_pipeline = make_pipeline(preprocessor, minority_ros, model)

#Defining the parameter space that the grid search will focus on for the RF model.
parameters_dict = {
    'randomforestclassifier__n_estimators': [50,100,150,200,250,300,350,400],
    'randomforestclassifier__max_depth': [None,10,20,30,40,50],
    'randomforestclassifier__min_samples_split': [2,4,8],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4, 8, 12, 16],
    'randomforestclassifier__max_features': [10, 25, 35, 50, 100, 200],
    'randomforestclassifier__class_weight': [{0: 1, 1: 1},{0: 1, 1: 8}, {0: 1, 1: 16}, {0: 1, 1: 24}, {0: 1, 1: 32}],
    'randomforestclassifier__random_state': [0],
    'randomforestclassifier__n_jobs' : [-1]
}

#Defines the metrics that will be calculated during the training of the models. 
scoring = {
    'Accuracy' : 'accuracy',
    'Balanced Accuracy' : 'balanced_accuracy', 
    'Precision' : 'precision', 
    'Recall' : 'recall' , 
    'F1' : 'f1', 
    'ROC AUC' : 'roc_auc', 
    'MCC' : make_scorer(matthews_corrcoef, greater_is_better=True)
}

#Defines the grid search object over the defined parameter space for the pipeline aiming at finding the model with best MCC. 
# This is done with a 10 fold cross-validation for each combination in the parameter space.
# The training results are stored for all metrics defined in the scoring variable.
grid_imba = GridSearchCV(imba_pipeline, param_grid=parameters_dict, cv=10, scoring=scoring, return_train_score=True, refit='MCC')
grid_imba.fit(train_set,y_train)

#Prints the best parameters found for the MCC metric and the best MCC score.
print('Best parameters found for MCC: ', grid_imba.best_params_)
print('Best score for MCC: ', grid_imba.best_score_)

#The best models and the training results are stored for later use and testing. 
dump(grid_imba.best_estimator_, 'trained_models\\rf_model.joblib')
dump(grid_imba.cv_results_, 'trained_models\\rf_training_results.joblib') 

