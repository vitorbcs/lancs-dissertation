##################
## Author: Vitor B. C. Silva
## Description: This script loads the best Random Forest and XGB models and evaluate them against the testing feature set.
# The script prints the static scores of the models in the testing set for several metrics. Additionally, it outputs csv files with the features importance for each model.
# Moreover, the script generates charts for the ROC and PR curves and confusion matrices. Finally, the script outputs an image of the first decision tree estimator for both models.
# The script expects a csv file containing the test feature set as defined in the dissertation.
# Besides all features in the columns, a column named target is expected to be present, holding the expected outcome of an entry.
# Rows in this file are the entries describing user-window pairs. The test feature set is expected to be in the data directory, and named as test_feature_set.csv.
# This script also expects the presence of the XGB and RF models outputted by the random_forest_training.py and xgboost_training.py scripts in the trained_models folder.
# All the files outputted by this script are saved to the charts folder.
##

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import load
from matplotlib.colors import ListedColormap
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, balanced_accuracy_score, precision_score,roc_auc_score,\
    matthews_corrcoef, RocCurveDisplay, PrecisionRecallDisplay
from xgboost import plot_tree

#This function receives the predictions made by models and the expected outcomes and calculates the static scores for multiple metrics.
#All scores are returned in a dictionary.
def measure_performance(expected_y, predicted_y):
    scores_dict = {}
    scores_dict['confusion_matrix'] = confusion_matrix(y_true=expected_y, y_pred=predicted_y)
    scores_dict['recall_score'] = recall_score(y_true=expected_y, y_pred=predicted_y)
    scores_dict['f1_score'] = f1_score(y_true=expected_y, y_pred=predicted_y)
    scores_dict['accuracy_score'] = accuracy_score(y_true=expected_y, y_pred=predicted_y)
    scores_dict['balanced_accuracy_score'] = balanced_accuracy_score(y_true=expected_y, y_pred=predicted_y)
    scores_dict['precision_score'] = precision_score(y_true=expected_y, y_pred=predicted_y)
    scores_dict['roc_auc_score'] = roc_auc_score(y_true=expected_y, y_score=predicted_y)
    scores_dict['matthews_corrcoef'] = matthews_corrcoef(y_true=expected_y, y_pred=predicted_y)
    return scores_dict

#This function calculates and plots the confusion matrix of a model.
# The function uses the expected targets and the predictions made to compute the confusion matrix.
def plot_confusion_matrix(expected_y, predicted_y, title_prefix):
    #Calculating the confusion matrix.
    cf_matrix = confusion_matrix(expected_y, predicted_y)
    
    #Creating the image object.
    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(2)
    
    #Calculating true and false, positives and negatives percentages.
    true_negative = cf_matrix[0][0]
    false_positive = cf_matrix[0][1]
    false_negative = cf_matrix[1][0]
    true_positive = cf_matrix[1][1]
    percentage_true_negative = cf_matrix[0][0]/np.sum(cf_matrix)
    percentage_false_positive = cf_matrix[0][1]/np.sum(cf_matrix)
    percentage_false_negative = cf_matrix[1][0]/np.sum(cf_matrix)
    percentage_true_positive = cf_matrix[1][1]/np.sum(cf_matrix)

    #Derive true and false positive rates for the confusion matrix.
    false_positive_rate = false_positive / (false_positive + true_negative)
    true_positive_rate = true_positive / (true_positive + false_negative)

    #Reorder the confusion matrix as it appears in the dissertation.
    cf_matrix = [[true_positive, false_positive],[false_negative, true_negative]]
    
    #Setting the labels for the image.
    label_true_negative = 'True Negative' + '\n' + str(true_negative) + '\n' + "{0:.2%}".format(percentage_true_negative)
    label_false_positive = 'False Positive' + '\n' + str(false_positive) + '\n' + "{0:.2%}".format(percentage_false_positive) + '\n' + "FPR: {0:.2%}".format(false_positive_rate)
    label_false_negative = 'False Negative' + '\n' + str(false_negative) + '\n' + "{0:.2%}".format(percentage_false_negative)
    label_true_positive = 'True Positive' + '\n' + str(true_positive) + '\n' + "{0:.2%}".format(percentage_true_positive) + '\n' + " TPR: {0:.2%}".format(true_positive_rate)
    
    matrix_labels = [
        [ label_true_positive, label_false_positive ],
        [ label_false_negative, label_true_negative ]
    ]

    #Plotting the confusion matrix chart.
    ax = sns.heatmap(cf_matrix, annot=matrix_labels, fmt='', cmap=ListedColormap(['lightblue']), cbar=False, linewidths=1, linecolor='black', clip_on=False)
    ax.set_title(title_prefix + ' Confusion Matrix')

    #Adding the axis labels.
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    #Adding the true and false labels.
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])

    #Saves the image and displays it.
    plt.savefig(charts_path + title_prefix + 'confusion_matrix.pdf')
    plt.show(block=False)

#This function plots the ROC curve of both models together to allow comparisons.
# The function uses the expected targets and the predictions made by each model to compute each plot.
def plot_roc(expected_y, xgb_y_pred_proba, rf_y_pred_proba):
    #Creating the image object.
    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    fig.set_figheight(5)

    #Defining the axis limits.
    ax.set_ylim(top=1.02)
    ax.set_ylim(bottom=0)
    
    #Setting the axis labels.
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    #Drawing each model's curve.
    roc_xgb = RocCurveDisplay.from_predictions(y_true=expected_y, y_pred=xgb_y_pred_proba, name='XGBoost', pos_label=1, ax=ax)
    roc_rf = RocCurveDisplay.from_predictions(y_true=expected_y, y_pred=rf_y_pred_proba, name='Random Forest', pos_label=1, ax=ax)

    #Plotting the naive model line.
    ax.plot([0, 1], [0, 1], linewidth=1, linestyle="--", color='C7', label='Naive model')

    #Setting the legend position.
    leg = plt.legend(loc='best')

    #Showing the file and saving it to a file.
    plt.tight_layout()
    plt.savefig(charts_path+'roc_curve.pdf')
    plt.show(block=False)

#This function plots the PR curve of both models together to allow comparisons.
# The function uses the expected targets and the predictions made by each model to compute each plot.
def plot_precision_recall(pipeline_rf, pipeline_xgb, X, y, baseline):
    #Creating the image object.
    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    fig.set_figheight(5)
   
    #Setting the axis labels.
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")

    #Drawing each model's curve.
    pr_xgb = PrecisionRecallDisplay.from_estimator(pipeline_xgb, X, y, name='XGB', pos_label=1, ax=ax)
    pr_rf = PrecisionRecallDisplay.from_estimator(pipeline_rf, X, y, name='RF', pos_label=1, ax=ax)

    #Plotting the naive model line.
    ax.plot([0, 1], [baseline, baseline], linewidth=1, linestyle="--", color='C7', label='Naive model')
    
    #Adding a tick for the baseline Y value
    plt.yticks(list(plt.yticks()[0]) + [baseline])

    #Defining the axis limits.
    ax.set_ylim(top=1.02)
    ax.set_ylim(bottom=0)

    #Setting the legend position.
    leg = plt.legend(loc='best')

    #Showing the file and saving it to a file.
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(charts_path+'precision_recall_curve.pdf')

#Sets the folder where charts will be saved.
charts_path = 'charts\\'

#Setting the test feature set filename and loading it.
test_feature_set_filename = "data\\test_feature_set.csv"
try:
    test_set = pd.read_csv(test_feature_set_filename, index_col='index')
except FileNotFoundError:
    print("Expected file defined in the test_feature_set_filename variable not found. Please fix it and run again.")
    exit(0)
except pd.errors.EmptyDataError:
    print("Test feature set file is empty. Please fix it and run again.")
    exit(0)

#Separating the target information from the features before testing the models.
try:
    y_test = test_set.target
except AttributeError:
    print("Test feature set file does not contain target column. Please fix it and run again.")
    exit(0)

#Removing the target from the pandas data object.
test_set.drop(['target'], axis=1, inplace=True)

#Loading the pipelined RF and XGB models already trained from their files.
try:
    xgb_pipeline = load('trained_models\\xgb_model.joblib')
except FileNotFoundError:
    print("XGB model expected to be in folder trained_models and named xgb_model.joblib not found. Please fix it and run again.")
    exit(0)

try:
    rf_pipeline = load('trained_models\\rf_model.joblib')
except FileNotFoundError:
    print("XGB model expected to be in folder trained_models and named rf_model.joblib not found. Please fix it and run again.")
    exit(0)

#Making the predictions for the testing set with both models.
xgb_y_predicted_model = xgb_pipeline.predict(test_set)
rf_y_predicted_model = rf_pipeline.predict(test_set)

#Calculating the static measurements from the models' results and printing them.
xgb_scores_dict = measure_performance(expected_y=y_test, predicted_y=xgb_y_predicted_model)
rf_scores_dict = measure_performance(expected_y=y_test, predicted_y=rf_y_predicted_model)
print('Best model scores per user window XGB: ',xgb_scores_dict)
print('Best model scores per user window RF: ',rf_scores_dict)

#Outputting the models' feature importance list to CSV files.
xgb_model = xgb_pipeline.named_steps["xgbclassifier"]
xgb_feature_names = xgb_pipeline.named_steps["columntransformer"].get_feature_names_out().tolist()
xgb_feature_scores = pd.Series(xgb_model.feature_importances_, index=xgb_feature_names).sort_values(ascending=False)
xgb_feature_scores.to_csv(charts_path+'xgb_feature_importance_list.csv')

rf_model = rf_pipeline.named_steps["randomforestclassifier"]
rf_feature_names = rf_pipeline.named_steps["columntransformer"].get_feature_names_out()
rf_feature_scores = pd.Series(rf_model.feature_importances_, index=rf_feature_names).sort_values(ascending=False)
rf_feature_scores.to_csv(charts_path+'rf_feature_importance_list.csv')

################# PLOTS ######################

#Plotting the first decision tree for the XGB model.
xgb_model.get_booster().feature_names = xgb_feature_names
plot_tree(xgb_model)
plt.savefig(charts_path + 'xgb_one_tree.pdf')

#Plotting the first decision tree for the RF model.
rf_decision_tree = rf_model.estimators_[0]
export_graphviz(rf_decision_tree, out_file=charts_path+'rf_one_tree.dot', feature_names = rf_feature_names, class_names = ['Clean', 'Infected'],\
     rounded = True, proportion = False, precision = 2, filled = True, max_depth=2)


#Plotting the confusion matrix for both models.
plot_confusion_matrix(expected_y=y_test, predicted_y=xgb_y_predicted_model, title_prefix="XGB User-Window Pairs")
plot_confusion_matrix(expected_y=y_test, predicted_y=rf_y_predicted_model, title_prefix="RF User-Window Pairs")

#Getting the predictions probabilities from the models for the test set.
xgb_y_pred_proba = xgb_pipeline.predict_proba(test_set)[:, 1] 
rf_y_pred_proba = rf_pipeline.predict_proba(test_set)[:, 1] 

#Plotting the ROC for both models.
plot_roc(expected_y=y_test, xgb_y_pred_proba=xgb_y_pred_proba, rf_y_pred_proba=rf_y_pred_proba)

#Calculating the baseline for the PR plot
count_positive_cases = y_test.value_counts()[1]
count_negative_cases = y_test.value_counts()[0]
pr_baseline = count_positive_cases / (count_positive_cases+count_negative_cases)

#Plotting the PR curve for both models.
plot_precision_recall(pipeline_rf=rf_pipeline, pipeline_xgb=xgb_pipeline,X=test_set, y=y_test, baseline=pr_baseline)

#Show all plots.
plt.show()

