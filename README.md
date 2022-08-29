# Introduction
This repository holds the source code utilized to generate the Random Forest and XGBoost models employed in the dissertation entitled Predicting Users' Infection from Online Behaviour and Demographics in a Military Environment. The dissertation was submitted to Lancaster University for the degree of Master of Science. 

# Instructions 
Given the sensitive nature of the data and models developed, they cannot be disclosed. Nonetheless, this work can be replicated by reusing the code with different data to generate new models. To generate new models from the source code provided, first, one has to produce the training and test feature sets as described in Appendix A in the dissertation as CSV files. Then, these files should be added to the data directory. 

Afterwards, the random_forest_training.py and xgboost_training.py scripts can be executed to search the parameter space for the best models. Finally, the test_models.py script can be executed to obtain the results of the best models found by the previous scripts. The results are given in the form of confusion matrices, Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curves.
