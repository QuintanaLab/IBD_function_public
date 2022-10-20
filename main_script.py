# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:57:51 2021

@author: Gummy (Chien-Jung Huang)
"""

"""
IBD readme! (before running the script...)

The machine learning (ML) pipeline of IBD contain two scripts and one input data (excel file).
1. main_script.py (main program)
2. IBD_function.py (the function defined by author)
3. IBD_input_data.xlsx (input data)

The following version of python packages were used in the scripts.
pandas==1.3.4
numpy==1.20.3
scikit-learn==0.24.2
scipy==1.7.1
statsmodels==0.12.2
seaborn==0.11.2
matplotlib==3.4.3
"""

from IBD_function import (filtering_testing_compounds, AC50_M_dropna, KNN_imputer, undersampling_by_eigen, feature_selection,
                          RF_model_tuning, RF_model_best_predict, pca_for_features, RF_model_best_predict_testing, RWR, RWR_ranking)

import pandas as pd
import random


# Import data
# AC50 matrix and label
label = pd.read_excel('IBD_input_data.xlsx', sheet_name='label', index_col='CASRN')

ac50_M = pd.read_excel('IBD_input_data.xlsx', sheet_name='AC50_matrix', index_col='casn')
ac50_M_id = pd.read_excel('IBD_input_data.xlsx', sheet_name='AC50_matrix_id')


# Data preprocessing
# filtering low correlated test compounds, missing values and KNN imputation
ac50_M_train = ac50_M[ac50_M.index.isin(label.index)] #49x1569
ac50_M_test = ac50_M[ac50_M.index.isin(label.index) != True] #9249x1569
ac50_M_test_f = filtering_testing_compounds(ac50_M_train, ac50_M_test, label, p=0.05, bonferroni_step=False, adjp=0.05)
ac50_M_f = pd.concat([ac50_M_train, ac50_M_test_f])

ac50_M_f, ac50_M_train, ac50_M_test = AC50_M_dropna(ac50_M_f, label, percentage=0.75, filtering_by_3groups=False, filtering_only_2step=True)
ac50_M_done, ac50_M_train_done, ac50_M_test_done = KNN_imputer(ac50_M_f, label)

# Undersampling for No effect compounds
top_no = undersampling_by_eigen(ac50_M_train_done, label, 13, method= 'euclidean_d', out_path='')


# Feature selection by kw-test
IBD_good_index = list(label[label['effect in IBD zebrafish model']=='IBD ameliorating'].index)
IBD_bad_index = list(label[label['effect in IBD zebrafish model']=='IBD promoting'].index)
X = ac50_M_train_done.loc[IBD_good_index + IBD_bad_index + top_no]  # Features
y = label.loc[X.index, :]['effect in IBD zebrafish model']  # Labels
sig_features = feature_selection(X, y, method='kruskal', selection_by_3time=False)


# Test the RF model and PCA plot
#random_seed = random.sample(range(0, 100), 1)[0]
# We randomly select the seed and the result is 21.
random_seed = 21
best_parameter = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
predict_acc, predict_conm = RF_model_best_predict(X, y, sig_features, best_parameter, ac50_M_test_done, seed=random_seed)
pca_for_features(X, y, sig_features, label, out_path='')


# Prediction and RWR ranking
ac50_M_test_result, feature_imp = RF_model_best_predict_testing(X, y, sig_features, best_parameter, ac50_M_test_done, seed=random_seed, out_path='')
rwr_result_name = RWR_ranking(X, y, sig_features, ac50_M_test_done, ac50_M_test_result, numbers_of_seed=13, r=0.3, out_path='')





