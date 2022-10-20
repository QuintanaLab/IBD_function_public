# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 18:19:21 2021

@author: Gummy (Chien-Jung Huang)
"""

# Based
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.impute import KNNImputer
from scipy.spatial import distance
from scipy.stats import kruskal, mannwhitneyu
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# RF model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def filtering_testing_compounds(ac50_M_train_done, ac50_M_test_done, label, p=0.05, bonferroni_step=True, adjp=0.05):
    # Filter low correlated test compounds
    # (Bonferroni-corrected p-value < 0.05, Spearman correlation) with IBD compounds (training data)
    compounds = []
    for i in range(len(ac50_M_test_done.index)):
        pvalues = []
        for j in range(len(ac50_M_train_done.index)):
            train = ac50_M_train_done.iloc[j, 1:].tolist()
            test = ac50_M_test_done.iloc[i, 1:].tolist()
            try:
                pvalues.append(spearmanr(train, test, nan_policy='omit')[1])
            except:
                pvalues.append(1)
        if bonferroni_step == True:
            bonferroni = multipletests(pvalues, adjp, method='bonferroni')[0].tolist()
            pass_id = ac50_M_train_done.index[bonferroni].tolist()
        else:
            pass_id = ac50_M_train_done.index[[x < p for x in pvalues]].tolist()
        pass_L = label.loc[pass_id]['effect in IBD zebrafish model'].tolist()
        if (pass_L.count('IBD ameliorating') > 2) or (pass_L.count('IBD promoting') > 6) or (pass_L.count('No effect') > 16):
            compounds.append(ac50_M_test_done.index[i])
    ac50_M_test_done = ac50_M_test_done.loc[compounds]
    return(ac50_M_test_done)



def AC50_M_dropna(ac50_M, label, percentage=0.75, filtering_by_3groups=False, filtering_only_2step=False):
    # Training data
    ac50_M_train = ac50_M[ac50_M.index.isin(label.index)] #49x1569
    # Testing data
    ac50_M_test = ac50_M[ac50_M.index.isin(label.index) != True] #9249x1569
    
    # Filter training data (first filtering)
    # The bioassays with more than X% missing values in training data
    bioassays = []
    for i in range(len(ac50_M_train.columns)):
        if len(ac50_M_train.iloc[:, i][ac50_M_train.iloc[:, i].isnull()]) > (len(ac50_M_train.index)*percentage):
            bioassays.append(ac50_M_train.columns[i])    
    # First filter results
    ac50_M_train_1f = ac50_M_train.drop(columns=bioassays)
    ac50_M_test_1f = ac50_M_test.drop(columns=bioassays)
    ac50_M_1f = ac50_M.drop(columns=bioassays)
    
    if filtering_by_3groups == True:
        # The bioassays with more than X% missing values in training data by each group
        IBD_good = ac50_M_train.loc[label.index[label.loc[:, 'effect in IBD zebrafish model'] == 'IBD ameliorating'], :]
        IBD_bad = ac50_M_train.loc[label.index[label.loc[:, 'effect in IBD zebrafish model'] == 'IBD promoting'], :]
        IBD_no = ac50_M_train.loc[label.index[label.loc[:, 'effect in IBD zebrafish model'] == 'No effect'], :]
        
        bioassays = []
        for i in range(len(IBD_good.columns)):
            if len(IBD_good.iloc[:, i][IBD_good.iloc[:, i].isnull()]) > (4*percentage):
                bioassays.append(IBD_good.columns[i])
        for i in range(len(IBD_bad.columns)):
            if len(IBD_bad.iloc[:, i][IBD_bad.iloc[:, i].isnull()]) > (13*percentage):
                bioassays.append(IBD_bad.columns[i])
        for i in range(len(IBD_no.columns)):
            if len(IBD_no.iloc[:, i][IBD_no.iloc[:, i].isnull()]) > (32*percentage):
                bioassays.append(IBD_no.columns[i])    
        bioassays = list(set(bioassays))

        # First filter results (filtering_by_3groups)
        ac50_M_train_1f = ac50_M_train.drop(columns=bioassays)
        ac50_M_test_1f = ac50_M_test.drop(columns=bioassays)
        ac50_M_1f = ac50_M.drop(columns=bioassays)
    
    # Filter AC50 matrix (Second filtering)
    # The compounds/bioassays with more than X% missing values
    # compounds
    compounds = []
    for i in range(len(ac50_M_test_1f.index)):
        if len(ac50_M_test_1f.iloc[i, 1:][ac50_M_test_1f.iloc[i, 1:].isnull()]) > ((len(ac50_M_test_1f.columns)-1)*percentage):
            compounds.append(ac50_M_test_1f.index[i])      
    # Second filter results after filter compounds
    ac50_M_test_2f = ac50_M_test_1f.drop(index=compounds)
    ac50_M_2f = ac50_M_1f.drop(index=compounds)
    
    # bioassays
    bioassays = []
    for i in range(len(ac50_M_2f.columns)):
        if len(ac50_M_2f.iloc[:, i][ac50_M_2f.iloc[:, i].isnull()]) > ((len(ac50_M_2f.index))*percentage):
            bioassays.append(ac50_M_2f.columns[i])
    # Third filter results after filter bioassays
    ac50_M_train_2f = ac50_M_train_1f.drop(columns=bioassays)
    ac50_M_test_2f1 = ac50_M_test_2f.drop(columns=bioassays)
    ac50_M_2f1 = ac50_M_2f.drop(columns=bioassays)

    if filtering_only_2step == True:
        return(ac50_M_2f, ac50_M_train_1f, ac50_M_test_2f)
    else:
        return(ac50_M_2f1, ac50_M_train_2f, ac50_M_test_2f1)



def KNN_imputer(ac50_M_f, label):
    # Imputation for completing missing values using k-Nearest Neighbors
    imputer = KNNImputer(n_neighbors=5)
    ac50_M_f.iloc[:, 1:] = imputer.fit_transform(ac50_M_f.iloc[:, 1:])
    ac50_M_train_done = ac50_M_f[ac50_M_f.index.isin(label.index)]
    ac50_M_test_done = ac50_M_f[ac50_M_f.index.isin(label.index) != True]
    
    return(ac50_M_f, ac50_M_train_done, ac50_M_test_done)



def undersampling_by_eigen(ac50_M_train_done, label, numbers_of_under, method= 'euclidean_d', out_path=''):
    # 32 no effect compounds
    no_effect32 = ac50_M_train_done.loc[label[label['effect in IBD zebrafish model'] == 'No effect'].index, :]
    
    # Find eigenvector
    A = np.matrix(no_effect32.iloc[:, 1:].T)
    U, S, W = np.linalg.svd(A)
    U.shape, S.shape, W.shape
    eigen_compounds_svd = U[:, 0]
    weighting = np.median([np.median(no_effect32.iloc[:, i].tolist())/eigen_compounds_svd[i-1, 0] for i in range(1, len(no_effect32.columns))])
    eigen_compounds_svd = eigen_compounds_svd*weighting
    
    if out_path == '':
        plt.figure(figsize = (10, 8))
        for i in no_effect32.index:
            plt.plot(range(1, len(no_effect32.columns)), no_effect32.loc[i][1:], color='grey')
        plt.plot(range(1, len(no_effect32.columns)), eigen_compounds_svd, color='black', label='eigen-compound')
        plt.xlabel('Bioassays (' + str(len(ac50_M_train_done.columns)-1) + ')')
        plt.ylabel('Abundance for 32 no effect compounds')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize = (10, 8))
        for i in no_effect32.index:
            plt.plot(range(1, len(no_effect32.columns)), no_effect32.loc[i][1:], color='grey')
        plt.plot(range(1, len(no_effect32.columns)), eigen_compounds_svd, color='black', label='eigen-compound')
        plt.xlabel('Bioassays (' + str(len(ac50_M_train_done.columns)-1) + ')')
        plt.ylabel('Abundance for 32 no effect compounds')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(out_path + 'Abundance of no effect compounds and eigenvector(svd).tif', dpi=300, bbox_inches='tight')

    # Distance between no effect compounds and eigenevector
    if method == 'euclidean_d':
        eu_distance = []
        for i in no_effect32.index:
            eu_distance.append(distance.euclidean(eigen_compounds_svd, no_effect32.loc[i][1:]))
        eu_distance_df = pd.DataFrame({'compounds': no_effect32.index, 'euclidean_d': eu_distance})
        eu_distance_df = eu_distance_df.sort_values(by='euclidean_d').reset_index(drop=True)
        top_no = eu_distance_df.iloc[:numbers_of_under, :]['compounds'].tolist()
    if method == 'spearman_corr':
        sp_corr = []
        for i in no_effect32.index:
            sp_corr.append(abs(spearmanr(eigen_compounds_svd, no_effect32.loc[i][1:], nan_policy='omit')[0]))
        sp_corr_df = pd.DataFrame({'compounds': no_effect32.index, 'spearman_corr': sp_corr})
        sp_corr_df = sp_corr_df.sort_values(by='spearman_corr', ascending=False).reset_index(drop=True)
        top_no = sp_corr_df.iloc[:numbers_of_under, :]['compounds'].tolist()
    
    # 32 no effect compounds (+eigen compounds)
    pca = PCA(n_components=2)
    no_effect32 = ac50_M_train_done.loc[label[label['effect in IBD zebrafish model'] == 'No effect'].index, :]
    new_row = dict(zip(no_effect32.iloc[:, 1:].columns.tolist(), list(eigen_compounds_svd)))
    new_row = pd.Series(data=new_row, name='000')
    no_effect32 = no_effect32.append(new_row)
    new_data = pca.fit_transform(no_effect32.iloc[:, 1:])
    top_index = no_effect32.reset_index()[no_effect32.reset_index()['CASRN'].isin(top_no)].index.tolist()
    test_df = no_effect32.reset_index()
    
    if out_path == '':
        plt.figure(figsize = (10, 8))
        for i in range(len(new_data)-1):
            plt.text(new_data[i, 0], new_data[i, 1], no_effect32.index[i], color='blue', fontsize=8, horizontalalignment='center')
        for i in top_index:
            plt.text(new_data[i, 0], new_data[i, 1], test_df.loc[i]['CASRN'], color='red', fontsize=8, horizontalalignment='center')
        plt.text(new_data[-1, 0], new_data[-1, 1]-0.05, 'eigen-compound', color='red', horizontalalignment='center', verticalalignment='bottom')
        plt.plot(new_data[:, 0], new_data[:, 1], color='grey', marker='o', markerfacecolor='grey', linestyle ='None', label='No effect')
        plt.xlabel("principal component 1({}%)".format(round(pca.explained_variance_ratio_[0]*100,1)))
        plt.ylabel("principal component 2({}%)".format(round(pca.explained_variance_ratio_[1]*100,1)))
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize = (10, 8))
        for i in range(len(new_data)-1):
            plt.text(new_data[i, 0], new_data[i, 1], no_effect32.index[i], color='blue', fontsize=8, horizontalalignment='center')
        for i in top_index:
            plt.text(new_data[i, 0], new_data[i, 1], test_df.loc[i]['CASRN'], color='red', fontsize=8, horizontalalignment='center')
        plt.text(new_data[-1, 0], new_data[-1, 1]-0.05, 'eigen-compound', color='red', horizontalalignment='center', verticalalignment='bottom')
        plt.plot(new_data[:, 0], new_data[:, 1], color='grey', marker='o', markerfacecolor='grey', linestyle ='None', label='No effect')
        plt.xlabel("principal component 1({}%)".format(round(pca.explained_variance_ratio_[0]*100,1)))
        plt.ylabel("principal component 2({}%)".format(round(pca.explained_variance_ratio_[1]*100,1)))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(out_path + 'PCA_trainingset(No_effect)+eigenc(svd).tif', dpi=300, bbox_inches='tight')
    return(top_no)



def feature_selection(X, y, method='kruskal', selection_by_3time=False):
    if selection_by_3time == True:
        if method == 'kruskal':
            sig_features_bg = []
            sig_features_bn = []
            sig_features_gn = []
            for i in X.columns[1:]:
                IBD_bad = X.loc[y.index[y == 'IBD ameliorating'], i].tolist()
                IBD_good = X.loc[y.index[y == 'IBD promoting'], i].tolist()
                IBD_no = X.loc[y.index[y == 'No effect'], i].tolist()
                if kruskal(IBD_bad, IBD_good, nan_policy='omit')[1] < 0.05:
                    sig_features_bg.append(i)
                if kruskal(IBD_bad, IBD_no, nan_policy='omit')[1] < 0.05:
                    sig_features_bn.append(i)
                if kruskal(IBD_good, IBD_no, nan_policy='omit')[1] < 0.05:
                    sig_features_gn.append(i)
            sig_features = sorted(list(set(sig_features_bg+sig_features_bn+sig_features_gn)))
        elif method == 'mannwhitneyu':
            sig_features_bg = []
            sig_features_bn = []
            sig_features_gn = []
            for i in X.columns[1:]:
                IBD_bad = X.loc[y.index[y == 'IBD ameliorating'], i].tolist()
                IBD_good = X.loc[y.index[y == 'IBD promoting'], i].tolist()
                IBD_no = X.loc[y.index[y == 'No effect'], i].tolist()
                if mannwhitneyu(IBD_bad, IBD_good)[1] < 0.05:
                    sig_features_bg.append(i)
                if mannwhitneyu(IBD_bad, IBD_no)[1] < 0.05:
                    sig_features_bn.append(i)
                if mannwhitneyu(IBD_good, IBD_no)[1] < 0.05:
                    sig_features_gn.append(i)
            sig_features = sorted(list(set(sig_features_bg+sig_features_bn+sig_features_gn)))
    else:
        sig_features = []
        for i in X.columns[1:]:
            IBD_bad = X.loc[y.index[y == 'IBD ameliorating'], i].tolist()
            IBD_good = X.loc[y.index[y == 'IBD promoting'], i].tolist()
            IBD_no = X.loc[y.index[y == 'No effect'], i].tolist()
            if kruskal(IBD_bad, IBD_good, IBD_no, nan_policy='omit')[1] < 0.05:
                sig_features.append(i)
        sig_features = sorted(list(set(sig_features)))
    return(sig_features)



def RF_model_tuning(X, y, sig_features):
    X = X.loc[:, sig_features]
    # number of trees
    n_estimators = [100, 500, 1000]
    # max number of features to  consider at every split
    # max_features = ['auto', 'sqrt', 'log2']
    # max number of levels in tree
    max_depth = [None, 10, 30, 50]
    # minimum number of samples required to split a node
    min_samples_split = [2, 11, 20, 28]
    # minimum number of samples required at each leaf node
    min_samples_leaf = [1, 10, 19, 27]
    grid_param = {'n_estimators': n_estimators,
                  # 'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    RFR = RandomForestClassifier(random_state=0, n_jobs=-1)
    RFR_random = GridSearchCV(estimator=RFR, param_grid=grid_param, 
                              scoring=['accuracy', 'f1_micro', 'precision_micro', 'recall_micro'], 
                              refit='accuracy', cv=LeaveOneOut(), verbose=2, n_jobs=-1)
    RFR_random.fit(X, y)
    return(RFR_random.best_params_)



def RF_model_best_predict(X, y, sig_features, best_parameter, ac50_M_test_done, seed):
    X = X.loc[:, sig_features]
    kf = LeaveOneOut()
    y_pred = []
    for train, test in kf.split(X, y):
        clf = RandomForestClassifier(n_estimators=best_parameter['n_estimators'], 
                                 min_samples_split=best_parameter['min_samples_split'], 
                                 min_samples_leaf=best_parameter['min_samples_leaf'], 
                                 # max_features=best_parameter['max_features'], 
                                 max_depth=best_parameter['max_depth'], 
                                 random_state=seed, n_jobs=-1)
        clf.fit(X.iloc[train, :], y[train])
        y_pred.append(clf.predict(X.iloc[test, :])[0])
    predict_acc = accuracy_score(y.tolist(), y_pred)
    predict_conm = confusion_matrix(y.tolist(), y_pred, labels=['IBD ameliorating', 'IBD promoting', 'No effect'])
    return(predict_acc, predict_conm)



def pca_for_features(X, y, sig_features, label, out_path=''):
    X = X.loc[:, sig_features]
    names = y.reset_index().merge(label.reset_index(), left_on='casn', right_on='CASRN')['ChemName'].tolist()
    
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(X)
    
    if out_path == '':
        plt.figure(figsize = (10, 8))
        for i in range(len(new_data)):
            if names[i] == 'Phenylbutazone':
                plt.text(new_data[i, 0]-0.1, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Tetrachlorophthalic anhydride':
                plt.text(new_data[i, 0]+0.3, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Linuron':
                plt.text(new_data[i, 0]+0.35, new_data[i, 1]+0.05, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Di(2-ethylhexyl) phthalate':
                plt.text(new_data[i, 0]-0.5, new_data[i, 1], 'Di(2-ethylhexyl)', color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
                plt.text(new_data[i, 0]-0.7, new_data[i, 1]-0.1, 'phthalate', color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == 'PFNA':
                plt.text(new_data[i, 0]-0.25, new_data[i, 1]+0.05, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Hexadecyltrimethylammonium bromide':
                plt.text(new_data[i, 0]+1, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Diisopropyl phthalate':
                plt.text(new_data[i, 0]+0.5, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Heptylparaben':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == '4,4′-Dichlorodiphenyl sulfone':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == 'mono-2-Ethylhexyl phthalate':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == 'Methyl carbamate':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            else:
                plt.text(new_data[i, 0], new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
        plt.plot(new_data[:, 0][y == 'IBD ameliorating'], new_data[:, 1][y == 'IBD ameliorating'], 
                  color='blue', marker='o', markerfacecolor='None', markersize=10, linestyle ='None', label='Ameliorate intestinal inflammation')
        plt.plot(new_data[:, 0][y == 'IBD promoting'], new_data[:, 1][y == 'IBD promoting'], 
                  color='red', marker='x', markerfacecolor='red', markersize=10, linestyle ='None', label='Worsen intestinal inflammation')
        plt.plot(new_data[:, 0][y == 'No effect'], new_data[:, 1][y == 'No effect'], 
                  color='grey', marker='^', markerfacecolor='None', markersize=10, linestyle ='None', label='No effect on intestinal inflammation')
        plt.legend(loc='upper left')
        plt.xlabel("principal component 1({}%)".format(round(pca.explained_variance_ratio_[0]*100,1)))
        plt.ylabel("principal component 2({}%)".format(round(pca.explained_variance_ratio_[1]*100,1)))
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize = (10, 8))
        for i in range(len(new_data)):
            if names[i] == 'Phenylbutazone':
                plt.text(new_data[i, 0]-0.1, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Tetrachlorophthalic anhydride':
                plt.text(new_data[i, 0]+0.3, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Linuron':
                plt.text(new_data[i, 0]+0.35, new_data[i, 1]+0.05, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Di(2-ethylhexyl) phthalate':
                plt.text(new_data[i, 0]-0.5, new_data[i, 1], 'Di(2-ethylhexyl)', color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
                plt.text(new_data[i, 0]-0.7, new_data[i, 1]-0.1, 'phthalate', color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == 'PFNA':
                plt.text(new_data[i, 0]-0.25, new_data[i, 1]+0.05, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Hexadecyltrimethylammonium bromide':
                plt.text(new_data[i, 0]+1, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Diisopropyl phthalate':
                plt.text(new_data[i, 0]+0.5, new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
            elif names[i] == 'Heptylparaben':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == '4,4′-Dichlorodiphenyl sulfone':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == 'mono-2-Ethylhexyl phthalate':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            elif names[i] == 'Methyl carbamate':
                plt.text(new_data[i, 0], new_data[i, 1]+0.07, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='bottom')
            else:
                plt.text(new_data[i, 0], new_data[i, 1]-0.1, names[i], color='black', fontsize=9, horizontalalignment='center', verticalalignment='top')
        plt.plot(new_data[:, 0][y == 'IBD ameliorating'], new_data[:, 1][y == 'IBD ameliorating'], 
                  color='blue', marker='o', markerfacecolor='None', markersize=10, linestyle ='None', label='Ameliorate intestinal inflammation')
        plt.plot(new_data[:, 0][y == 'IBD promoting'], new_data[:, 1][y == 'IBD promoting'], 
                  color='red', marker='x', markerfacecolor='red', markersize=10, linestyle ='None', label='Worsen intestinal inflammation')
        plt.plot(new_data[:, 0][y == 'No effect'], new_data[:, 1][y == 'No effect'], 
                  color='grey', marker='^', markerfacecolor='None', markersize=10, linestyle ='None', label='No effect on intestinal inflammation')
        plt.legend(loc='upper left')
        plt.xlabel("principal component 1({}%)".format(round(pca.explained_variance_ratio_[0]*100,1)))
        plt.ylabel("principal component 2({}%)".format(round(pca.explained_variance_ratio_[1]*100,1)))
        plt.tight_layout()
        plt.show()
        plt.savefig(out_path + 'PCA_trainingset.pdf', bbox_inches='tight')



def RF_model_best_predict_testing(X, y, sig_features, best_parameter, ac50_M_test_done, seed, out_path=''):
    X = X.loc[:, sig_features]
    
    clf = RandomForestClassifier(n_estimators=best_parameter['n_estimators'], 
                                 min_samples_split=best_parameter['min_samples_split'], 
                                 min_samples_leaf=best_parameter['min_samples_leaf'], 
                                 # max_features=best_parameter['max_features'], 
                                 max_depth=best_parameter['max_depth'], 
                                 random_state=seed, n_jobs=-1)
    clf.fit(X, y)
    feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    ac50_M_test_result = pd.DataFrame({'casn':ac50_M_test_done.index,
                                        'pred':clf.predict(ac50_M_test_done.loc[:, feature_imp.index])})
    ac50_M_test_result = ac50_M_test_result.merge(ac50_M_test_done.reset_index()[['casn', 'chnm']], on='casn')
    if len(out_path) > 0:
        ac50_M_test_result.to_excel(out_path + 'ac50_M_test_result_pred.xlsx', index=False)
    return(ac50_M_test_result, feature_imp)



def RWR(r, P0, W):
    i = 0
    Pi = P0
    while i >= 0:
        Pi_1 = (1 - r) * W * Pi + r * P0
        if np.linalg.norm(Pi_1 - Pi) < 10 ** -6:
            break
        else:
            Pi = Pi_1
            i = i + 1
    return Pi_1



def RWR_ranking(X, y, sig_features, ac50_M_test_done, ac50_M_test_result, numbers_of_seed=13, r=0.3, out_path=''):
    X = X.loc[:, sig_features]
    
    pred_all = ac50_M_test_done.loc[:, sig_features]
    X_promoting = X[y == 'IBD promoting']
    X_other = X[y != 'IBD promoting']
    rwr_m = pd.concat((X_promoting, X_other, pred_all))
    
    # construct P0 matrix
    P0_seed = np.zeros((len(rwr_m), 1))
    P0_seed[:numbers_of_seed] = 1 / numbers_of_seed
    P0_seed = np.matrix(P0_seed)
    
    # Correlation network
    adj_M = abs(rwr_m.T.corr('spearman'))
    adj_M = adj_M.replace(1, 0)
    adj_M = adj_M.replace(np.nan, 0)
    A_M = np.matrix(adj_M.values)
    
    # row-normalized(A)
    for i in range(len(A_M)):
        A_M[i, :] = A_M[i, :] / np.sum(A_M[i, :])
    
    A_M = np.matrix(pd.DataFrame(A_M).replace(np.nan, 0))
    W_M = A_M.T
    
    # RWR
    Pi_M = RWR(r, P0_seed, W_M)
    rwr_result = pd.DataFrame({'probability': np.array(Pi_M).flatten().tolist()}, index=rwr_m.index)
    rwr_result_name = pd.concat((ac50_M_test_done['chnm'], rwr_result), axis=1, join='inner')
    rwr_result_name = rwr_result_name.loc[ac50_M_test_result['casn'][ac50_M_test_result['pred'] == 'IBD promoting'].tolist()]
    rwr_result_name = rwr_result_name.reset_index()
    rwr_result_name = rwr_result_name.sort_values(by='probability', ascending=False).reset_index(drop=True)
    rwr_result_name['rank'] = rwr_result_name['probability'].rank(method='min', ascending=False)
    if len(out_path) > 0:
        rwr_result_name.to_excel(out_path + 'rwr_result_name(all pred compounds +' + str(numbers_of_seed) + 'promoting).xlsx', index=False)
    return(rwr_result_name)




