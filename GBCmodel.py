# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:29:30 2018

@author: markl
"""

#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# Read in Data
root = 'C:/Users/markl/OneDrive/Documents/GG/'
sql = pd.read_csv(root + 'sql_data.csv')

# Try changing from 4 classes to 3 classes
for index,data in sql.iterrows():
    if sql.loc[index,'Class'] == 1: 
        sql.loc[index,'class'] = 1
    elif sql.loc[index,'Class'] == 2: 
        sql.loc[index,'class'] = 1    
    elif sql.loc[index,'Class'] == 3: 
        sql.loc[index,'class'] = 0
    elif sql.loc[index,'Class'] == 4: 
        sql.loc[index,'class'] = 0
        
X = sql.iloc[:,1:-2]
X = X.fillna(0)
feature_names = list(X.columns)
y = sql.iloc[:,-1]



# Separate into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Define a function to fine-tune the parameters of a model
def modelfit(alg, X, y, feature_names, performCV=True, 
             printFeatureImportance=True, cv_folds=3):
    #Fit the algorithm on the data
    alg.fit(X, y)
        
    #Predict training set:
    predictions = alg.predict(X)
    predict_proba = alg.predict_proba(X)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, X, y, cv=cv_folds, \
                                                    scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(y.values, predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y, predict_proba)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
        % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, \
                             feature_names).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances',fontsize=24)
        plt.ylabel('Feature Importance Score',fontsize=24)
        
 # 1) Build a baselines model using default values (no tuning)
gbc0 = GradientBoostingClassifier(random_state=10)       
modelfit(gbc0, X_train, y_train, feature_names)


# 2) Fix learnning rate and determine optimum number of trees:
#   Notes:
#   - Start with relatively high learning rate (default = 0.1). Somewhere 
#   between 0.05 and 0.2 should work.
#   - Choose a value which has high accuracy but still allows system to work 
#   fast, as this number of trees will be used for tuning tree parameters
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = \
            GradientBoostingClassifier(learning_rate=0.1, \
             min_samples_split=2,min_samples_leaf=2,max_depth=20, \
              max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=4)
gsearch1.fit(X_train,y_train)
# Check output
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# 3) Tune tree-specific parameters
#   Notes: order of tuning variables is not trivial. Start with variables with
#   a higher impact on outcome first

# 3A) Tune max_depth and num_samples_split
param_test2 = {'max_depth':range(2,20,1), 'min_samples_split':range(2,20,1)}
gsearch2 = GridSearchCV(estimator = \
            GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, \
             max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=4)
gsearch2.fit(X, y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

# 3B) Tune min_samples_split and min_samples_leaf
param_test3 = {'min_samples_split':range(2,12,1), 'min_samples_leaf':range(2,10,1)}
gsearch3 = GridSearchCV(estimator = \
            GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, \
             max_depth=9,max_features='sqrt', subsample=0.8, random_state=10),
param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=4)
gsearch3.fit(X,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# 3C) Tune max_features
param_test4 = {'max_features':range(2,25,2)}
gsearch4 = GridSearchCV(estimator = \
            GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, \
             max_depth=9, min_samples_split=9, min_samples_leaf=9, \
              subsample=0.8, random_state=10),
param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=4)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# 4) Tune subsample and make models with lower learning rate
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = \
            GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, \
             max_depth=9,min_samples_split=9, min_samples_leaf=2, \
              subsample=0.8, random_state=10,max_features=17),
param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=4)
gsearch5.fit(X,y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# 5) Now that we have parameters at optimum values, lower the learning rate
# and increase the number of estimators proportionally. Try decreasing learning
# rate in half (0.1->0.05) with twice (60->120) the trees
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,\
            max_depth=9, min_samples_split=9,min_samples_leaf=2, subsample=0.80,\
            random_state=10, max_features=17)
modelfit(gbm_tuned_1, X, y, feature_names)

gbm_tuned_1.fit(X_train,y_train)
gbm_tuned_1.predict(X_test)
y_test
