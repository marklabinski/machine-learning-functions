#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 24, 8

feat_imp=[] 
predictions=[]
predict_proba=[] 
accuracy=[]
roc_auc =[]
# Define a function to fine-tune the parameters of a model
def modelfit(alg, X, y, feature_names, performCV=True, 
             printFeatureImportance=True, cv_folds=3):
    """
    Function to fit a model using Stratified KFold cross validation
    
    Inputs:
            alg: 
                - scikit-learn algorithm of choice 
            X:
                - Feature set. Best practice is to split X, y into training and testing
                  sets. This can be done using scikit-learn train_test_split
                  X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)
                  Then use X_train, y_train in modelfit function
            y:
                - Targets. Best practice - use y_train for modelfit

            feature_names:
                - List of feature names. Can use feature_names=list(X.columns)

            performCV:
                - Default = True. Performs stratified kfold cross validation

            printFeatureImportance:
                - Default = True. Prints bar chart of feature importances

            cv_folds:
                - Default = 3. Number of folds in KFold cross validation
    Output:
            predictions:
                - Predicted labels 
            predict_proba:
                - Predicted probabilities
            accuracy:
                - Accuracy score
            roc_auc:
                - ROC_AUC score
            feat_imp:
                - Feature importance values
    Example:
        gbc = GradientBoostingClassifier(random_state=10)
        modelfit(gbc,X_train,y_train,feature_names)

    """
    #Fit the algorithm on the data
    alg.fit(X, y)

    #Predict training set:
    predictions = alg.predict(X)
    predict_proba = alg.predict_proba(X)[:,1]

    #Perform cross-validation:  
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, X, y, cv=cv_folds, \
                                                    scoring='roc_auc')

    accuracy = metrics.accuracy_score(y.values, predictions)
    roc_auc = metrics.roc_auc_score(y, predict_proba)
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % accuracy
    print "AUC Score (Train): %f" % roc_auc

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
        % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, \
                             feature_names).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances',fontsize=10)
        plt.ylabel('Feature Importance Score',fontsize=18)
    
