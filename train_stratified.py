# Train with k fold stratified validation

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

name = lambda x : str(x).split('(')[0]

def train_stratified(models, X, y, add_idf=False, nsplits=3, confusion=False):
    '''
    Take a sklearn model, feature set X, target set y and number of splits to compute Stratified k-fold validation.
    Args:
        X (array):       Numpy array of features.
        y (str):         Target - Personality list.
        add_idf (bool):  Wehther to use tf-idf on CountVectorizer.
        nsplits(int):    Number of splits for cross validation.
        confusion(bool): Wether to plot confusion matrix 
        
    Returns:
        dict: Dictionary of classifiers and their cv f1-score.
    '''
    fig_i = 0
    kf = StratifiedShuffleSplit(n_splits=nsplits)
    
    # Store fold score for each classifier in a dictionnary
    dico_score = {}
    dico_score['merged'] = 0
    for model in models:
        dico_score[name(model)] = 0
    
    # Stratified Split
    for train, test in kf.split(X,y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        X_train = cntizer.fit_transform(X_train)
        X_test = cntizer.transform(X_test)
        
        # tf-idf
        if add_idf == True:
            X_train_tfidf = tfizer.fit_transform(X_train_cnt)
            X_test_tfidf = tfizer.transform(X_test_cnt)
        
            X_train = np.column_stack((X_train_tfidf.todense(), X_train))
            X_test = np.column_stack((X_test_tfidf.todense(), X_test))
        
        probs = np.ones((len(y_test), 16))
        for model in models:
            # if xgboost use dmatrix
            if 'XGB' in name(model):
                xg_train = xgb.DMatrix(X_train, label=y_train)
                xg_test = xgb.DMatrix(X_test, label=y_test)
                watchlist = [(xg_train, 'train'), (xg_test, 'test')]
                num_round = 30
                bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=6)
                preds = bst.predict(xg_test)
                probs = np.multiply(probs, preds)
                preds = np.array([np.argmax(prob) for prob in preds])
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = np.multiply(probs, model.predict_proba(X_test))
            # f1-score
            score = f1_score(y_test, preds, average='weighted')
            dico_score[name(model)] += score
            print('%s : %s' % (str(model).split('(')[0], score))
            
            if confusion == True:
                # Compute confusion matrix
                cnf_matrix = confusion_matrix(y_test, preds)
                np.set_printoptions(precision=2)
                # Plot confusion matrix
                plt.figure(fig_i)
                fig_i += 1
                plot_confusion_matrix(cnf_matrix, classes=lab_encoder.inverse_transform(range(16)), normalize=True,
                                                          title=('Confusion matrix %s' % name(model)))
        
        # product of class probabilites of each classifier 
        merged_preds = [np.argmax(prob) for prob in probs]
        score = f1_score(y_test, merged_preds, average='weighted')
        print('Merged score: %s' % score)
        dico_score['merged'] += score
        
    return {k: v / nsplits for k, v in dico_score.items()}