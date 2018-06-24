def feat_select_important(alg,X,y, min_threshold=0.025):
"""
	Function to select important features based on cross validation accuracy
	
	Inputs:
			- alg: classification model to use in finding feature importances
			- X: feature set
			- y: target variables
			- min_threshold: cutoff threshold of importance. (default = 0.025)
	Outputs:
			- prints accuracy, roc_auc, and number of features
			- feature_names - most important features
"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Fit model using importances over 0.005 as a threshold
    thresholds = sort(alg.feature_importances_)
    thresholds = thresholds[thresholds>=0.025]
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(alg, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = GradientBoostingClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predict_proba = selection_model.predict_proba(select_X_test)[:,1]
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        roc_auc = metrics.roc_auc_score(y_test, predict_proba)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], roc_auc*100))
		
	feature_idx = selection.get_support()
	feature_names = X.columns[feature_idx]
	pd.DataFrame(feature_names)