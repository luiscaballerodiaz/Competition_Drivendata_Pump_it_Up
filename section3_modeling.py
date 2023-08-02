import pandas as pd
import numpy as np
import time
import tarea_functions as f
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS_mlxtend


def get_sim_params(algorithm, sweep_en, seed):
    grid_params = ['']
    if 'gradient' in algorithm.lower():
        grid_params = [{'preprocess': [''], 'estimator': ['gradient boosting'], 'estimator__n_estimators': [100],
                        'estimator__max_depth': [8], 'estimator__learning_rate': [0.2],
                        'estimator__random_state': [seed]}]
    elif 'extra' in algorithm.lower():
        grid_params = [{'preprocess': [''], 'estimator': ['extra'], 'estimator__n_estimators': [300],
                        'estimator__min_samples_leaf': [25], 'estimator__max_depth': [20],
                        'estimator__max_leaf_nodes': [100], 'estimator__max_features': ['sqrt'],
                        'estimator__bootstrap': [False], 'estimator__n_jobs': [-1], 'estimator__random_state': [seed]}]
    elif 'random' in algorithm.lower() or 'forest' in algorithm.lower():
        grid_params = [{'preprocess': [''], 'estimator': ['random forest'], 'estimator__n_jobs': [-1],
                        'estimator__n_estimators': [300], 'estimator__max_depth': [15],
                        'estimator__max_features': [70], 'estimator__random_state': [seed]}]
    elif 'lightgbm' in algorithm.lower() or 'light' in algorithm.lower():
        grid_params = [{'preprocess': [''], 'estimator': ['lightgbm'], 'estimator__n_estimators': [300],
                        'estimator__learning_rate': [0.1], 'estimator__num_leaves': [130], 'estimator__max_depth': [20],
                        'estimator__boosting_type': ['dart'], 'estimator__reg_alpha': [0.05],
                        'estimator__reg_lambda': [0.05], 'estimator__n_jobs': [-1], 'estimator__random_state': [seed]}]
    elif 'xgb' in algorithm.lower():
        grid_params = [{'preprocess': [''], 'estimator': ['xgb'], 'estimator__n_estimators': [80],
                        'estimator__n_jobs': [-1], 'estimator__learning_rate': [0.075], 'estimator__max_leaves': [30],
                        'estimator__random_state': [seed]}]
    elif 'logreg' in algorithm.lower() or 'logistic' in algorithm.lower() or 'regression' in algorithm.lower():
        grid_params = [{'preprocess': ['std'], 'estimator': ['logreg'], 'estimator__random_state': [seed],
                        'estimator__penalty': ['l1'], 'estimator__C': [1], 'estimator__solver': ['saga'],
                        'estimator__n_jobs': [-1]}]
    elif 'mlp' in algorithm.lower():
        grid_params = [{'preprocess': ['std'], 'estimator': ['mlp'], 'estimator__random_state': [seed],
                        'estimator__alpha': [0.1], 'estimator__activation': ['relu'],
                        'estimator__hidden_layer_sizes': [256]}]
    elif 'linear svc' in algorithm.lower() or 'linearsvc' in algorithm.lower():
        grid_params = [{'preprocess': ['std'], 'estimator': ['linearsvc'], 'estimator__C': [0.1],
                        'estimator__penalty': ['l1'], 'estimator__random_state': [seed], 'estimator__dual': [False]}]
    elif 'svm' in algorithm.lower():
        grid_params = [{'preprocess': ['std'], 'estimator': ['svm'], 'estimator__random_state': [seed],
                        'estimator__gamma': [0.005], 'estimator__C': [50]}]

    # Create estimator for sequential feature selection and improving features with decision trees
    params_model = f.create_model(grid_params[0]['estimator'][0])
    parms = {}
    for key, value in grid_params[0].items():
        if 'estimator__' in key:
            key = key.replace('estimator__', '')
            parms[key] = value[0]
    params_model.set_params(**parms)

    if sweep_en == 1:
        if 'gradient' in algorithm.lower():
            grid_params = [{'preprocess': [''], 'estimator': ['gradient boosting'], 'estimator__random_state': [seed],
                            'estimator__n_estimators': [25, 35, 50, 65, 80, 100], 'estimator__max_depth': [3, 4, 5, 6],
                            'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.35, 0.5]}]
        elif 'extra' in algorithm.lower():
            grid_params = [{'preprocess': [''], 'estimator': ['extra'], 'estimator__n_estimators': [100, 200, 300],
                            'estimator__min_samples_leaf': [25, 40, 60, 80, 100], 'estimator__n_jobs': [-1],
                            'estimator__max_depth': [10, 16, 23, 30, 40], 'estimator__max_features': ['sqrt'],
                            'estimator__max_leaf_nodes': [25, 40, 60, 80, 100], 'estimator__bootstrap': [False],
                            'estimator__random_state': [seed]}]
        elif 'random' in algorithm.lower() or 'forest' in algorithm.lower():
            grid_params = [{'preprocess': [''], 'estimator': ['random forest'], 'estimator__n_jobs': [-1],
                            'estimator__n_estimators': [100, 200, 300], 'estimator__max_depth': [5, 8, 12, 16, 20, 25],
                            'estimator__max_features': [30, 40, 50, 60, 70, 80]}]
        elif 'lightgbm' in algorithm.lower() or 'light' in algorithm.lower():
            grid_params = [{'preprocess': [''], 'estimator': ['lightgbm'],
                            'estimator__n_estimators': [300], 'estimator__learning_rate': [0.1, 0.2],
                            'estimator__num_leaves': [35, 60, 85, 110, 135],
                            'estimator__max_depth': [10, 20, 30, 40],
                            'estimator__boosting_type': ['dart'], 'estimator__reg_alpha': [0.05],
                            'estimator__reg_lambda': [0.05], 'estimator__n_jobs': [-1],
                            'estimator__random_state': [seed]}]
        elif 'xgb' in algorithm.lower():
            grid_params = [{'preprocess': [''], 'estimator': ['xgb'], 'estimator__random_state': [seed],
                            'estimator__n_estimators': [25, 35, 50, 65, 80, 100],
                            'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.35, 0.5],
                            'estimator__max_leaves': [15, 25, 35, 50], 'estimator__n_jobs': [-1]}]
        elif 'logreg' in algorithm.lower() or 'logistic' in algorithm.lower() or 'regression' in algorithm.lower():
            grid_params = [{'preprocess': ['std'], 'estimator': ['logreg'], 'estimator__random_state': [seed],
                            'estimator__penalty': ['l1'], 'estimator__C': [0.01, 0.1, 0.5, 1, 2.5, 5, 7.5, 10, 25, 100],
                            'estimator__solver': ['saga'], 'estimator__n_jobs': [-1]},
                           {'preprocess': ['std'], 'estimator': ['logreg'], 'estimator__random_state': [seed],
                            'estimator__penalty': ['l2'], 'estimator__C': [0.01, 0.1, 0.5, 1, 2.5, 5, 7.5, 10, 25, 100],
                            'estimator__solver': ['saga', 'lbfgs', 'liblinear'], 'estimator__n_jobs': [-1]}]
        elif 'mlp' in algorithm.lower():
            grid_params = [{'preprocess': ['std'], 'estimator': ['mlp'], 'estimator__random_state': [seed],
                            'estimator__alpha': [0.01, 0.1, 0.5, 1, 2.5, 5, 10], 'estimator__activation': ['relu'],
                            'estimator__hidden_layer_sizes': [64, 128, 256, [128, 64]]}]
        elif 'linear svc' in algorithm.lower() or 'linearsvc' in algorithm.lower():
            grid_params = [{'preprocess': ['std'], 'estimator': ['linearsvc'], 'estimator__random_state': [seed],
                            'estimator__C': [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 2.5, 5, 7.5, 10, 25, 100],
                            'estimator__penalty': ['l1', 'l2'], 'estimator__dual': [False]}]
        elif 'svm' in algorithm.lower():
            grid_params = [{'preprocess': ['std'], 'estimator': ['svm'], 'estimator__random_state': [seed],
                            'estimator__gamma': [0.001, 0.01, 0.1, 1, 10],
                            'estimator__C': [0.01, 0.1, 1, 5, 10, 50, 100]}]
    return grid_params, params_model


def user_features(x_tr, x_ts, feats):
    print('\nUSER FEATURES ENABLED!!!\n')
    feat_list = x_tr.columns.values.tolist()
    feats_to_remove = [x for x in feat_list if x not in feats]
    x_tr.drop(feats_to_remove, inplace=True, axis=1)
    x_ts.drop(feats_to_remove, inplace=True, axis=1)
    return x_tr, x_ts


def select_features(x_tr, x_ts, y_tr, sfs_model, kfeat, sco):
    print('\nSELECT FEATURES ALGORITHM ENABLED!!!\n')
    time0 = time.time()
    feat_list = x_tr.columns.values.tolist()
    x_tr = np.array(x_tr)
    x_ts = np.array(x_ts)
    print('\nX TRAIN SHAPE for sequential feature selection: {}'.format(x_tr.shape))
    # Scaling the numerical features
    scale = StandardScaler()
    x_train_std = scale.fit_transform(x_tr)
    # Sequential feature selection
    sfs = SFS_mlxtend(sfs_model, cv=5, k_features=kfeat, scoring=sco, floating=True, forward=True, verbose=2, n_jobs=-1)
    sfs.fit(x_train_std, y_tr)
    x_tr = sfs.transform(x_tr)
    x_ts = sfs.transform(x_ts)
    opt_feats = [feat_list[x] for x in sfs.k_feature_idx_]
    print('\nSequential feature selection time: {:.1f}\n'.format(time.time() - time0))  # Calculate SFS timing
    return x_tr, x_ts, opt_feats


def pipeline_gridsearch(x_tr, x_ts, y_tr, y_ts, params, sco, cv_rep, seed):
    time0 = time.time()
    param_grid = f.decode_gridsearch_params(params)
    pipe = Pipeline([('preprocess', []), ('estimator', [])])
    cv = RepeatedKFold(n_splits=5, n_repeats=cv_rep, random_state=seed)  # Define the fold performance
    grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring=sco)  # Define grid search cross validation
    grid_search.fit(x_tr, y_tr)  # Fit grid search for training set
    # Save a CSV file with the results for the grid search
    grid_results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    print(grid_results)
    model = grid_search.best_estimator_  # Create model with best parametrization
    print("\nBEST MODEL PARAMETERS:\n{}".format(grid_search.best_params_))  # Show best parameters
    cm = confusion_matrix(y_ts, model.predict(x_ts))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.savefig('Confusion Matrix.png', bbox_inches='tight')
    plt.close()
    print("\nBEST MODEL CROSS VALIDATION SCORE: {:.4f}".format(grid_search.best_score_))  # Show best scores
    print('\nBEST MODEL TRAIN SCORE: {:.4f}'.format(model.score(x_tr, y_tr)))
    print('\nBEST MODEL TEST SCORE: {:.4f}'.format(model.score(x_ts, y_ts)))
    print('\nCONFUSION MATRIX:\n{}'.format(cm))
    print('\nGrid search time: {:.1f}\n'.format(time.time() - time0))  # Calculate grid search timing
    try:
        index = np.argsort(-model['estimator'].feature_importances_)
        print('Important TRAIN feats: {}'.format([x_tr.columns.values.tolist()[x] for x in index[:100]]))
    except (Exception,):
        pass
    return grid_results, model


def generate_submission(model, x_tr, y_tr, x_ts, id_ts):
    model.fit(x_tr, y_tr)
    df_submission = id_ts.to_frame()
    df_submission['status_group'] = model.predict(x_ts)
    df_submission['status_group'].replace({2: 'functional', 1: 'functional needs repair', 0: 'non functional'},
                                          inplace=True)
    df_submission.to_csv('Submission.csv', index=False)
    try:
        index = np.argsort(-model['estimator'].feature_importances_)
        print('Important ALL feats: {}'.format([x_tr.columns.values.tolist()[x] for x in index[:100]]))
    except (Exception,):
        pass
