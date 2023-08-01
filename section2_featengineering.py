import functions as f
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


def oversampling(x_tr, y_tr, seed, samp0=0, samp1=0, samp2=0):
    # Oversampling to compensate target class imbalanced
    print('\nInitial Target Distribution: \n{}'.format(y_tr.value_counts()))
    vals = y_tr.value_counts().to_dict()
    for i, samp in enumerate([samp0, samp1, samp2]):
        if samp > 0:
            vals[i] = samp
    sm = SMOTE(random_state=seed, sampling_strategy=vals)
    x_tr, y_tr = sm.fit_resample(x_tr, y_tr)
    print('\nFinal Target Distribution: \n{}'.format(y_tr.value_counts()))
    return x_tr, y_tr


def split_data(test_split, df, seed):
    y = df['Target']
    df.drop('Target', axis=1, inplace=True)
    print('\nDF SHAPE - ORIGINAL: {}'.format(df.shape))
    print('FEATURES: {}\n'.format(df.columns.values.tolist()[:100]))

    # Create training and testing sets keeping target distribution same as original
    x_tr, x_ts, y_tr, y_ts = train_test_split(df, y, test_size=test_split, shuffle=True, stratify=y, random_state=seed)
    x_tr.reset_index(drop=True, inplace=True)
    x_ts.reset_index(drop=True, inplace=True)
    y_tr.reset_index(drop=True, inplace=True)
    y_ts.reset_index(drop=True, inplace=True)
    return x_tr, x_ts, y_tr, y_ts, df, y


def data_transform(x_tr, x_ts, y_tr, cat_feats, num_feats, enc, depth, sco, seed, min_th=0.0, min_incr=0.0, deg3=0):
    x_tr[cat_feats] = x_tr[cat_feats].astype('category')
    x_ts[cat_feats] = x_ts[cat_feats].astype('category')
    x_tr[num_feats] = x_tr[num_feats].astype('float64')
    x_ts[num_feats] = x_ts[num_feats].astype('float64')

    # Calculate V cramer and pearson correlation per each feature
    x_tr, x_ts = f.remove_low_significant_feats(x_tr, x_ts, y_tr, min_th=min_th, vcramer=True, corr=True)

    feat_list = x_tr.columns.values.tolist()
    print('\nDF SHAPE - REMOVING LOW SIGNIFICANT MARGINAL FEATURES: {}'.format(x_tr.shape))
    print('FEATURES: {}\n'.format(feat_list[:100]))

    # Apply categorical encoder to categorical variables
    fcat = x_tr.select_dtypes(include='category').columns.values.tolist()
    for e in enc:
        x_tr, x_ts = f.categorical_encoder(e, fcat, x_tr, x_ts, y_train=y_tr, remove_cats=False)
    x_tr.drop(fcat, axis=1, inplace=True)
    x_ts.drop(fcat, axis=1, inplace=True)
    x_tr = x_tr.astype('float64')
    x_ts = x_ts.astype('float64')

    feat_list = x_tr.columns.values.tolist()
    print('\nDF SHAPE - AFTER APPLYING CATEGORICAL ENCODER: {}'.format(x_tr.shape))
    print('FEATURES: {}\n'.format(feat_list[:100]))

    # Calculate V cramer and pearson correlation per each feature
    x_tr, x_ts = f.remove_low_significant_feats(x_tr, x_ts, y_tr, min_th=min_th, vcramer=True, corr=True)

    feat_list = x_tr.columns.values.tolist()
    print('\nDF SHAPE - REMOVING LOW SIGNIFICANT ENCODER FEATURES: {}'.format(x_tr.shape))
    print('FEATURES: {}\n'.format(feat_list[:100]))

    # Create new features based on decision tree
    tree_model = GridSearchCV(DecisionTreeClassifier(random_state=seed), cv=5, scoring=sco,
                              param_grid={'max_depth': depth})
    combs = list(combinations(feat_list, 2))
    if deg3 == 1:
        combs += list(combinations(feat_list, 3))
    print('\nTotal combs: {}'.format(len(combs)))
    x_tr, x_ts = f.new_tree_feats(x_tr, x_ts, y_tr, combs, tree_model, min_incr)

    feat_list = x_tr.columns.values.tolist()
    print('\nDF SHAPE - AFTER NEW FEATURES CREATION: {}'.format(x_tr.shape))
    print('FEATURES: {}\n'.format(feat_list[:100]))
    return x_tr, x_ts