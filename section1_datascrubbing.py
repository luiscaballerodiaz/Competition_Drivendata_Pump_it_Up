import pandas as pd
import numpy as np
import functions as f
from sklearn.preprocessing import PowerTransformer
from feature_engine.discretisation import DecisionTreeDiscretiser
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def dataframe_overview(df):
    print('\nDATASET ASSESSMENT:\n')
    print(f'GENERAL INFO\n{df.info()}\n'
          f'\nNUMBER UNIQUES VALUES\n{df.nunique()}\n'
          f'\nSTATS FOR NUMERICAL FEATURES\n{df.describe()}\n'
          f'\nSTATS FOR CATEGORICAL FEATURES\n{df.describe(exclude=np.number)}\n')
    target_dist = df['Target'].value_counts(normalize=True)
    print('\nTarget distribution: \n{}'.format(target_dist))


def num_feat_engineering(df, num_feats, binning_tree=None, models_predefined=None, num_transformations=None, seed=0):
    print('\n')
    fnum = num_feats.copy()

    # %%%%%%%%%%%% NUMERICAL FEATURES %%%%%%%%%%%%%
    # FEATURES TO DROP
    feat = 'id'
    df.drop(feat, axis=1, inplace=True)
    fnum.remove(feat)

    # DATETIME FEATURES
    feat = 'date_recorded'
    df[feat] = pd.to_datetime(df[feat])
    df['X'] = (pd.to_datetime("2022-12-31") - df[feat]).dt.days
    df[feat] = df['X']
    df.drop('X', axis=1, inplace=True)
    fnum.append(feat)
    df = f.manage_outliers(df, feat)

    # CONVERT 0 TO NAN
    df['latitude'].replace({-2E-08: 0}, inplace=True)
    print('Relation between zero values in longitude, latitude and gps_height: {}'.format(
        df.loc[(df['longitude'] == 0) & (df['gps_height'] == 0) & (df['latitude'] == 0), :].shape))
    feats = ['longitude', 'latitude', 'gps_height', 'population', 'construction_year']
    for feat in feats:
        print('Number of zeros in {}: {}'.format(feat, df.loc[df[feat] == 0, feat].shape))
        df[feat].replace({0: np.nan}, inplace=True)

    # BINARY CATEGORIES TO 0-1 NUMERICAL
    for feat in df.columns.values.tolist():
        if df[feat].nunique() == 2:
            df[feat] = df[feat].astype('category')
            df[feat].replace({df[feat].cat.categories.tolist()[0]: 0}, inplace=True)
            df[feat].replace({df[feat].cat.categories.tolist()[1]: 1}, inplace=True)
            if feat not in fnum:
                fnum.append(feat)

    # IMPUTER LONGITUDE and LATITUDE depending on REGION mean
    feats = ['longitude', 'latitude']
    ref = 'region'
    for feat in feats:
        print('\nSTANDARD DEVIATION when imputing {} using {}:\n{}'.format(feat, ref, df.groupby([ref])[feat].std()))
        cord = df.groupby([ref])[feat].mean().to_dict()
        df[feat] = df.apply(lambda row: cord[row[ref]] if np.isnan(row[feat]) else row[feat], axis=1)

    # IMPUTER GPS HEIGHT depending on LONGITUDE and LATITUDE AND THE REST OF FEATURES USING ALL NUMERICAL FEATURES
    feats = ['longitude', 'latitude', 'gps_height']
    if models_predefined is None:
        models = [StandardScaler(), KNNImputer(missing_values=np.nan, n_neighbors=5),
                  StandardScaler(), KNNImputer(missing_values=np.nan, n_neighbors=5)]
        df[feats] = models[0].fit_transform(df[feats])
        df[feats] = models[1].fit_transform(df[feats])
        df[feats] = models[0].inverse_transform(df[feats])
        df[fnum] = models[2].fit_transform(df[fnum])
        df[fnum] = models[3].fit_transform(df[fnum])
        df[fnum] = models[2].inverse_transform(df[fnum])
    else:
        models = None
        df[feats] = models_predefined[0].transform(df[feats])
        df[feats] = models_predefined[1].transform(df[feats])
        df[feats] = models_predefined[0].inverse_transform(df[feats])
        df[fnum] = models_predefined[2].transform(df[fnum])
        df[fnum] = models_predefined[3].transform(df[fnum])
        df[fnum] = models_predefined[2].inverse_transform(df[fnum])

    # NUMERICAL FEATURES TRANSFORMATIONS
    fnum_copy = fnum.copy()
    min_value = 1.
    if num_transformations is None:
        transf = []
    for i, feat in enumerate(fnum_copy):
        offset = min([np.min(df[feat]), min_value])
        df[feat + '_log'] = np.log(df[feat] - offset + min_value)
        df[feat + '_reciprocal'] = np.reciprocal(df[feat] - offset + min_value)
        transformer = PowerTransformer(method="box-cox", standardize=False)
        df[feat + '_power'] = transformer.fit_transform((df[feat] - offset + min_value).to_frame())
        if transformer.lambdas_ > 2:
            df[feat + '_power'] = np.power(df[feat] - offset + min_value, 2)
        if num_transformations is None:
            num_trans = [feat, feat + '_log', feat + '_reciprocal', feat + '_power']
            v_cramer, _ = f.v_cramer_function(df[num_trans], df['Target'])
            best_index = np.argmax(v_cramer.values)
            transf.append(best_index)
        else:
            transf = None
            best_index = num_transformations[i]
        if best_index == 0:
            df.drop(feat + '_log', axis=1, inplace=True)
            df.drop(feat + '_reciprocal', axis=1, inplace=True)
            df.drop(feat + '_power', axis=1, inplace=True)
        elif best_index == 1:
            df.drop(feat, axis=1, inplace=True)
            df.drop(feat + '_reciprocal', axis=1, inplace=True)
            df.drop(feat + '_power', axis=1, inplace=True)
            fnum.remove(feat)
            fnum.append(feat + '_log')
        elif best_index == 2:
            df.drop(feat, axis=1, inplace=True)
            df.drop(feat + '_log', axis=1, inplace=True)
            df.drop(feat + '_power', axis=1, inplace=True)
            fnum.remove(feat)
            fnum.append(feat + '_reciprocal')
        elif best_index == 3:
            df.drop(feat, axis=1, inplace=True)
            df.drop(feat + '_reciprocal', axis=1, inplace=True)
            df.drop(feat + '_log', axis=1, inplace=True)
            fnum.remove(feat)
            fnum.append(feat + '_power')

    # DISCRETIZATION FOR NUMERICAL FEATURES
    fbin = [x for x in fnum if df[x].nunique() > 2]
    if binning_tree is None:
        binning_tree = DecisionTreeDiscretiser(random_state=seed, cv=5, scoring='neg_mean_squared_error',
                                               regression=False, param_grid={'max_depth': [1, 2, 3],
                                                                             'min_samples_leaf': [10, 20, 50]})
        binning_tree.fit(df[fbin], df['Target'])
    df_binned = binning_tree.transform(df[fbin])
    fnum_bin = [x + '_binned' for x in fbin]
    df_binned.columns = fnum_bin
    df = pd.concat([df, df_binned], axis=1)

    df[fnum] = df[fnum].astype('float64')

    print('\n')
    return df, fnum, binning_tree, models, transf


def cat_feat_engineering(df, fnum, min_counts=1, cats_predefined=None, feat_out=None, imp_dict=None):
    print('\n')
    fcat = [i for i in df.columns.values.tolist() if i not in fnum]
    df[fcat] = df[fcat].astype('category')

    if cats_predefined is None:
        plot_name = 'Barplot categories count - training set'
        fcat.remove('Target')
    else:
        plot_name = 'Barplot categories count - submission set'

    # RECORDED_BY FEATURE
    feat = 'recorded_by'
    print('\nValue counts for feature {}: \n{}'.format(feat, df[feat].value_counts()))
    df.drop(feat, axis=1, inplace=True)
    fcat.remove(feat)

    # QUANTITY FEATURE
    feat1 = 'quantity'
    feat2 = 'quantity_group'
    print('\nValue counts for feature {}: \n{}'.format(feat1, df[feat1].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat2, df[feat2].value_counts()))
    df.drop(feat2, axis=1, inplace=True)
    fcat.remove(feat2)

    # PAYMENT FEATURE
    feat1 = 'payment'
    feat2 = 'payment_type'
    df[feat1].replace({'pay annually': 'annually', 'pay per bucket': 'per bucket',
                       'pay when scheme fails': 'on failure', 'pay monthly': 'monthly'}, inplace=True)
    print('\nValue counts for feature {}: \n{}'.format(feat1, df[feat1].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat2, df[feat2].value_counts()))
    df.drop(feat2, axis=1, inplace=True)
    fcat.remove(feat2)

    # WATER FEATURE
    feat1 = 'water_quality'
    feat2 = 'quality_group'
    df[feat1].replace({'soft': 'good'}, inplace=True)
    print('\nValue counts for feature {}: \n{}'.format(feat1, df[feat1].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat2, df[feat2].value_counts()))
    df.drop(feat2, axis=1, inplace=True)
    fcat.remove(feat2)

    # WATERPOINT FEATURE
    feat1 = 'waterpoint_type'
    feat2 = 'waterpoint_type_group'
    print('\nValue counts for feature {}: \n{}'.format(feat1, df[feat1].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat2, df[feat2].value_counts()))
    df.drop(feat2, axis=1, inplace=True)
    fcat.remove(feat2)

    # SOURCE FEATURE
    feat1 = 'source'
    feat2 = 'source_type'
    feat3 = 'source_class'
    print('\nValue counts for feature {}: \n{}'.format(feat1, df[feat1].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat2, df[feat2].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat3, df[feat3].value_counts()))
    df.drop([feat2, feat3], axis=1, inplace=True)
    fcat.remove(feat2)
    fcat.remove(feat3)

    # EXTRACTION FEATURE
    feat1 = 'extraction_type'
    feat2 = 'extraction_type_group'
    feat3 = 'extraction_type_class'
    print('\nValue counts for feature {}: \n{}'.format(feat1, df[feat1].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat2, df[feat2].value_counts()))
    print('Value counts for feature {}: \n{}'.format(feat3, df[feat3].value_counts()))
    df.drop([feat2, feat3], axis=1, inplace=True)
    fcat.remove(feat2)
    fcat.remove(feat3)

    df['funder'].replace({'0': np.nan}, inplace=True)
    df['installer'].replace({('0', '-'): np.nan}, inplace=True)
    df['wpt_name'].replace({('24', '21'): np.nan}, inplace=True)
    df['region_code'].replace({40: 60}, inplace=True)
    for feat in fcat:
        df[feat].replace({('Unknown', 'unknown'): np.nan}, inplace=True)
    feats = ['funder', 'installer', 'wpt_name', 'scheme_name']
    for feat in feats:
        df[feat] = df[feat].apply(lambda row: row.replace('_', '').replace(' ', '').replace('/', ''))
        df[feat] = df[feat].apply(lambda row: '_' + str(row)[:3].lower() + '_' if row is not np.nan else row)

    if feat_out is not None:
        df.drop(feat_out, axis=1, inplace=True)
        fcat = [x for x in fcat if x not in feat_out]

    # Reduce number of categories by grouping the categories with low occurrence
    df, fcat, cats, fout = f.reduce_categories(df, fcat, min_counts, cat_predefined=cats_predefined)
    f.value_counts_barplot(df, fcat, name=plot_name)

    # IMPUTER WPT NAME per MODE MOST FREQUENT
    df['wpt_name'].fillna(pd.Series.mode(df['wpt_name']).tolist()[0], inplace=True)

    # IMPUTER SUBVILLAGE per WARD
    df['subvillage'].fillna(df['ward'], inplace=True)

    # IMPUTER CATEGORICAL FEATURES with most common category per BASIN (most meaningful feature with no missing values)
    print(df['region_code'].value_counts())
    print(df['district_code'].value_counts())
    print(df['region'].value_counts())
    print(df['basin'].value_counts())
    ref = 'basin'
    feats = ['funder', 'installer', 'scheme_name', 'scheme_management', 'management', 'management_group', 'payment',
             'water_quality', 'quantity', 'source']
    if imp_dict is None:
        imp_dict = df.groupby(ref)[feats].agg(pd.Series.mode).to_dict()
    for feat in feats:
        df[feat] = df.apply(lambda row: imp_dict[feat][row[ref]] if row[feat] is np.nan else row[feat], axis=1)

    return df, fcat, fout, cats, imp_dict
