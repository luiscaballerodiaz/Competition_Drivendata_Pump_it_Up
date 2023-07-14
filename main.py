import pandas as pd
import numpy as np
import functions as f
import section1_datascrubbing as s1
import section2_featengineering as s2
import section3_modeling as s3
from gridsearch_postprocess import GridSearchPostProcess


# Pandas configuration settings
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_rows', None)  # Enable option to display all dataframe rows
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
pd.set_option('display.max_seq_items', None)  # Enable printing the whole sequence content

# Instantiate an object for GridSearchPostProcess to manage the grid search results
sweep = GridSearchPostProcess()

# GENERAL SETTINGS
test_size = 0.2
score = 'accuracy'
load_data = 0
# 0 --> generate cleaned dataframe, transform it and apply modeling
# 1 --> upload cleaned dataframe, transform it and apply modeling
# 2 --> upload transformed dataframes and apply modeling

# CLEANING SETTINGS
min_occurrence = 25      # Minimum number of occurrences to consider the category

# TRANSFORMATION SETTINGS
min_threshold = 0.1  # Minimum statistic for a marginal feature to be considered
encoder = ['woe', 'mean_target']    # onehot, binary, count, ordinal, woe, ordinal_target and  mean_target
min_increase = 0.05  # Minimum Vcramer increase to consider the new poly/decision tree feature
cubic = 0            # 0 --> only quadratic feats considered and 1 --> both quadratic and cubic feats considered
tree_depth = [3, 4, 5, 6, 7]

# MODELING SETTINGS
sim_model = 'light'
samples0 = 0         # Number of samples for class 0 (when 0 the class is not modified)
samples1 = 0         # Number of samples for class 1 (when 0 the class is not modified)
samples2 = 0         # Number of samples for class 2 (when 0 the class is not modified)
cv_repeat = 1        # Number of cross validation repetitions in the grid search
grid_sweep = 0       # 0 --> single model and 1 --> grid sweep
n_feats = 0          # 0 --> optimal feats disabled and >0 --> optimal feats enabled to the corresponding value
user_feats_en = 1    # 0 --> use all feats and 1 --> use user_feats (NOTE: ignored when n_feats>0)
user_feats = ['ward_woe_2', 'construction_year_reciprocal', 'treefeat_554_0', 'treefeat_77_2', 'scheme_name_woe_2', 'treefeat_1748_0', 'lga_woe_2', 'treefeat_95_0', 'treefeat_643_2', 'treefeat_1667_2', 'treefeat_1749_0', 'treefeat_99_2', 'treefeat_110_2', 'treefeat_93_2', 'treefeat_957_0', 'treefeat_911_0', 'treefeat_1749_2', 'treefeat_1641_2', 'treefeat_63_2', 'treefeat_556_0', 'treefeat_957_2', 'treefeat_94_0', 'treefeat_1748_2', 'treefeat_1089_0', 'treefeat_449_2', 'treefeat_90_0', 'treefeat_77_0', 'treefeat_1390_0', 'funder_woe_2', 'treefeat_95_2', 'treefeat_172_2', 'treefeat_1086_2', 'treefeat_911_2', 'treefeat_90_2', 'lga_woe_1', 'treefeat_554_2', 'treefeat_67_0', 'treefeat_85_0', 'treefeat_556_2', 'treefeat_1392_0', 'treefeat_1089_2', 'treefeat_67_2', 'treefeat_0_2', 'treefeat_85_2', 'treefeat_139_0', 'treefeat_172_0', 'treefeat_666_0', 'treefeat_937_0', 'treefeat_1642_0', 'treefeat_1390_2', 'treefeat_139_2', 'ward_woe_1', 'treefeat_31_2', 'treefeat_277_2', 'treefeat_96_0', 'treefeat_138_2', 'treefeat_15_0', 'treefeat_1642_2', 'treefeat_10_0', 'treefeat_1139_2', 'treefeat_75_0', 'treefeat_73_0', 'treefeat_25_2', 'treefeat_448_2', 'treefeat_94_2', 'treefeat_404_0', 'treefeat_158_0', 'treefeat_80_0', 'treefeat_158_2', 'treefeat_80_2', 'treefeat_96_2', 'treefeat_1085_2', 'treefeat_903_2', 'treefeat_76_0', 'source_woe_0', 'treefeat_942_0', 'treefeat_107_2', 'treefeat_367_0', 'treefeat_1319_0', 'treefeat_1244_2', 'treefeat_1099_0', 'treefeat_136_0', 'treefeat_717_0', 'treefeat_112_0', 'treefeat_155_0', 'treefeat_1299_0', 'treefeat_937_2', 'treefeat_65_0', 'treefeat_188_0', 'treefeat_1038_2', 'treefeat_896_0', 'treefeat_1700_2', 'treefeat_250_0', 'treefeat_666_2', 'treefeat_1298_2', 'treefeat_112_2', 'treefeat_986_0', 'treefeat_1691_0', 'treefeat_1536_2', 'treefeat_133_0']

if load_data == 0:
    # Read CSV files
    df_train = pd.read_csv('Train.csv')
    target = pd.read_csv('Train labels.csv')
    df_train['Target'] = target['status_group']
    df_train['Target'].replace({'functional': 2, 'functional needs repair': 1, 'non functional': 0}, inplace=True)
    df_sub = pd.read_csv('Test.csv')
    id_sub = df_sub['id']
    fnum_ini = ['id', 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'population',
                'construction_year']

    # CSV file overview
    s1.dataframe_overview(df_train)
    f.histogram(df_train[fnum_ini], target=df_train['Target'], name='Histogram original')
    f.boxplot(df_train[fnum_ini], name='Boxplot original')

    # Numerical features engineering
    df_train, feat_num, train_binning, train_models, train_transf = s1.num_feat_engineering(df_train, fnum_ini)
    df_sub, _, _, _, _ = s1.num_feat_engineering(df_sub, fnum_ini, binning_tree=train_binning,
                                                 models_predefined=train_models, num_transformations=train_transf)

    # Categorical features engineering
    df_train, feat_cat, train_fout, train_cats, train_imp_dict = s1.cat_feat_engineering(df_train, feat_num,
                                                                                         min_counts=min_occurrence)
    df_sub, _, _, _, _ = s1.cat_feat_engineering(df_sub, feat_num, feat_out=train_fout,
                                                 cats_predefined=train_cats, imp_dict=train_imp_dict)

    f.histogram(df_train[feat_num], target=df_train['Target'], name='Histogram after modifications')
    f.boxplot(df_train[feat_num], name='Boxplot after modifications')

    df_train.to_csv('df_train.csv', index=False)
    df_sub.to_csv('df_sub.csv', index=False)
    id_sub.to_csv('id_sub.csv', index=False)
    df_fcat = pd.Series(feat_cat)
    df_fcat.to_csv('df_fcat.csv', index=False)
    df_fnum = pd.Series(feat_num)
    df_fnum.to_csv('df_fnum.csv', index=False)

if load_data <= 1:
    df_train = pd.read_csv('df_train.csv')
    df_sub = pd.read_csv('df_sub.csv')
    id_sub = pd.read_csv('id_sub.csv')
    id_sub = id_sub.iloc[:, 0]
    feat_cat = pd.read_csv('df_fcat.csv')
    feat_cat = feat_cat.iloc[:, 0].tolist()
    feat_num = pd.read_csv('df_fnum.csv')
    feat_num = feat_num.iloc[:, 0].tolist()

    print(f'\nFINAL NUMBER NAN VALUES IN TRAIN DATAFRAME\n{df_train.isna().sum()}\n')
    print(f'\nFINAL NUMBER NAN VALUES IN SUBMISSION DATAFRAME\n{df_sub.isna().sum()}\n')

    # Transform dataframe to train and test sets
    x_train, x_test, y_train, y_test, x_all, y_all = s2.split_data(test_size, df_train, seed=7)

    # Apply the corresponding transformations
    x_train, x_test = s2.data_transform(x_train, x_test, y_train, feat_cat, feat_num, encoder, tree_depth, score,
                                        min_th=min_threshold, min_incr=min_increase, deg3=cubic)
    x_all, x_sub = s2.data_transform(x_all, df_sub, y_all, feat_cat, feat_num, encoder, tree_depth, score,
                                     min_th=min_threshold, min_incr=min_increase, deg3=cubic)

    id_sub.to_csv('id_sub.csv', index=False)
    x_train.to_csv('x_train.csv', index=False)
    x_sub.to_csv('x_sub.csv', index=False)
    x_test.to_csv('x_test.csv', index=False)
    x_all.to_csv('x_all.csv', index=False)
    y_all.to_csv('y_all.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

elif load_data <= 2:
    x_train = pd.read_csv('x_train.csv')
    x_sub = pd.read_csv('x_sub.csv')
    x_test = pd.read_csv('x_test.csv')
    x_all = pd.read_csv('x_all.csv')
    y_train = pd.read_csv('y_train.csv')
    y_train = y_train.iloc[:, 0]
    y_test = pd.read_csv('y_test.csv')
    y_test = y_test.iloc[:, 0]
    y_all = pd.read_csv('y_all.csv')
    y_all = y_all.iloc[:, 0]
    id_sub = pd.read_csv('id_sub.csv')
    id_sub = id_sub.iloc[:, 0]

x_train, y_train = s2.oversampling(x_train, y_train, samples0, samples1, samples2)
y_all = np.array(y_all)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define the parameters grid for each simulated model
gridsearch_params, estimator_model = s3.get_sim_params(sim_model, grid_sweep)

if n_feats > 0:
    x_train, x_test, feat_optimal = s3.select_features(x_train, x_test, y_train, estimator_model, n_feats, score)
    x_all, x_sub = s3.user_features(x_all, x_sub, feat_optimal)
elif user_feats_en == 1:
    x_train, x_test = s3.user_features(x_train, x_test, user_feats)
    x_all, x_sub = s3.user_features(x_all, x_sub, user_feats)

print(f'X ALL SHAPE: {x_all.shape}\nY ALL SHAPE: {y_all.shape}\n')
print(f'X SUB SHAPE: {x_sub.shape}\n')
print('ALL SELECTED FEATURES: {}\n'.format(x_all.columns.values.tolist()))
print(f'X TRAIN SHAPE: {x_train.shape}\nY TRAIN SHAPE: {y_train.shape}\n')
print(f'X TEST SHAPE: {x_test.shape}\nY TEST SHAPE: {y_test.shape}\n')
print('TRAIN SELECTED FEATURES: {}\n'.format(x_train.columns.values.tolist()))

# Apply grid search
pd_grid, best_train_model = s3.pipeline_gridsearch(
    x_train, x_test, y_train, y_test, gridsearch_params, score, cv_repeat)
sweep.param_sweep_matrix(params=pd_grid['params'], test_score=pd_grid['mean_test_score'])

# Submission generation with model referred to all data
s3.generate_submission(best_train_model, x_all, y_all, x_sub, id_sub)
