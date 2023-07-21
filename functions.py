import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from itertools import combinations
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import math
import sys
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.encoding import OrdinalEncoder
from feature_engine.encoding import MeanEncoder
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures


def woe_calculation(df, feat, target, target_good, target_bad, print_data=False):
    """ Intermediate function to calculate WoE and IV information from good and bad distribution
    :param df: input dataframe
    :param feat: feature to calculate WoE and IV information
    :param target: series with the target information
    :param target_good: value corresponding to good class
    :param target_bad: value corresponding to bad class
    :param print_data: if printing or not the data
    :return:
    """
    df[feat] = df[feat].astype('category')
    lst = []
    for j in range(df[feat].nunique()):
        val = list(df[feat].unique())[j]
        lst.append([val, df[feat][(df[feat] == val) & (target == target_bad)].count(),
                    df[feat][(df[feat] == val) & (target == target_good)].count()])
    data = pd.DataFrame(lst, columns=['Value', 'Bad', 'Good'])
    data['Distribution Good'] = data['Good'] / data['Good'].sum()
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['WoE'].replace(np.nan, 0, inplace=True)
    data['WoE'].replace(-np.inf, min(data['WoE'].loc[data['WoE'] != -np.inf]), inplace=True)
    data['WoE'].replace(np.inf, max(data['WoE'].loc[data['WoE'] != np.inf]), inplace=True)
    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
    if print_data:
        print('\nInformation value dataframe feature {}: \n{}'.format(feat, data))
    return data


def information_value(df, feats, target, target_good=0, target_bad=1, print_data=True):
    """ Information value is calculated in the feats of the input dataframe. Target must be provided as a series
    :param df: input dataframe
    :param feats: feats in input dataframe to calculate the IV
    :param target: series with the target information
    :param target_good: value corresponding to good class
    :param target_bad: value corresponding to bad class
    :param print_data: if printing or not the data
    :return:
    """
    vi = np.zeros([len(feats)])
    for i, feat in enumerate(feats):
        data = woe_calculation(df, feat, target, target_good, target_bad)
        vi[i] = data['IV'].sum()
    ivalue = pd.DataFrame(vi, index=feats, columns=['Information Value'])
    if print_data:
        print('\nInformation Value: \n{}'.format(ivalue))
    return ivalue


def categorical_encoder(encoder, feat_cat, x_train, x_test, y_train=None, remove_cats=True):
    """ Categorical encoder to transform the feat_cat categorical features into new numerical features accordingly
    to the selected encoder algorithm.
    :param encoder: string referring to the type of encoder to apply
    :param feat_cat: categorical features to apply the encoder
    :param x_train: dataframe to fit the encoder method
    :param x_test: dataframe to apply the fitted train encoder
    :param remove_cats: it defines if the original categorical features are removed. If only one encoder is applied,
    remove_cats must be True, but if different encoder algorithms are applied in a row, remove_cats must be False
    :param y_train: series with the target class (only needed in some particular encoder types)
    :return:
    """
    if encoder == 'woe':
        combs = list(combinations(range(y_train.nunique()), 2))
        for i, comb in enumerate(combs):
            x1 = pd.DataFrame()
            x2 = pd.DataFrame()
            for feat in feat_cat:
                woe = woe_calculation(x_train, feat, y_train, target_good=comb[0], target_bad=comb[1])
                woe.set_index('Value', inplace=True)
                x1[feat] = x_train[feat].map(woe['WoE'])
                x2[feat] = x_test[feat].map(woe['WoE'])
            x1.columns = [x + '_' + encoder + '_' + str(i) for x in x1.columns]
            x2.columns = x1.columns
            x_train = pd.concat([x_train, x1], axis=1)
            x_test = pd.concat([x_test, x2], axis=1)
    else:
        if encoder == 'binary':
            enc = BinaryEncoder(cols=feat_cat, return_df=True)
            x1 = enc.fit_transform(x_train[feat_cat])
            x2 = enc.transform(x_test[feat_cat])
        elif encoder == 'mean_target':
            enc = MeanEncoder(variables=feat_cat)
            y_train_object = y_train.astype('object')
            x1 = enc.fit_transform(x_train[feat_cat], y_train_object)
            x2 = enc.transform(x_test[feat_cat])
        elif encoder == 'ordinal_target':
            enc = OrdinalEncoder(encoding_method='ordered', variables=feat_cat)
            y_train_object = y_train.astype('object')
            x1 = enc.fit_transform(x_train[feat_cat], y_train_object)
            x2 = enc.transform(x_test[feat_cat])
        elif encoder == 'ordinal':
            enc = OrdinalEncoder(encoding_method='arbitrary', variables=feat_cat)
            x1 = enc.fit_transform(x_train[feat_cat])
            x2 = enc.transform(x_test[feat_cat])
        elif encoder == 'count':
            x1 = pd.DataFrame()
            x2 = pd.DataFrame()
            for feat in feat_cat:
                counts = x_train[feat].value_counts().to_dict()
                x1[feat] = x_train[feat].map(counts)
                x2[feat] = x_test[feat].map(counts)
        elif encoder == 'onehot':
            onehot = OneHotEncoder(drop='if binary', sparse_output=False, handle_unknown='ignore')
            enc = ColumnTransformer(transformers=[('onehot', onehot, feat_cat)], remainder='passthrough')
            x_train[feat_cat] = x_train[feat_cat].astype('string')
            x_test[feat_cat] = x_test[feat_cat].astype('string')
            x1 = enc.fit_transform(x_train[feat_cat])
            oh_list = enc.get_feature_names_out()
            oh_list = [x.replace('onehot__', '').replace('remainder__', '') for x in oh_list]
            x1 = pd.DataFrame(x1, columns=oh_list)
            x2 = enc.transform(x_test[feat_cat])
            x2 = pd.DataFrame(x2, columns=oh_list)
        else:
            sys.exit('ERROR: NO CATEGORICAL ENCODER WAS SELECTED')

        x1.columns = [x + '_' + encoder for x in x1.columns]
        x2.columns = x1.columns
        x_train = pd.concat([x_train, x1], axis=1)
        x_test = pd.concat([x_test, x2], axis=1)
    if remove_cats:
        x_train.drop(feat_cat, axis=1, inplace=True)
        x_test.drop(feat_cat, axis=1, inplace=True)
        x_train = x_train.astype('float64')
        x_test = x_test.astype('float64')
    return x_train, x_test


def reduce_categories(df, fcat, min_counts=1, cat_predefined=None):
    """ Reduce the number of categories by categorizing as "Rare" variables with less than min_counts occurrences. If
    a feature has all samples as "Rare" after reduction is removed.
    :param df: input dataframe
    :param fcat: list of categorical features to apply the reduction
    :param min_counts: minimum counts of a category to be considered
    :param cat_predefined: predefined categories when wanting to repeat the same process in a different dataframe
    :return:
    """
    fout = []
    categories = []
    min_counts = max([min_counts, 1])
    for i, feat in enumerate(fcat):
        if feat.find('Target') == -1:
            print('\nFeat {} initial unique values: {}'.format(feat, df[feat].nunique()))
            if cat_predefined is None:
                cats = []
                valcounts = dict(df[feat].value_counts())
                for key, value in valcounts.items():
                    if value >= min_counts:
                        cats.append(str(key))
                df[feat] = df[feat].apply(lambda row: row if (str(row) in cats) or (row is np.nan) else 'Rare')
                if df[feat].nunique() == 1:
                    print('Feat {} discarded - no categories after min occurrence analysis'.format(feat))
                    fout.append(feat)
                    df.drop(feat, axis=1, inplace=True)
                else:
                    print('Feat {} categories after min occurrence analysis: {}'.format(feat, df[feat].nunique()))
                    categories.append(cats)
            else:
                cats = cat_predefined[i]
                df[feat] = df[feat].apply(lambda row: row if (str(row) in cats) or (row is np.nan) else 'Rare')
    fcat = [x for x in fcat if x not in fout]
    return df, fcat, categories, fout


def manage_outliers(df, feat):
    """ Limit the samples of the feature between 3 std dev or q1/3 -/+ 1.5 IQR
    :param df: input dataframe
    :param feat: feature to be updated having managed the outliers
    :return:
    """
    q3 = df[feat].quantile(0.75)
    q1 = df[feat].quantile(0.25)
    std_dev = df[feat].std()
    lower = min(q1 - 1.5 * (q3 - q1), np.mean(df[feat]) - 3 * std_dev)
    higher = max(q3 + 1.5 * (q3 - q1), np.mean(df[feat]) + 3 * std_dev)
    df[feat] = df[feat].apply(lambda row: min(max(row, lower), higher))
    return df


def remove_low_significant_feats(x_train, x_test, y_train, min_th=0, vcramer=True, corr=True):
    """ Remove low significant feats according to and/or vcramer and correlation. Low significant are considered
    when the statistics are below the min_th input.
    :param x_train: training data to apply fit_transform
    :param x_test: testing data to apply transform
    :param y_train: training data target information for fit process
    :param min_th: minimum statistic value for which the feature is considered significant
    :param vcramer: True when v-cramer statistic is considered
    :param corr: True when correlation coefficient is considered
    :return:
    """
    if vcramer:
        v_cramer, drop_feats1 = v_cramer_function(x_train, y_train, min_th)
    else:
        drop_feats1 = []
    if corr:
        df_corr, drop_feats2 = correlation_analysis(x_train, y_train, min_th)
    else:
        drop_feats2 = []
    more_drop = [x for x in drop_feats2 if x not in drop_feats1]
    drop_feats = drop_feats1 + more_drop
    x_train.drop(drop_feats, axis=1, inplace=True)
    x_test.drop(drop_feats, axis=1, inplace=True)
    return x_train, x_test


def new_tree_feats(x_train, x_test, y_train, combs, tree_model, min_increase=0.):
    """Generate new features based on decision tree model predictions and probabilistic predictions. The new features
    are based on combinations of 2 or 3 baseline features and are included in the dataframe only if the new feature
    increases th V-Cramer statistics more than min_increase when compared to baseline features
    :param x_train: training set to apply fit_transform
    :param x_test: testing set to apply transform
    :param y_train: training data target information for fit process
    :param combs: list of baseline features combinations
    :param tree_model: decision tree model to generate the predictions
    :param min_increase: minimum increase in V-Cramer statistic compared to baseline features to consider the new feats
    :return:
    """
    ind_fixed = round(len(combs) / 100)
    ind_moving = ind_fixed
    for i, comb in enumerate(combs):
        feats = [comb[x] for x in range(len(comb))]
        cubic_feats = np.mod(len(feats), 3)
        # New features based on decision tree
        tree_model.fit(x_train[feats], y_train)
        new_feats = ['treefeat_' + str(i)]
        for c in tree_model.classes_:
            new_feats.append(new_feats[0] + '_' + str(c))
        x_train[new_feats[0]] = tree_model.predict(x_train[feats])
        x_test[new_feats[0]] = tree_model.predict(x_test[feats])
        x_train[new_feats[1:]] = tree_model.predict_proba(x_train[feats])
        x_test[new_feats[1:]] = tree_model.predict_proba(x_test[feats])
        # Only select new significant features improving the marginal baseline
        if cubic_feats == 0:
            quadratic_combs = list(combinations([0, 1, 2], 2))
            for c in quadratic_combs:
                f1 = [feats[c[0]], feats[c[1]]]
                f1_name = feats[c[0]] + ' ' + feats[c[1]]
                feats.append(f1_name)
                tree_model.fit(x_train[f1], y_train)
                x_train[f1_name] = tree_model.predict(x_train[f1])
        cram_feats = feats + new_feats
        v_cramer, _ = v_cramer_function(x_train[cram_feats], y_train, print_data=False)
        vcram_poly = v_cramer['V Cramer']
        if cubic_feats == 0:
            for c in quadratic_combs:
                f1_name = feats[c[0]] + ' ' + feats[c[1]]
                x_train.drop(f1_name, axis=1, inplace=True)
        min_vcramer = 0
        for feat in feats:
            min_vcramer = max([min_vcramer, vcram_poly[feat]])
        min_vcramer += min_increase
        for feat in new_feats:
            if vcram_poly[feat] <= min_vcramer or np.isnan(vcram_poly[feat]):
                x_train.drop(feat, axis=1, inplace=True)
                x_test.drop(feat, axis=1, inplace=True)
            else:
                print('New feat {} added'.format(feat))
        if i > ind_moving:
            ind_moving += ind_fixed
            print('\nNew feature creation: {:.2f} %\nCurrent features in dataframe: {}'.format(
                100 * (i + 1) / len(combs), x_train.shape[1]))
    return x_train, x_test


def boxplot(df_ini, ncolumns=5, show=0, figwidth=20, figheight=10, name='Boxplot'):
    """ Plot all features in the input dataframe independently in a boxplot
    :param df_ini: input dataframe
    :param ncolumns: each feature is individually plot, and ncolumns define the number of plots in a row
    :param show: if 1 it is showed and if 0 it is saved in a file
    :param figwidth: plot width
    :param figheight: plot height
    :param name: string name to save the png file when show=0
    :return:
    """
    df = df_ini.copy().dropna()
    if isinstance(df, pd.Series):
        df = df.to_frame()
        ncolumns = 2
    feats = df.columns.values.tolist()
    fig, axes = plt.subplots(math.ceil(len(feats) / ncolumns), ncolumns, figsize=(figwidth, figheight))
    spare_axes = ncolumns - len(feats) % ncolumns
    if spare_axes == ncolumns:
        spare_axes = 0
    for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
        if math.ceil(len(feats) / ncolumns) - 1 == 0:
            fig.delaxes(axes[axis])
        else:
            fig.delaxes(axes[math.ceil(len(feats) / ncolumns) - 1, axis])
    ax = axes.ravel()
    for i in range(len(feats)):
        ax[i].boxplot(df[feats[i]])
        ax[i].grid(visible=True)
        ax[i].tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        ax[i].tick_params(axis='y', labelsize=10)
        ax[i].set_xlabel(feats[i], fontsize=10, weight='bold')
        ax[i].set_ylabel('Feature magnitude', fontsize=10, weight='bold')
    fig.suptitle('Boxplot per each feature', fontsize=20, weight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def value_counts_barplot(df, feats, name='Barplot categories counts', ncolumns=5, show=0, figwidth=20, figheight=10):
    """ Create an occurrence plot individually for each feature in feats in the input dataframe indicating the number
    of occurrences for each category in the feature.
    :param df: initial dataframe
    :param feats: list of categorical features to apply the reduction
    :param name: string name to save the png file when show=0
    :param ncolumns: each feature is individually plot, and ncolumns define the number of plots in a row
    :param show: if 1 it is showed and if 0 it is saved in a file
    :param figwidth: plot width
    :param figheight: plot height
    :return:
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
        ncolumns = 2
    fig, axes = plt.subplots(math.ceil(len(feats) / ncolumns), ncolumns, figsize=(figwidth, figheight))
    spare_axes = ncolumns - len(feats) % ncolumns
    if spare_axes == ncolumns:
        spare_axes = 0
    for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
        if math.ceil(len(feats) / ncolumns) - 1 == 0:
            fig.delaxes(axes[axis])
        else:
            fig.delaxes(axes[math.ceil(len(feats) / ncolumns) - 1, axis])
    ax = axes.ravel()
    for i in range(len(feats)):
        df[feats[i]].value_counts().plot(kind='bar', ax=ax[i])
        ax[i].tick_params(axis='both', labelsize=6)
        ax[i].tick_params(axis='x', rotation=30)
        ax[i].set_xlabel('Categories', fontsize=7, weight='bold')
        ax[i].set_ylabel('Frequency', fontsize=7, weight='bold')
        ax[i].set_title(feats[i], fontsize=8, weight='bold')
    fig.suptitle('Barplot value counts for categorical features', fontsize=20, weight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def histogram(df_ini, norm=False, target=None, nbins=15, ncolumns=5, show=0, figwidth=20, figheight=10,
              name='Histogram'):
    """ Plot all features in the input dataframe in a histogram independently
    :param df_ini: input dataframe with numerical features
    :param norm: histogram is normalized when True
    :param target: series target
    :param nbins: histogram number of bins
    :param ncolumns: each feature is individually plot, and ncolumns define the number of plots in a row
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    df = df_ini.copy().dropna()
    nbins = max([nbins, 2])
    if isinstance(df, pd.Series):
        df = df.to_frame()
        ncolumns = 2
    feats = df.columns.values.tolist()
    fig, axes = plt.subplots(math.ceil(len(feats) / ncolumns), ncolumns, figsize=(figwidth, figheight))
    spare_axes = ncolumns - len(feats) % ncolumns
    if spare_axes == ncolumns:
        spare_axes = 0
    for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
        if math.ceil(len(feats) / ncolumns) - 1 == 0:
            fig.delaxes(axes[axis])
        else:
            fig.delaxes(axes[math.ceil(len(feats) / ncolumns) - 1, axis])
    ax = axes.ravel()
    for i in range(len(feats)):
        if target is None:
            ax[i].hist(df.loc[:, feats[i]], density=norm)
        else:
            cmap = cm.get_cmap('tab10')
            colors = cmap.colors
            labels = target.unique()
            bins_vector = list(np.linspace(np.min(df[feats[i]]) - 1E-3, np.max(df[feats[i]]) + 1E-3, nbins))
            for ind, lab in enumerate(labels):
                ax[i].hist(df.loc[target == lab, feats[i]],
                           bins=bins_vector, density=norm, color=colors[ind], alpha=0.5, label=lab)
            ax[i].legend()
        ax[i].grid(visible=True)
        ax[i].tick_params(axis='both', labelsize=10)
        ax[i].set_xlabel(feats[i], fontsize=10, weight='bold')
        if norm:
            ax[i].set_ylabel('Normalized frequency', fontsize=10, weight='bold')
        else:
            ax[i].set_ylabel('Frequency', fontsize=10, weight='bold')
    fig.suptitle('Histogram per each feature', fontsize=20, weight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def scatter(df_ini, target, ncolumns=5, show=0, figwidth=20, figheight=10, name='Scatter'):
    """ Plot all features in the input dataframe in a scatter plot independently
    :param df_ini: input dataframe with numerical features
    :param target: series target
    :param ncolumns: each feature is individually plot, and ncolumns define the number of plots in a row
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    df = df_ini.copy().dropna()
    if isinstance(df, pd.Series):
        df = df.to_frame()
        ncolumns = 2
    feats = df.columns.values.tolist()
    fig, axes = plt.subplots(math.ceil(len(feats) / ncolumns), ncolumns, figsize=(figwidth, figheight))
    spare_axes = ncolumns - len(feats) % ncolumns
    if spare_axes == ncolumns:
        spare_axes = 0
    for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
        if math.ceil(len(feats) / ncolumns) - 1 == 0:
            fig.delaxes(axes[axis])
        else:
            fig.delaxes(axes[math.ceil(len(feats) / ncolumns) - 1, axis])
    ax = axes.ravel()
    for i in range(len(feats)):
        ax[i].scatter(df.loc[:, feats[i]], target)
        ax[i].grid(visible=True)
        ax[i].tick_params(axis='both', labelsize=10)
        ax[i].set_xlabel(feats[i], fontsize=10, weight='bold')
        ax[i].set_ylabel(target.name, fontsize=10, weight='bold')
    fig.suptitle('Scatter per each feature vs target', fontsize=20, weight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def timeplot(df_ini, ncolumns=5, show=0, figwidth=20, figheight=10, name='Timeplot'):
    """ Plot all features in the input dataframe in a timeplot plot independently
    :param df_ini: input dataframe with numerical features
    :param ncolumns: each feature is individually plot, and ncolumns define the number of plots in a row
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    df = df_ini.copy().dropna()
    if isinstance(df, pd.Series):
        df = df.to_frame()
        ncolumns = 2
    feats = df.columns.values.tolist()
    fig, axes = plt.subplots(math.ceil(len(feats) / ncolumns), ncolumns, figsize=(figwidth, figheight))
    spare_axes = ncolumns - len(feats) % ncolumns
    if spare_axes == ncolumns:
        spare_axes = 0
    for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
        if math.ceil(len(feats) / ncolumns) - 1 == 0:
            fig.delaxes(axes[axis])
        else:
            fig.delaxes(axes[math.ceil(len(feats) / ncolumns) - 1, axis])
    ax = axes.ravel()
    for i in range(len(feats)):
        ax[i].plot(df.loc[:, feats[i]])
        ax[i].grid(visible=True)
        ax[i].tick_params(axis='both', labelsize=10)
        ax[i].set_xlabel('Timeline', fontsize=10, weight='bold')
        ax[i].set_ylabel(feats[i], fontsize=10, weight='bold')
    fig.suptitle('Timeplot per each feature', fontsize=20, weight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def create_preprocess(pre, index_num):
    """ Create preprocess in a pipeline gridsearch among normalization, standardization and polynomial features.
    Note the preprocess technique is only applied in the indexes in the index_num list using ColumnTransformer
    :param pre: input preprocessor string name (norm, std or poly)
    :param index_num: list of feature indexes to apply the preprocess technique. If -1, it applies to all features
    :return:
    """
    if 'norm' in pre.lower():
        if index_num == -1:
            preprocess = MinMaxScaler()
        else:
            preprocess = ColumnTransformer(transformers=[('scaling', MinMaxScaler(), index_num)],
                                           remainder='passthrough')
    elif 'std' in pre.lower():
        if index_num == -1:
            preprocess = StandardScaler()
        else:
            preprocess = ColumnTransformer(transformers=[('scaling', StandardScaler(), index_num)],
                                           remainder='passthrough')
    elif 'poly' in pre.lower():
        preprocess = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    else:
        preprocess = None
        print('WARNING: no preprocessor was selected\n')
    return preprocess


def create_model(algorithm):
    """ Create model in a pipeline gridsearch among different machine learning algorithms.
    :param algorithm: model string name (knn, logreg, linearsvc, svm, tree, random...)
    :return:
    """
    if 'knn' in algorithm.lower():
        model = KNeighborsClassifier()
    elif 'logistic' in algorithm.lower() or 'logreg' in algorithm.lower():
        model = LogisticRegression()
    elif 'linear svc' in algorithm.lower() or 'linearsvc' in algorithm.lower():
        model = LinearSVC()
    elif 'gaussian' in algorithm.lower():
        model = GaussianNB()
    elif 'bernoulli' in algorithm.lower():
        model = BernoulliNB()
    elif 'multinomial' in algorithm.lower():
        model = MultinomialNB()
    elif 'tree' in algorithm.lower():
        model = DecisionTreeClassifier()
    elif 'extra' in algorithm.lower():
        model = ExtraTreesClassifier()
    elif 'forest' in algorithm.lower() or 'random' in algorithm.lower():
        model = RandomForestClassifier()
    elif 'gradient' in algorithm.lower() or 'boosting' in algorithm.lower():
        model = GradientBoostingClassifier()
    elif 'xgb' in algorithm.lower():
        model = XGBClassifier()
    elif 'lightgbm' in algorithm.lower() or 'light' in algorithm.lower():
        model = LGBMClassifier()
    elif 'svm' in algorithm.lower():
        model = SVC()
    elif 'mlp' in algorithm.lower():
        model = MLPClassifier()
    else:
        print('\nERROR: Algorithm was NOT provided. Note the type must be a list.\n')
        model = None
    return model


def decode_gridsearch_params(params_ini, index_num=-1):
    """ Decode the string parameter grid input dictionary to generate the real sklearn models to accept fit/transform methods.
    It does not affect to hyperparameters, only to decode string estimators to real sklearn models. It assumes the grid
    parameter has a first step to "preprocess" and a second step to "estimator", so other steps are ignored.
    :param params_ini: the string parameter grid input dictionary
    :param index_num: list of feature indexes to apply the preprocess technique. If -1, it applies to all features
    :return:
    """
    params = params_ini.copy()
    for i in range(len(params)):
        model = []
        for j in range(len(params[i]['estimator'])):
            model.append(create_model(params[i]['estimator'][j]))
        params[i]['estimator'] = model
        preproc = []
        for j in range(len(params[i]['preprocess'])):
            preproc.append(create_preprocess(params[i]['preprocess'][j], index_num))
        params[i]['preprocess'] = preproc
    return params


def correlation_analysis(df, target, min_corr=0., method='pearson'):
    """ Calculates the correlation coefficient per each non-categorical feature in the input dataframe (categorical
    features are ignored)
    :param df: input dataframe
    :param target: series target
    :param min_corr: minimum acceptable correlation coefficient. The features with lower values will be grouped together
    in a list and returned (user might want to remove them)
    :param method: correlation coefficient method (default is person, but kendall and spearman are also accepted)
    :return:
    """
    correlation = []
    to_drop = []
    feats = []
    i = 0
    for feat in df.columns.values.tolist():
        if not df[feat].dtypes == 'category':
            df_corr = pd.DataFrame()
            df_corr[feat] = df[feat]
            df_corr['Target'] = target
            df_corr['Target'] = df_corr['Target'].astype('float64')
            c = df_corr.corr(method=method)
            correlation.append(c.values[0][1])
            feats.append(feat)
            if np.abs(correlation[i]) < min_corr:
                to_drop.append(feat)
            i += 1
    corr = pd.DataFrame(correlation, index=feats, columns=['Pearson Correlation'])
    print('\nPEARSON CORRELATION: \n{}'.format(corr))
    return corr, to_drop


def v_cramer_function(df_ini, target, v_cramer_min=0, print_data=True):
    """ Calculates the V-Cramer coefficient per each feature in the input dataframe. V-Cramer statistics requires
    categorical features, but the function calculates the statistic for both numerical and categorical features by
    applying a binning technique if the feature is non-categorical.
    :param df_ini: input dataframe
    :param target: series target
    :param v_cramer_min: minimum acceptable V-Cramer coefficient. The features with lower values will be grouped
    together in a list and returned (user might want to remove them)
    :param print_data: when True returns V-Cramer calculations
    :return:
    """
    df = df_ini.copy()
    feats = df.columns.values.tolist()
    cramer = np.zeros([len(feats)])
    list_to_drop = []
    for i, feat in enumerate(feats):
        if not df[feat].dtypes == 'category':
            df[feat] = pd.cut(df[feat], bins=5)
        if not target.dtypes == 'category':
            target = pd.cut(target, bins=5)
        data = pd.crosstab(df[feat], target).values
        cramer[i] = (stats.contingency.association(data, method='cramer'))
        if cramer[i] < v_cramer_min:
            list_to_drop.append(feat)
    vcramer = pd.DataFrame(cramer, index=df.columns.values.tolist(), columns=['V Cramer'])
    if print_data:
        print('\nV CRAMER: \n{}'.format(vcramer))
    return vcramer, list_to_drop


def plot_pca_breakdown(pca, list_features, show=0, figwidth=20, figheight=10, name='PCA breakdown'):
    """ Plot the weight of each initial feature in each principal component
    :param pca: PCA matrix already transformed
    :param list_features: list with the feature names
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    ncomps = pca.components_.shape[0]
    nfeats = pca.components_.shape[1]
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    plt.pcolormesh(pca.components_, cmap=plt.cm.cool)
    plt.colorbar()
    pca_yrange = [x + 0.5 for x in range(ncomps)]
    pca_xrange = [x + 0.5 for x in range(nfeats)]
    plt.xticks(pca_xrange, list_features, rotation=80, ha='center')
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', labelsize=10)
    str_ypca = pca.get_feature_names_out()
    plt.yticks(pca_yrange, str_ypca)
    plt.xlabel("Feature", weight='bold', fontsize=10)
    plt.ylabel("Principal components", weight='bold', fontsize=10)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    for h in range(nfeats):
        for j in range(ncomps):
            ax.text(h + 0.5, j + 0.5, str(round(pca.components_[j, h], 2)),
                    ha="center", va="center", color="k", fontweight='bold', fontsize=12)
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def plot_pca_scree(pca, show=0, figwidth=20, figheight=10, name='PCA scree plot'):
    """ Plot the PCA scree --> cumulative explained variance vs number of principal components
    :param pca: PCA matrix already transformed
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
    ax2 = ax1.twinx()
    label1 = ax1.plot(range(1, len(pca.components_) + 1), pca.explained_variance_ratio_,
                      'ro-', linewidth=2, label='Individual PCA variance')
    label2 = ax2.plot(range(1, len(pca.components_) + 1), np.cumsum(pca.explained_variance_ratio_),
                      'b^-', linewidth=2, label='Cumulative PCA variance')
    plt.title('PCA Scree Plot', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Principal Components', fontsize=10)
    ax1.set_ylabel('Proportion of Variance Explained', fontsize=10, color='r')
    ax2.set_ylabel('Cumulative Proportion of Variance Explained', fontsize=10, color='b')
    la = label1 + label2
    lb = [la[0].get_label(), la[1].get_label()]
    ax1.legend(la, lb, loc='center right', prop={'size': 10})
    ax1.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    ax1.grid(visible=True)
    ax2.grid(visible=True)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def pca_biplot_PC1_PC2(df_ini, show=0, figwidth=20, figheight=10, name='PCA biplot'):
    """ Plot biplot between first and second principal component for the input dataframe, which is scaled and
    transformed to PCA internally in the function
    :param df_ini: input dataframe
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    df = df_ini.copy()
    try:
        labels = df.columns.values.tolist()
    except (Exception,):
        labels = None
    scale = StandardScaler()
    df_std = scale.fit_transform(df)
    pca = PCA()
    pca_transformed = pca.fit_transform(df_std)
    # Select the transformed values to principal components only for the first and second principal components
    score = pca_transformed[:, 0:2]
    # Select the components (direction of maximum variance) only for the first and second principal components
    coeff = pca.components_[0:2, :]
    xs = score[:, 0]
    ys = score[:, 1]
    nfeats = coeff.shape[1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    for i in range(nfeats):
        plt.arrow(0, 0, coeff[0, i], coeff[1, i], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[0, i] * 1.15, coeff[1, i] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center',
                     fontweight='bold', fontsize=10)
        else:
            plt.text(coeff[0, i] * 1.15, coeff[1, i] * 1.15, labels[i], color='g', ha='center', va='center',
                     fontweight='bold', fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    plt.title('PCA Biplot', fontsize=20, fontweight='bold')
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC2', fontsize=10)
    plt.grid()
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def plot_first_second_pca(pca_transformed, target, val1=1, val0=0, show=0, figwidth=20, figheight=10,
                          name='PCA first vs second'):
    """ Plot first vs second principal component grouping by target class (printing differently when target is val1 and
    when target is val0, mainly considering binary classification exercises).
    :param pca_transformed: PCA matrix already transformed
    :param target: series target
    :param val1: value when the target class is positive
    :param val0: value when the target class is negative
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    pca_transformed = pd.DataFrame(pca_transformed)
    pca_output1 = pca_transformed.loc[target == val1, :]
    pca_output0 = pca_transformed.loc[target == val0, :]
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    plt.scatter(pca_output1.iloc[:, 0], pca_output1.iloc[:, 1], s=99, marker='^', c='red', label='output=' + str(val1))
    plt.scatter(pca_output0.iloc[:, 0], pca_output0.iloc[:, 1], s=99, marker='o', c='blue', label='output=' + str(val0))
    plt.title('Plot for first PCA component vs second PCA component', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    plt.xlabel('First PCA', fontsize=10)
    plt.ylabel('Second PCA', fontsize=10)
    plt.legend(prop={'size': 10})
    plt.grid()
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def plot_inertia_silhouette(inertia, calinski, silhouette, show=0, figwidth=20, figheight=10,
                            name='Clusters Inertia Scores'):
    """ Plot a cluster performance outcome by plotting the inertia, calinski index and silhouette
    score vs the number of considered clusters. The purpose is to help with selecting the optimal number of clusters.
    :param inertia: list of inertia per each number of clusters (it must start with only single cluster)
    :param calinski: list of calinski index per each number of clusters (it must start with two clusters)
    :param silhouette: list of silhouette score per each number of clusters (it must start with two clusters)
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    fig, axes = plt.subplots(1, 3, figsize=(figwidth, figheight))
    max_clusters = len(inertia)
    ax = axes.ravel()
    ax[0].plot(range(1, max_clusters + 1), inertia, color='red', marker='o', markersize=10, linewidth=2,
               label='inertia')
    ax[0].set_ylabel('Inertia', fontsize=10)
    ax[1].plot(range(2, max_clusters + 1), calinski, color='green', marker='s', markersize=10, linewidth=2,
               label='calinski index')
    ax[1].set_ylabel('Calinski index', fontsize=10)
    ax[2].plot(range(2, max_clusters + 1), silhouette, marker='^', markersize=10, color='blue', linewidth=2,
               label='silhouette score')
    ax[2].set_ylabel('Silhouette score', fontsize=10)
    for i in range(3):
        ax[i].set_xlabel('Number of clusters', fontsize=10)
        ax[i].grid(visible=True)
        ax[i].tick_params(axis='both', labelsize=10)
        ax[i].legend(loc='best', prop={'size': 10})
    fig.suptitle('Cluster sweep tuning', fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def plot_dendrogram(model, linkage='', show=0, figwidth=20, figheight=10):
    """ Plot dendrogram for the introduced input fit model. Note linkage was already applied in the input fit model, and
    so that is not in the function scope.
    :param model: input fit model
    :param linkage: string for type of linkage applied in the input model (only for display purposes)
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :return:
    """
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, truncate_mode='level', p=6)
    ax.grid(visible=True)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel('Datapoints', fontsize=10)
    ax.set_ylabel('Clusters distance', fontsize=10)
    plt.title('Dendrogram with linkage ' + str(linkage), fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig('Dendrogram ' + str(linkage) + '.png', bbox_inches='tight')
        plt.close()


def plot_cluster_features(df_ini, cluster_class, ncolumns=5, bar_width=0.25, show=0, figwidth=20, figheight=10,
                          name='Cluster features analysis'):
    """ Plot the mean value of each feature in the input dataframe for each cluster class. This plot helps to identify
    the relation grouped in each cluster class.
    :param df_ini: input dataframe
    :param cluster_class: list indicating the cluster class per each sample in the input dataframe
    :param ncolumns: each feature is individually plot, and ncolumns define the number of plots in a row
    :param bar_width: bar width in the plots
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :param name: string name to save the png file when show=0
    :return:
    """
    n_clusters = max(cluster_class) + 1
    df = df_ini.copy().dropna()
    if isinstance(df, pd.Series):
        df = df.to_frame()
        ncolumns = 2
    feats = df.columns.values.tolist()
    fig, axes = plt.subplots(math.ceil(len(feats) / ncolumns), ncolumns, figsize=(figwidth, figheight))
    spare_axes = ncolumns - len(feats) % ncolumns
    if spare_axes == ncolumns:
        spare_axes = 0
    for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
        if math.ceil(len(feats) / ncolumns) - 1 == 0:
            fig.delaxes(axes[axis])
        else:
            fig.delaxes(axes[math.ceil(len(feats) / ncolumns) - 1, axis])
    ax = axes.ravel()
    cmap = cm.get_cmap('tab10')
    colors = cmap.colors
    for i in range(len(feats)):
        for cluster in range(n_clusters):
            ax[i].bar(1 + cluster * bar_width, df.iloc[cluster_class == cluster, i].mean(),
                      color=colors[cluster % len(colors)], width=bar_width, edgecolor='black',
                      label='cluster' + str(cluster))
        ax[i].set_title(feats[i], fontsize=10, y=1.0, pad=-14, fontweight='bold')
        ax[i].grid(visible=True)
        ax[i].tick_params(axis='both', labelsize=10)
        ax[i].set_ylabel('Feature magnitude', fontsize=10)
    ax[0].legend()
    fig.suptitle('Cluster features analysis', fontsize=20, weight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig(name + '.png', bbox_inches='tight')
        plt.close()


def plot_2d_cluster(df, feat_x, feat_y, cluster_class, show=0, figwidth=20, figheight=10):
    """ Plot feature x vs feature y grouping the samples according to the cluster they belong
    :param df: input dataframe
    :param feat_x: string for the name of the feature to capture in axis x
    :param feat_y: string for the name of the feature to capture in axis y
    :param cluster_class: list indicating the cluster class per each sample in the input dataframe
    :param show: plot is showed when show=1, and it is saved to png file when show=0
    :param figwidth: figure width
    :param figheight: figure height
    :return:
    """
    n_clusters = max(cluster_class) + 1
    clust_count = np.zeros([n_clusters])
    for k in range(n_clusters):
        clust_count[k] = np.count_nonzero(cluster_class == k)
    clust_max = np.argsort(-clust_count)
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    cmap = cm.get_cmap('tab10')
    colors = cmap.colors
    for i, cluster in enumerate(clust_max):
        ax.scatter(df.loc[cluster_class == cluster, feat_x], df.loc[cluster_class == cluster, feat_y],
                   color=colors[cluster % len(colors)], edgecolor='black', label='cluster' + str(cluster), s=(i+1)*100)
    ax.grid(visible=True)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel(feat_x, fontsize=10, weight='bold')
    ax.set_ylabel(feat_y, fontsize=10, weight='bold')
    ax.legend(prop={'size': 10})
    fig.suptitle('Cluster scatter plot 2D ' + feat_x + ' vs ' + feat_y, fontsize=20, weight='bold')
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    if show == 1:
        plt.show()
    else:
        plt.savefig('Cluster ' + feat_x + ' vs ' + feat_y + '.png', bbox_inches='tight')
        plt.close()
