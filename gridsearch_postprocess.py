import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import itertools


class GridSearchPostProcess:

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.subplot_row = 2
        self.models = []
        self.dicts = []
        self.estimator = 'estimator'
        self.pre = 'preprocess'

    def param_sweep_matrix(self, params, test_score):
        """Postprocess the cross validation grid search results and plot the parameter sweep"""
        self.decode_gridsearch_results(params, test_score)
        for m, model in enumerate(self.models):
            feat_sweep, feat_no_sweep = self.unique_params_sweep(m)
            if len(feat_sweep) == 0:  # No sweep parameters
                test_matrix = np.array([self.dicts[m]['test']])
                self.plot_params_sweep(model, test_matrix, feat_no_sweep)
            elif len(feat_sweep) == 1:  # 1 sweep parameter
                keys = list(feat_sweep.keys())
                values = list(feat_sweep.values())
                test_matrix = np.array([self.dicts[m]['test']])
                self.plot_params_sweep(model, test_matrix, feat_no_sweep,
                                       xtick=values[0], xtag=keys[0])
            elif len(feat_sweep) == 2:  # 2 sweep parameters
                keys = list(feat_sweep.keys())
                values = list(feat_sweep.values())
                test_matrix = np.zeros([len(values[1]), len(values[0])])
                for j in range(len(values[1])):
                    for h in range(len(values[0])):
                        for z in range(len(self.dicts[m]['test'])):
                            if self.dicts[m][keys[0]][z] == values[0][h] and self.dicts[m][keys[1]][z] == values[1][j]:
                                test_matrix[j, h] = self.dicts[m]['test'][z]
                self.plot_params_sweep(model, test_matrix, feat_no_sweep,
                                       xtick=values[0], xtag=keys[0],
                                       ytick=values[1], ytag=keys[1])
            elif len(feat_sweep) >= 3:  # 3 or more sweep parameter
                keys = list(feat_sweep.keys())
                values = list(feat_sweep.values())
                test_matrix, zdict, i1, i2 = self.matrix3d_calculation(m, keys, values)
                self.plot_params_sweep(model, test_matrix, feat_no_sweep,
                                       xtick=values[i1], xtag=keys[i1],
                                       ytick=values[i2], ytag=keys[i2],
                                       zdict=zdict)

    def decode_gridsearch_results(self, params, test_score):
        """Assess the output grid search params to identify the sweep parametrization used per each model. It returns:
        - Models: list of the models used in the gridsearch
        - Dicts: list of dicts (one per each model) with the sweep parametrization"""
        for i in range(len(params)):
            string = str(params[i][self.estimator])
            model = string[:string.index('(')]
            if model not in self.models:
                self.models.append(model)
                self.dicts.append({})
                index = -1
            else:
                index = self.models.index(model)
            if 'test' in list(self.dicts[index].keys()):
                self.dicts[index]['test'] += [test_score[i]]
            else:
                self.dicts[index]['test'] = [test_score[i]]
            for key, value in params[i].items():
                if key == self.estimator:
                    continue
                elif '__' in key:
                    key = key.replace(key[:key.index('__') + 2], '')
                elif key == self.pre and value is not None:
                    value = str(value)[87:str(value).index('()')]
                if key in list(self.dicts[index].keys()):
                    self.dicts[index][key] += [value]
                else:
                    self.dicts[index][key] = [value]

    def unique_params_sweep(self, index):
        """Generate one dict for non sweep parameters and other for sweep parameters only capturing unique values"""
        feat_sweep = {}
        feat_no_sweep = {}
        for key, value in self.dicts[index].items():
            if key == 'test':
                continue
            if value.count(value[0]) == len(value):
                feat_no_sweep[key] = value[0]
            else:
                feat_sweep[key] = []
                [feat_sweep[key].append(val) for val in value if val not in feat_sweep[key]]
        return feat_sweep, feat_no_sweep

    def matrix3d_calculation(self, index, keys, values):
        """Calculate the 3D test matrix to plot consisting of a group of 2D matrix. The parameters with longer
        sweep parametrization are kept in a 2D matrix and depth is added to sweep the rest of parameters"""
        ind_values = np.argsort(np.array([len(values[i]) for i in range(len(values))]))
        i1 = ind_values[-1]
        i2 = ind_values[-2]
        extra_dims = [list(range(len(values[x]))) for x in range(len(values)) if x not in [i1, i2]]
        combs = list(itertools.product(*extra_dims))
        depth = len(combs)
        test_matrix = np.zeros([len(values[i2]), len(values[i1]), depth])
        zdict = []
        for p in range(depth):
            zdict.append({})
            ind = 0
            for i in range(len(values)):
                if i != i1 and i != i2:
                    zdict[p][keys[i]] = values[i][combs[p][ind]]
                    ind += 1
            for j in range(len(values[i2])):
                for h in range(len(values[i1])):
                    for z in range(len(self.dicts[index]['test'])):
                        if self.dicts[index][keys[i1]][z] == values[i1][h] \
                                and self.dicts[index][keys[i2]][z] == values[i2][j]:
                            ind = 0
                            for i in range(len(values)):
                                if i != i1 and i != i2:
                                    if self.dicts[index][keys[i]][z] == values[i][combs[p][ind]]:
                                        ind += 1
                                    else:
                                        break
                            else:
                                test_matrix[j, h, p] = self.dicts[index]['test'][z]
        return test_matrix, zdict, i1, i2

    def plot_params_sweep(self, algorithm, test_values, fixed_params,
                          xtick=None, ytick=None, xtag='', ytag='', zdict=None):
        """Plot parameter sweep for the cross validation grid search"""
        if xtick is None:
            xtick = ''
        if ytick is None:
            ytick = ''
        if zdict is None:
            fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
            plt.pcolormesh(test_values, cmap=plt.cm.PuBuGn)
            plt.colorbar()
            ax.set_xlabel('Parameter sweep ' + xtag.upper(), fontsize=14)
            ax.set_ylabel('Parameter sweep ' + ytag.upper(), fontsize=14)
            ax.set_title('Mean cross validation test score sweep with ' + algorithm.upper() +
                         '\n' + str(fixed_params), fontsize=24)
            ax.set_xticks(np.arange(0.5, len(xtick) + 0.5), labels=xtick, fontsize=14)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
            ax.set_yticks(np.arange(0.5, len(ytick) + 0.5), labels=ytick, fontsize=14)
            ax.text(0.5, 0.5, str(round(test_values[0, 0], 4)),
                    ha="center", va="center", color="k", fontweight='bold', fontsize=12)
            for h in range(len(xtick)):
                ax.text(h + 0.5, 0.5, str(round(test_values[0, h], 4)),
                        ha="center", va="center", color="k", fontweight='bold', fontsize=12)
                for j in range(len(ytick)):
                    ax.text(h + 0.5, j + 0.5, str(round(test_values[j, h], 4)),
                            ha="center", va="center", color="k", fontweight='bold', fontsize=12)
        else:
            fig, axes = plt.subplots(math.ceil(len(zdict) / self.subplot_row), self.subplot_row,
                                     figsize=(self.fig_width, self.fig_height))
            spare_axes = self.subplot_row - len(zdict) % self.subplot_row
            if spare_axes == self.subplot_row:
                spare_axes = 0
            for axis in range(self.subplot_row - 1, self.subplot_row - 1 - spare_axes, -1):
                if (math.ceil(len(zdict) / self.subplot_row) - 1) == 0:
                    fig.delaxes(axes[axis])
                else:
                    fig.delaxes(axes[math.ceil(len(zdict) / self.subplot_row) - 1, axis])
            ax = axes.ravel()
            for p in range(len(zdict)):
                pcm = ax[p].pcolormesh(test_values[:, :, p], cmap=plt.cm.PuBuGn)
                divider = make_axes_locatable(ax[p])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(pcm, cax=cax, orientation='vertical')
                ax[p].set_xlabel(xtag.upper(), fontsize=16)
                ax[p].set_ylabel(ytag.upper(), fontsize=16)
                ax[p].set_title('Parameter ' + str(zdict[p]), fontsize=18)
                ax[p].set_xticks(np.arange(0.5, len(xtick) + 0.5), labels=xtick, fontsize=14)
                plt.setp(ax[p].get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
                ax[p].set_yticks(np.arange(0.5, len(ytick) + 0.5), labels=ytick, fontsize=14)
                for h in range(len(xtick)):
                    for j in range(len(ytick)):
                        ax[p].text(h + 0.5, j + 0.5, str(round(test_values[j, h, p], 4)),
                                   ha="center", va="center", color="k", fontweight='bold', fontsize=12)
            fig.suptitle('Mean cross validation test score parameter sweep with ' + algorithm.upper() +
                         '\n' + str(fixed_params), fontsize=24)
            plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig('Parameter sweep ' + algorithm.upper() + ' algorithm.png', bbox_inches='tight')
        plt.close()
