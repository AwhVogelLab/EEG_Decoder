import os
import warnings
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sista
import seaborn as sns
import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter
from mne.parallel import parallel_func
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('always', 'Warning', RuntimeWarning) # include runtime warnings


class RDMFileHandler:
    '''
    Class to handle RDM scripts. Used by the other modules in this package
    '''

    def __init__(self, file: str = 'all_rdms.hdf5'):
        self.file = file
        self._rdm_dict = defaultdict(lambda x: None)  # initialize dict
        self.loaded = False
        self.loaded_subs = []

    def check_exists(self):
        '''
        check if the file exists so we know whether to write or read to it
        '''

        if not os.path.isfile(self.file):
            print(f'File {self.file} does not exist')
            return False
        else:
            return True

    def load_data(self, force: bool = False):
        '''
        Loads data (either all subjects or just unloaded ones)

        Arguments:
        force: if true, reload all subjects
        '''
        if not self.check_exists():
            raise FileNotFoundError('File needs to be created first')
        with h5py.File(self.file, 'r') as f:
            if force:
                toload = self.subs  # load everything
            else:
                toload = np.setxor1d(
                    self.loaded_subs, self.subs)  # load unloaded
            for sub in toload:
                # load unloaded subjects
                self._rdm_dict[sub] = f[sub][()]
                self.loaded_subs.append(sub)
        self.loaded = True  # set this flag so

    def write_subject(self, data, sub, overwrite: bool = False):
        '''
        writes data to a subject, creating if it does not exist
        '''
        with h5py.File(self.file, 'a') as f:

            # if overwrite, overwrite the existing key
            if overwrite and sub in f.keys():
                self._rdm_dict[sub] = data
                del f[sub]
                f.create_dataset(sub, data=data)

            # if it doesn't exist, create it
            if sub not in f.keys():
                self._rdm_dict[sub] = data
                f.create_dataset(sub, data=data)
                self.loaded_subs.append(sub)

    @property
    def rdms(self):
        '''property to return all rdms in the subject'''
        if not self.loaded:
            self.load_data()
        return np.stack(list(self._rdm_dict.values()))

    @property
    def subs(self):
        ''' property to return the subjects existing in the file'''
        if self.loaded:
            return self.loaded_subs  # if everything is loaded return the loaded subjects
        if self.check_exists():
            with h5py.File(self.file, 'r') as f:
                # otherwise return subjects in the file
                return sorted(f.keys())
        else:
            return []  # return an empty list if no file


class Crossnobis:

    def __init__(self, exp, condition_dict: dict, t_win: int = 50, t_step: int = 25,
                 n_splits: int = 1000, n_jobs: int = -1, file: str = 'all_rdms.hdf5'):
        '''
        Class to calculate crossnobis distances between a list of conditions. Saves these to the given file for use
        by the RSA and MDS classes
        Arguments:
        exp: eeg_decoder.experiment object
        condition_dict: dict with condition key: code values
        t_win,t_step: window and step size for sliding window
        n_splits: how many splits to caluculate distances over
        n_jobs: how many cores to use for parallelization (default -1 - all)
        file: file to save final RDMs to 
        '''

        self.exp = exp
        self.nsub = exp.nsub
        self.times = exp.times
        self.n_jobs = n_jobs
        self.labels = list(condition_dict.keys())
        self.conditions = list(condition_dict.values())
        self.ridx, self.cidx = np.triu_indices(len(self.conditions), k=1)
        self.t_win = t_win
        self.n_splits = n_splits
        # Calculate the number of time steps per 20ms
        t_step_ms = int(t_step//(exp.times[1]-exp.times[0]))
        self.t = exp.times.astype(int)[t_step_ms:-t_step_ms:t_step_ms]

        self.f = RDMFileHandler(file=file)

    def _mean_by_condition(self, X, conds):
        '''
        computes the average of each condition in X, ordered by conds
        returns a n_conditions x n_channels array
        '''
        avs = np.zeros((len(np.unique(conds)), *X.shape[1:]))
        for cond in sorted(np.unique(conds)):
            X_cond = X[conds == cond]
            avs[cond] = X_cond.mean(axis=0)
        return avs

    def _means_and_prec(self, X, conds):
        '''
        Returns condition averages and demeaned inverse covariance
        Covariance is regularized by ledoit-wolf procedure
        '''
        cond_means = self._mean_by_condition(X, conds)
        cond_means_for_each_trial = cond_means[conds]
        X_demean = X - cond_means_for_each_trial  # demean

        return cond_means, LedoitWolf(assume_centered=True).fit(X_demean).precision_

    def _calc_rdm_crossnobis_single(self, meas1, meas2, noise):
        '''
        Calculates RDM using crossnobis distance using means from x and y, and covariance
        Largely taken from https://github.com/rsagroup/rsatoolbox/blob/main/src/rsatoolbox/rdm/calc.py#L429
        Updated to return the signed square root of the RDM because
        LDC is an estimator of the squared mahalonobis distance
        '''
        kernel = meas1 @ noise @ meas2.T
        rdm = np.expand_dims(np.diag(kernel), 0) + \
            np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
        return np.sign(rdm) * np.sqrt(np.abs(rdm))

    def _crossnobis_single(self, X_train, conds_train, X_test, conds_test):
        '''
        Uses condition means from both train and test, but only uses the training
        examples to compute the noise covariance/precision matrix. You may have another
        preference, but I did it this way to avoid train-test leakage. 
        '''
        means_train, noise_train = self._means_and_prec(X_train, conds_train)
        means_test = self._mean_by_condition(X_test, conds_test)
        rdm = self._calc_rdm_crossnobis_single(
            means_train, means_test, noise_train)
        return rdm

    def _crossnobis_train_test_across_time(self, Xdata, y, train, test, cond_order):
        # assumes Xdata is n_trials x n_features x n_times

        X_train, y_train = Xdata[train], y[train]
        X_test, y_test = Xdata[test], y[test]

        # calculate RDMS over time for this fold
        rdms = [self._crossnobis_single(
            X_train[..., t], y_train, X_test[..., t], y_test) for t in range(Xdata.shape[-1])]

        # concatenate over time and resort to the given cond_order
        return np.stack(rdms, axis=2)[np.ix_(cond_order, cond_order)]

    def crossnobis(self, Xdata, ydata, cond_order, test_size=0.5, n_splits=1000, n_jobs=-1):
        '''
        Wrapper for a parallel function to calculate a series of crossnobis distances
        n_splits and n_jobs should be given as arguments upon class initialization


        '''

        enc = LabelEncoder()  # converts condition labels to integer codes
        conds = enc.fit_transform(ydata)
        cond_order = enc.transform(cond_order)  # how to resort the final RDMs

        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

        parallel, p_func, _ = parallel_func(
            self._crossnobis_train_test_across_time, n_jobs)
        rdms = parallel(
            p_func(
                Xdata=Xdata,
                y=conds,
                train=train_idx,
                test=test_idx,
                cond_order=cond_order
            )
            for train_idx, test_idx in cv.split(Xdata, conds)
        )

        rdms = np.stack(rdms, axis=0)
        return rdms.mean(0)  # average over folds

    def calculate_rdms(self, overwrite: bool = True):
        '''
        Main wrapper function to calculate RDMs for each subject
        Keyword arguments:
        overwrite: whether or not to overwrite existing RDMs in the file (default: false)
        '''

        for isub in tqdm.tqdm(range(self.nsub)):
            # if it exists already then skip this subject
            if not overwrite and self.exp.subs[isub] in self.f.subs:
                continue
            xdata, sub_condition = self.exp.load_subject(isub)

            # Average the EEG data within the time window and store it in xdata_time_binned
            xdata_time_binned = np.zeros(
                (xdata.shape[0], xdata.shape[1], len(self.t)))
            for tidx, t in enumerate(self.t):
                timepoints = (self.exp.times >= t - self.t_win //
                              2) & (self.exp.times <= t + self.t_win//2)
                xdata_time_binned[:, :, tidx] = xdata[:,
                                                      :, timepoints].mean(-1)

            # Calculate the RDMs using the crossnobis function and store them in rdms for the current subject
            sub_rdm = self.crossnobis(xdata_time_binned, sub_condition,
                                      self.conditions, n_splits=self.n_splits, n_jobs=self.n_jobs)
            self.f.write_subject(
                sub_rdm, self.exp.subs[isub], overwrite=overwrite)


class RSA:
    def __init__(self, condition_labels, times, file: str = 'all_rdms.hdf5', delay_period_start=500, theoretical_models: dict = None):
        """
        Class to perform and visualize RSA analyses
        Keyword arguments:
        condition_labels: labels for each condition, in the order that they were calculated in
        times: list of each timepoint that RDMs were calculated at
        file: file where RDMs are stored
        delay_period start: beginning of delay period (for averaging)
        theoretical models: dict of RDMs per model

        """
        self.labels = condition_labels
        self.ridx, self.cidx = np.triu_indices(len(self.labels), k=1)
        self.theoretical_models = theoretical_models

        self.color_palette = {factor: sns.color_palette()[i] for i, factor in enumerate(
            list(self.theoretical_models.keys())+['Intercept'])}

        self.t = times

        self.delay_period_start = delay_period_start
        self.delay_period_end = max(self.t)

        self.rdms = RDMFileHandler(file=file).rdms
        self.nsub = self.rdms.shape[0]

    ##################################
    # CALCULATE FITS
    ##################################

    def fit_theoretical_models(self, models=None, ret_VIF=False):
        '''
        Applies a linear regression fit of specified theoretical models
        Arguments:
        models: list of models (found in self.theoretical_models) to run
        ret_VIF: returns a list of VIFs per condition
        '''
        if models is None:  # if unset use all available options
            models = self.theoretical_models.keys()

        self.r2 = np.full((self.nsub, len(self.t)), np.nan)

        self.factor_df = pd.DataFrame(np.transpose(
            [self.theoretical_models[key][self.ridx, self.cidx] for key in models]), columns=models)  # convert to 1D dataframe
        self.factor_df['Intercept'] = 1

        # rank factors by relative dissimilarity
        ranked_vals = sista.rankdata(self.factor_df, axis=0)

        if ret_VIF:  # calculate and return VIFs
            desmat_with_intercept = pd.DataFrame(ranked_vals)
            desmat_with_intercept['intercept'] = 1
            vif_data = pd.DataFrame()
            vif_data['regressor'] = desmat_with_intercept.columns.drop(
                'intercept')
            vif_data['VIF'] = [variance_inflation_factor(desmat_with_intercept.values, i)
                               for i in range(len(desmat_with_intercept.columns))
                               if desmat_with_intercept.columns[i] != 'intercept']
            self.vif_data = vif_data
            vif_data['regressor'] = self.factor_df.columns.tolist()
            print(vif_data)
        partial_r_df = pd.DataFrame()

        for isub in range(self.nsub):
            ranked_dists = sista.rankdata(
                self.rdms[isub, self.ridx, self.cidx, :], axis=0)
            # Rank the RDMs across each time point by row

            r_scores = defaultdict(lambda: np.zeros((ranked_dists.shape[1])))

            for t in range(ranked_dists.shape[1]):
                curr_dists = ranked_dists[:, t]

                fitted_lm = LinearRegression().fit(ranked_vals, curr_dists)
                full_r2 = fitted_lm.score(ranked_vals, curr_dists)
                self.r2[isub, t] = full_r2
                # Fit a linear regression model and calculate the R-squared for the full model

                # Calculate partial correlation for each factor
                # Skip the intercept column
                for col in range(ranked_vals.shape[1]-1):
                    submodel_r2 = LinearRegression().fit(np.delete(ranked_vals, col, axis=1),
                                                         curr_dists).score(np.delete(ranked_vals, col, axis=1), curr_dists)
                    # Fit a linear regression model without the current factor and calculate the R-squared
                    r_scores[col][t] = np.sqrt(
                        full_r2 - submodel_r2) * np.sign(fitted_lm.coef_[col])
                    # Calculate the partial correlation and store it in r_scores

                r_scores[ranked_vals.shape[1]][t] = np.sqrt(full_r2)
                # Store the total correlation (sqrt of R-squared) for the full model

            r_df = pd.DataFrame(r_scores)
            r_df.columns = self.factor_df.columns
            r_df['sid'] = isub
            r_df['timepoint'] = self.t

            sub_df = pd.melt(r_df, id_vars=['sid', 'timepoint'], value_vars=r_df.columns[:-2],
                             var_name='factor', value_name='semipartial correlation')

            # Append the correlation dataframe
            partial_r_df = pd.concat([partial_r_df, sub_df], axis=0)

        partial_r_df = partial_r_df.reset_index(drop=True)
        self.partial_r_df = partial_r_df[partial_r_df['factor'] != 'Intercept']

    def fit_theoretical_models_independently(self, models=None):
        '''
        this is essentially the same thing, but we only calculate the full r2 scores for each one
        Useful for testing exactly how good / bad models are
        '''
        if models is None:
            models = list(self.theoretical_models.keys())

        self.factor_df = pd.DataFrame(np.transpose(
            [self.theoretical_models[key][self.ridx, self.cidx] for key in models]), columns=models)
        self.factor_df['Intercept'] = 1

        ranked_vals = sista.rankdata(self.factor_df, axis=0)
        correlations_separate = np.full(
            (len(models), self.nsub, len(self.t)), np.nan)
        for ifac in range(len(models)):
            factor_rank = ranked_vals[:, ifac]
            for isub in range(self.nsub):
                ranked_dists = sista.rankdata(
                    self.rdms[isub, self.ridx, self.cidx, :], axis=0)
                # Rank the RDMs for each time point by row
                for t in range(ranked_dists.shape[1]):
                    curr_dists = ranked_dists[:, t]

                    correlations_separate[ifac, isub, t] = np.corrcoef(
                        factor_rank, curr_dists)[0, 1]
        # this is because seaborn likes dataframes, so also get a list of subjects, times, and factors
        # return correlations_separate
        nfac, nsub, ntime = correlations_separate.shape
        sub_reshape = np.moveaxis(np.broadcast_to(
            np.arange(0, nsub), (nfac, ntime, nsub)), 2, 1)  # subject list
        # time list reshaped to proper dimensions
        time_reshape = np.broadcast_to(self.t, (nfac, nsub, ntime))
        factor_reshape = np.moveaxis(np.broadcast_to(
            models, (nsub, ntime, nfac)), 2, 0)  # factor list
        self.correlation_df = pd.DataFrame(
            (factor_reshape.flat, sub_reshape.flat, time_reshape.flat, correlations_separate.flat)).T
        self.correlation_df.columns = [
            'factor', 'subject', 'timepoint', 'correlation']

    ##################################
    # VISUALIZATIONS
    ##################################

    def visualize_rdm(self, key: str = 'Empirical', title='Dataset RDM', ax=None):
        '''
        Plot a RDM
        Arguments:
        key: which RDM, one of any theoretical model or "Empirical" to plot the empirical RDM
        averaged over the delay period
        title: plot title
        ax: subplot axis. Useful for plotting multiple RDMs on 1 axis
        '''
        if ax is None:
            _, ax = plt.subplots()
        if key == 'Empirical':
            model = self.rdms[..., self.t >
                              self.delay_period_start].mean((0, -1))
        else:
            if key not in self.theoretical_models.keys():
                raise ValueError(
                    'Key must be one of "Empirical" or a valid theoretical model')
            model = self.theoretical_models[key]

        sns.heatmap(model, ax=ax, xticklabels=self.labels,
                    yticklabels=self.labels)  # plot the RDM
        ax.set_title(title)

    def plot_corrs(self, fac_order=None, y_sig=0.3, t_start=None, t_end=None, title='semipartial correlation of RDMs During Delay Period'):
        '''
        Plots a barplot of partial correlations for each factor, averaged over time
        Arguments:
        fac_order: list of factors to plot, in order (default all)
        y_sig: where to put stars for significance
        t_start, t_end: time range to use, default delay_period_start and end
        title: figure title

        '''

        if fac_order is None:
            fac_order = self.factor_df.columns.tolist()
            fac_order.remove('Intercept')

        # default to beginning and end of delay period
        t_start = self.delay_period_start if t_start is None else t_start
        t_end = self.delay_period_end if t_end is None else t_end

        # average partial correlations over selected time
        delay_summary_df = self.partial_r_df.query(
            f'timepoint > {t_start} & timepoint < {t_end}').groupby(['sid', 'factor']).mean().reset_index()
        delay_summary_df = delay_summary_df[~(
            delay_summary_df.factor == 'Total')]  # ignore total

        plt.figure(facecolor='white', figsize=(8, 4))  # set up figure
        plt.hlines(0, xmin=-.5, xmax=3.5, color='black',
                   linestyle='--')  # 0 line
        ax = sns.barplot(data=delay_summary_df, x='factor', y='semipartial correlation',
                         ci=68, palette=self.color_palette, order=fac_order)  # plot correlations

        # significance testing
        for i, factor in enumerate(fac_order):
            x = delay_summary_df.query(f'factor=="{factor}"')[
                'semipartial correlation'].values
            # wilcoxcon rank-signed test
            w, p = sista.wilcoxon(x=x, nan_policy='omit')
            if any(np.isnan(x)):
                warnings.warn(
                    'Warning: Partial correlations contain nans. Check your data', RuntimeWarning)
            # print out test statistics and factors
            print(factor, np.mean(x), w, p)

            plt.scatter(i, y_sig, alpha=0)  # dummy points to annotate

            # annotate spots with significance labels
            if p < .001:
                plt.annotate('***', (i, y_sig), size=20,
                             color=self.color_palette[factor], label='p < .001', horizontalalignment='center')
            elif p < .01:
                plt.annotate('**', (i, y_sig), size=20,
                             color=self.color_palette[factor], label='p < .01', horizontalalignment='center')
            elif p < .05:
                plt.annotate('*', (i, y_sig), size=20,
                             color=self.color_palette[factor], label='p < .05', horizontalalignment='center')

        ax.spines[['right', 'top']].set_visible(False)
        _ = plt.title(title, fontsize=20, pad=20)
        plt.tight_layout()

    def plot_independent_corrs(self, y_sig=0.4, t_start=None, t_end=None, title='Correlation of RDM to Each Factor During Delay Period'):
        '''
        Plots a barplot of absolute correlations for each factor, averaged over time
        Arguments:
        y_sig - where to put stars for significance
        t_start, t_end - time range to use, default delay_period_start and end
        title: figure title

        '''

        # default to beginning and end of delay period
        t_start = self.delay_period_start if t_start is None else t_start
        t_end = self.delay_period_end if t_end is None else t_end

        factors = self.correlation_df.factor.unique()
        # average across time
        delay_summary_df = self.correlation_df.query(f'timepoint > {t_start} & timepoint < {t_end}').groupby([
            'subject', 'factor']).mean().reset_index()

        plt.figure(facecolor='white', figsize=(8, 4))
        plt.hlines(0, xmin=-.5, xmax=len(factors)+.5,
                   color='black', linestyle='--')  # 0 line
        ax = sns.barplot(data=delay_summary_df, x='factor',
                         y='correlation', order=factors)

        # test for significance and plot stars
        for i, factor in enumerate(factors):

            x = delay_summary_df.query(f'factor=="{factor}"')[
                'correlation'].values
            w, p = sista.wilcoxon(x=x)
            print(factor, np.mean(x), w, p)
            plt.scatter(i, y_sig, alpha=0)

            if p < .001:
                plt.annotate('***', (i, y_sig), size=20,
                             color=self.color_palette[factor], label='p < .001', horizontalalignment='center')
            elif p < .01:
                plt.annotate('**', (i, y_sig), size=20,
                             color=self.color_palette[factor], label='p < .001', horizontalalignment='center')
            elif p < .05:
                plt.annotate('*', (i, y_sig), size=20,
                             color=self.color_palette[factor], label='p < .001', horizontalalignment='center')

        ax.spines[['right', 'top']].set_visible(False)
        _ = plt.title(title, fontsize=20, pad=20)
        plt.tight_layout()

    def plot_corrs_temporal(self, title='Model Fits across time', stim_time=[0, 500], hide_stim=False, ax=None, factors: list[str] = None, ylim=[None, None]):
        '''
        Plots correlations of empirical RDM to each factor over timepoints
        Arguments:
        title: plot title
        stim_time: to plot a gray bar over these times
        hide_stim: do not show the stimulus gray bar
        ax: axis to use (default creates a new one)
        factors: iterable of factor names to plot (default: all)
        ylim: figure y axes

        '''

        if ax is None:
            ax = plt.subplot()

        if factors is None:
            factors = self.partial_r_df.factor.unique()
            factors = factors[factors != 'Intercept']

        if ylim[0] is not None and ylim[1] is not None:  # set ylim
            ax.set_ylim(ylim)

        ax = sns.lineplot(x='timepoint', y='semipartial correlation', hue='factor', data=self.partial_r_df[np.in1d(
            self.partial_r_df.factor, factors)], palette=self.color_palette)  # plot relevant factors

        sig_y = -0.2  # where to start significance boxes

        ax.hlines(0, xmin=self.t[0], xmax=self.t[-1],
                  color='black', linestyle='--')  # 0 bar

        # significance testing using wilcoxcon test for each timepoint and condition
        for factor in factors:
            tmp_df = self.partial_r_df.query(f'factor=="{factor}"')
            p_values = []
            for t in tmp_df.timepoint.unique():
                x = tmp_df[tmp_df['timepoint'] ==
                           t]['semipartial correlation'].values
                _, p = sista.wilcoxon(x=x, nan_policy='omit')
                p_values.append(p)
            # correct for n_timepoints comparisons
            _, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')

            sig05 = corrected_p < 0.05

            ax.scatter(self.t[sig05], np.ones(sum(sig05))*(sig_y),
                       marker='s', s=10, color=self.color_palette[factor])  # mark significant points on axis
            sig_y -= 0.05
            ax.get_legend().set_title(None)  # remove legend title because it gets in the way
        plt.title(title)

        # gray stim bar ofver stim period
        if not hide_stim:
            y_min, y_max = ax.get_ylim()

            ax.fill_between(stim_time, [y_min, y_min], [y_max, y_max],
                            color='gray', alpha=.5, zorder=0)

        return ax  # return the axis for further modification

    def correlate_regressors(self, x_factor: str, y_factor: str, title: str = None, xlab=None, ylab=None, ax=None):
        '''
        Function to plot correlations of two factors.
        Useful for seeing if they explain similar sources of variance
        Arguments:
        x_factor, y_factor: factors on each axis
        title: plot tiel
        xlab,ylab: axis labels (default: factor names)
        '''
        if ax is None:
            fig, ax = plt.subplots()

        delay_summary_df = self.partial_r_df.query(
            f'timepoint > {self.delay_period_start}').groupby(['sid', 'factor']).mean().reset_index()

        x_corr = delay_summary_df.query(f'factor == "{x_factor}"')[
            'semipartial correlation']  # pick out correlations
        y_corr = delay_summary_df.query(f'factor == "{y_factor}"')[
            'semipartial correlation']

        # scatterplot and linear regression
        ax = sns.regplot(x=x_corr, y=y_corr, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(xlab if xlab is not None else x_factor)
        ax.set_ylabel(ylab if ylab is not None else y_factor)

        # calculate linear regression and plot r2 and p values
        lm = sista.linregress(x_corr, y_corr)
        plt.text(0.99, 0.95, f'r2 = {np.round(lm.rvalue**2,3)}',
                 horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes)

        p_text = f'p = {lm.pvalue:.2E}' if lm.pvalue < 0.001 else f'p = {round(lm.pvalue,3)}'
        plt.text(0.99, 0.9, p_text,
                 horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes)


class MDS:
    def __init__(self, times, labels, file: str = 'all_rdms.hdf5', n_components=2, stress_thresh=0.1, stress_behavior: str = 'warn'):
        """
        Class to calculate MDS projections from a RDM and visualize them
        inputs:
        times: list of timepoints each RDM is calculated at
        labels: condition labels
        file: filename where RDMS are stored
        n_components: how many MDS dimensions to include. Should probably always stay at 2
        stress_thresh: at what threshold is is the stress function problematic
        stress_behavior: "warn" or "raise" - whether to raise a warning or error if stress exceeds this
        """
        from sklearn.manifold import MDS as sklearn_MDS  # should be instanced earlier but here because I renamed it for conveience

        self.t = times
        self.rdms = RDMFileHandler(file=file).rdms
        self.mds = sklearn_MDS(dissimilarity='precomputed', random_state=0,
                               n_components=n_components, normalized_stress=False)  # instance transformer
        self.labels = labels
        self.stress_thresh = stress_thresh
        self.stress_behavior = stress_behavior
        self.stress_log = []  # log of stress values

    def check_stress(self):
        '''
        Helper function that checks if the projection stress is above our threshold
        '''
        if self.mds.stress_ > self.stress_thresh:
            if self.stress_behavior == 'warn':
                warnings.warn(
                    f'Warning: stress for MDS projection {self.mds.stress_} is above threshold {self.stress_thresh}', RuntimeWarning)
            elif self.stress_behavior == 'raise':
                raise RuntimeError(
                    f'Stress for MDS projection {self.mds.stress_} is above threshold {self.stress_thresh}')

    def calculate_MDS(self, t_start=500, t_stop=1500):
        """
        Helper function to calculate MDS projections in a certain range.
        Arguments:
        t_start,t_stop: define window to average over

        """
        tsub_rdm = self.rdms[..., np.logical_and(self.t >= t_start, self.t <= t_stop)].mean(
            (0, 3))  # average over subjects and times
        transform = self.mds.fit_transform(tsub_rdm)  # apply MDS scaling
        self.check_stress()
        self.stress_log.append(self.mds.stress_)  # helpful for debugging

        if transform.shape[-1] == 2:  # if 2D return both dimensions separately
            return transform[:, 0], transform[:, 1]
        elif transform.shape[-1] == 3:  # if 3D return x,y,z
            return transform[:, 0], transform[:, 1], transform[:, 2]
        else:  # otherwise return a tuple
            return transform

    def plot_MDS(self, ax=None, t_start=500, t_stop=1800, title=None, xlim=None, ylim=None, hide_axes: bool = True, circwidth: int = 300):
        """
        Displays MDS projection, and labels each condition
        Arguments:
        ax: axis to plot on
        t_start,t_stop: times to average over (passed to calculate_MDS)
        title: plot title
        xlim,ylim: axis limits. always specifiy manually if you want to compare multiple graphs
        hide_axes: bool, should axes be shown?
        circwidth: width of circles. Change this if the circles at each point overlap your condition labels
        """
        if ax is None:
            _, ax = plt.subplots()
        x, y = self.calculate_MDS(t_start, t_stop)
        ax.scatter(x, y, facecolors='none', edgecolors='black',
                   s=circwidth)  # draws circles centered at points
        for i, label in enumerate(self.labels):
            # labels points with condition labels
            ax.annotate(label, (x[i], y[i]), ha='center', va='center')

        ax.set_title(title)

        if hide_axes:
            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False)  # no axis labels or ticks

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    def _animation_wrapper(self, itime):
        '''
        helper function for the animator
        do not manually call this
        '''
        self.ani_ax.clear()
        try:  # plot projection averaged across [itime,itime+1]
            self.plot_MDS(ax=self.ani_ax, t_start=self.ani_times[itime], t_stop=self.ani_times[itime+1],
                          title=f'{self.ani_times[itime]}<t<{self.ani_times[itime+1]}', xlim=self.ani_xlim, ylim=self.ani_ylim)

        except ValueError as e:
            raise RuntimeError(
                f'i={itime},tstart={self.ani_times[itime]},tstop={self.ani_times[itime]}') from e

    def animate_MDS(self, t_start, t_stop, t_step, filename='./animation.gif', fps=1, xlim=(-0.005, 0.005), ylim=(-0.005, 0.005)):
        """
        Animates a MDS projection over time as a gif

        Arguments:
        t_start,t_stop: absolute minimum and maximum times
        t_step: interval between steps (reasonable is usually 50-250 ms)
        filename: filename to save as
        fps: adjust this to control speed
        xlim,ylim: axis limits

        """
        fig, self.ani_ax = plt.subplots()
        self.ani_xlim = xlim
        self.ani_ylim = ylim

        # set up times to iterate over
        self.ani_times = np.arange(t_start, t_stop+t_step, t_step)

        ani = FuncAnimation(fig, self._animation_wrapper, frames=len(self.ani_times)-2,
                            interval=500, repeat=False)  # instance matplotlib animator

        ani.save(filename, dpi=300,
                 writer=PillowWriter(fps=fps))  # output to file
        plt.close()
        print(f'Saved as {filename}')
