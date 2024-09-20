#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy as np
import datetime
from scipy import stats
from scipy.stats import qmc

# Configuration
anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'
figsize=(9, 3)

def load_series(file_name, data_folder):
    # Load the input data
    data_path = f'{data_folder}/data/{file_name}'
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    # Load the labels
    label_path = f'{data_folder}/labels/combined_labels.json'
    with open(label_path) as fp:
        labels = pd.Series(json.load(fp)[file_name])
    labels = pd.to_datetime(labels)
    # Load the windows
    window_path = f'{data_folder}/labels/combined_windows.json'
    window_cols = ['begin', 'end']
    with open(window_path) as fp:
        windows = pd.DataFrame(columns=window_cols,
                data=json.load(fp)[file_name])
    windows['begin'] = pd.to_datetime(windows['begin'])
    windows['end'] = pd.to_datetime(windows['end'])
    # Return data
    return data, labels, windows


def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    threshold=None,
                    figsize=figsize):
    # Open a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot data
    plt.plot(data.index, data.values, zorder=0)
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)

    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3)
    # Plot threshold
    if threshold is not None:
        plt.plot([data.index[0], data.index[-1]], [threshold, threshold], linestyle=':', color='tab:red')
    plt.grid()
    plt.tight_layout()


def plot_autocorrelation(data, max_lag=100, figsize=figsize):
    # Open a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Autocorrelation plot
    pd.plotting.autocorrelation_plot(data['value'])
    # Customized x limits
    plt.xlim(0, max_lag)
    # Rotated x ticks
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()


def plot_histogram(data, bins=10, vmin=None, vmax=None, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist(data, density=True, bins=bins)
    # Update limits
    lims = plt.xlim()
    if vmin is not None:
        lims = (vmin, lims[1])
    if vmax is not None:
        lims = (lims[0], vmax)
    plt.xlim(lims)
    plt.grid()
    plt.tight_layout()


def plot_histogram2d(xdata, ydata, bins=10, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist2d(xdata, ydata, density=True, bins=bins)
    plt.tight_layout()


def plot_density_estimator_1D(estimator, xr, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot the estimated density
    xvals = xr.reshape((-1, 1))
    dvals = np.exp(estimator.score_samples(xvals))
    plt.plot(xvals, dvals)
    plt.grid()
    plt.tight_layout()


def plot_density_estimator_2D(estimator, xr, yr, figsize=figsize):
    # Plot the estimated density
    nx = len(xr)
    ny = len(yr)
    xc = np.repeat(xr, ny)
    yc = np.tile(yr, nx)
    data = np.vstack((xc, yc)).T
    dvals = np.exp(estimator.score_samples(data))
    dvals = dvals.reshape((nx, ny))
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    plt.pcolor(dvals.T)
    plt.tight_layout()
    plt.xticks(ticks=np.linspace(0, len(xr), 5),
            labels=[f'{v:.2f}' for v in np.linspace(xr[0], xr[-1], 5)])
    plt.yticks(ticks=np.linspace(0, len(yr), 5),
            labels=[f'{v:.2f}' for v in np.linspace(yr[0], yr[-1], 5)])


def get_pred(signal, thr):
    return pd.Series(signal.index[signal >= thr])


def get_metrics(pred, labels, windows):
    tp = [] # True positives
    fp = [] # False positives
    fn = [] # False negatives
    advance = [] # Time advance, for true positives
    # Loop over all windows
    used_pred = set()
    for idx, w in windows.iterrows():
        # Search for the earliest prediction
        pmin = None
        for p in pred:
            if p >= w['begin'] and p < w['end']:
                used_pred.add(p)
                if pmin is None or p < pmin:
                    pmin = p
        # Compute true pos. (incl. advance) and false neg.
        l = labels[idx]
        if pmin is None:
            fn.append(l)
        else:
            tp.append(l)
            advance.append(l-pmin)
    # Compute false positives
    for p in pred:
        if p not in used_pred:
            fp.append(p)
    # Return all metrics as pandas series
    return pd.Series(tp, dtype='datetime64[ns]'), \
            pd.Series(fp, dtype='datetime64[ns]'), \
            pd.Series(fn, dtype='datetime64[ns]'), \
            pd.Series(advance)


class ADSimpleCostModel:

    def __init__(self, c_alrm, c_missed, c_late):
        self.c_alrm = c_alrm
        self.c_missed = c_missed
        self.c_late = c_late

    def cost(self, signal, labels, windows, thr):
        # Obtain predictions
        pred = get_pred(signal, thr)
        # Obtain metrics
        tp, fp, fn, adv = get_metrics(pred, labels, windows)
        # Compute the cost
        adv_det = [a for a in adv if a.total_seconds() <= 0]
        cost = self.c_alrm * len(fp) + \
           self.c_missed * len(fn) + \
           self.c_late * (len(adv_det))
        return cost


def opt_thr(signal, labels, windows, cmodel, thr_range):
    costs = [cmodel.cost(signal, labels, windows, thr)
            for thr in thr_range]
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return thr_range[best_idx], costs[best_idx]




class KDEDetector:

    def __init__(self, bandwidth=0.1, thr=0.0):
        self.est = KernelDensity(kernel='gaussian',
                bandwidth=bandwidth)
        self.thr = thr

    def fit_estimator(self, X):
        kde2.fit(X)

    def fit_threshold(self, cmodel, tr):
        pass


def sliding_window_1D(data, wlen):
    assert(len(data.columns) == 1)
    # Get shifted columns
    m = len(data)
    lc = [data.iloc[i:m-wlen+i+1].values for i in range(0, wlen)]
    # Stack
    wdata = np.hstack(lc)
    # Wrap
    wdata = pd.DataFrame(index=data.index[wlen-1:],
            data=wdata, columns=range(wlen))
    return wdata


def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=figsize, s=4):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(labels))
    plt.tight_layout()


def plot_signal(signal, labels=None,
        figsize=figsize, s=4):
    plt.figure(figsize=figsize)
    plt.plot(signal.index, signal, label='signal')
    if labels is not None:
        nonzero = signal.index[labels != 0]
        smin, smax = np.min(signal),  np.max(signal)
        lvl = smin - 0.05 * (smax-smin)
        plt.scatter(nonzero, np.ones(len(nonzero)) * lvl,
                s=s, color='tab:orange')
    plt.grid()
    plt.tight_layout()



class GMMDistribution:
    def __init__(self, mu, sigma, weights):
        assert(len(mu) == len(sigma))
        assert(len(mu) == len(weights))
        self.mu = mu
        self.sigma = sigma
        self.weights = weights
        # Build the individual Gaussian distributions
        self.dist = [stats.multivariate_normal(mean=m, cov=s)
                for m, s in zip(mu, sigma)]

    def sample(self, size, seed=None):
        # Reseed the RNG, if needed
        if seed is not None:
            np.random.seed(seed)
        # Sample the components
        n_components = len(self.weights)
        comp = np.random.choice(range(n_components), size=size, p=self.weights)
        # Sample each component from the corrent Gaussian distribution
        res = np.array([self.dist[k].rvs() for k in comp])
        # Return both results
        return res, comp

    def pdf(self, x):
        # tmp = [w * d.pdf(x) for w, d in zip(self.weights, self.dist)]
        tmp = np.array([w * d.pdf(x) for w, d in zip(self.weights, self.dist)])
        return sum(tmp)

    def score_samples(self, x):
        return np.log(self.pdf(x))


def generate_gmm_dist(n_components=1, seed=42):
    # Seed the RNG
    np.random.seed(seed)
    # Define the weight of each component
    weights = np.random.random(size=n_components)
    alpha = 0.3
    weights = alpha + (1 - alpha) * weights # linear flattening
    weights /= weights.sum()
    # Define a mean covariance matrix per component
    all_mu = []
    all_sigma = []
    # Generate the means
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    all_mu = sampler.random(n=n_components)
    # Generate the covariance matrixes
    for k in range(n_components):
        # Generate a random symmetric matrix
        tmp = np.random.random(size=(2, 2))
        sigma = alpha * 0.1 * np.eye(n_components) + (1 - alpha) * (tmp @ tmp.T)
        all_sigma.append(sigma)
    # Build the distribution object
    res = GMMDistribution(mu=all_mu, sigma=all_sigma, weights=weights)
    return res


class HPCMetrics:
    def __init__(self, c_alarm, c_missed, tolerance):
        self.c_alarm = c_alarm
        self.c_missed = c_missed
        self.tolerance = tolerance

    def cost(self, signal, labels, thr):
        # Obtain errors
        fp, fn = get_errors(signal, labels, thr, self.tolerance)

        # Compute the cost
        return self.c_alarm * len(fp) + self.c_missed * len(fn)


def opt_threshold(signal, labels, th_range, cmodel):
    costs = [cmodel.cost(signal, labels, th) for th in th_range]
    best_th = th_range[np.argmin(costs)]
    best_cost = np.min(costs)
    return best_th, best_cost


def get_errors(signal, labels, thr, tolerance=1):
    pred = signal[signal > thr].index
    anomalies = labels[labels != 0].index

    fp = set(pred)
    fn = set(anomalies)
    for lag in range(-tolerance, tolerance+1):
        fp = fp - set(anomalies+lag)
        fn = fn - set(pred+lag)
    return fp, fn


def plot_training_history(history, 
        figsize=figsize):
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label='loss')
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'], label='val. loss')
        plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_bars(data, figsize=figsize, tick_gap=1):
    plt.figure(figsize=figsize)
    x = 0.5 + np.arange(len(data))
    plt.bar(x, data, width=0.7)
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=45)
    plt.grid()
    plt.tight_layout()
