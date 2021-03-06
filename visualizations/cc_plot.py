"""This module provides plotting support for the objects and data structures defined in cross_calibration.py

Plots included are violin plots, box plots, scatter plots, and old, deprecated plots we used for analysis we don't do
anymore.
"""

from itertools import cycle

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mpl_scatter_density
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

import quadrangles as quad

__authors__ = ["Zach Werginz", "Andrés Muñoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

matplotlib.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)


def add_identity(axes, *line_args, **line_kwargs):
    """Adds the identity line to a plot (y=x).

    Args:
        axes (obj): matplotlib axes object for which to add the identity line to
        *line_args: additional line arguments
        **line_kwargs: additional line keywords
    """
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(ax):
        """Provides the mechanism for recalculating the identity line if the plot is resized.

        ax (obj): axis that is being resized

        """
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)


def power_law(x, a, b, c):
    """Define the power function."""
    return (a * (x ** b)) + c


def power_func(p, x, y):
    """Define the power function for use with scipy.optimize"""
    return (p[0] * (x ** p[1])) - y


def power_func_intercept(p, x, y):
    """Define a power function with an intercept for use with scipy.optimize"""
    return (p[0] * (x ** p[1])) + p[2] - y


def linear_law(x, a, b):
    """Define a linear function."""
    return a * x + b


def linear_func(p, x, y):
    """Define a linear function for use with scipy.optimize"""
    return (p * x) - y


def gaussian(x, A, sigma, mu):
    """Define a gaussian function."""
    return A / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def gaussian_func(p, x, y):
    """Define a gaussian function for use with scipy.optimize."""
    return p[0] / (np.sqrt(2 * np.pi) * p[1]) * np.exp(-(x - p[2]) ** 2 / (2. * p[1] ** 2)) - y


def fit_xy(x, y, fit_type='power'):
    """Fit a set of two variables using robust least squares and return coefficients.

    Args:
        x (array): independent variable
        y (array): dependent variable
        fit_type (str): choose from power, powerC, or linear for type of fit

    Returns:
        dict: dictionary of coefficients and their error terms

    """
    ind = (np.isfinite(x) & np.isfinite(y))
    if fit_type == 'power':
        popt = scipy.optimize.least_squares(power_func, [1, 1],
                                            args=(x[ind], y[ind]), bounds=([.0, 0], [np.inf, np.inf]),
                                            ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale='jac',
                                            jac='3-point', loss='soft_l1', f_scale=.1)
    elif fit_type == 'powerC':
        popt = scipy.optimize.least_squares(power_func_intercept, [1, .5, 1],
                                            args=(x[ind], y[ind]), max_nfev=10000,
                                            bounds=([0, 0, 0], [np.inf, np.inf,np.inf]),
                                            ftol=1e-3, xtol=1e-3, gtol=1e-3, x_scale='jac',
                                            jac='3-point', loss='soft_l1', f_scale=.5)
    else:
        popt = scipy.optimize.least_squares(linear_func, [1], args=(x[ind], y[ind]), loss='huber', f_scale=.1)

    # ---------------Acquire fit uncertainties-----------------------------------
    # taken from scipy fitting scheme
    _, s, vt = np.linalg.svd(popt.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(popt.jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    stderr = np.sqrt(np.diag(pcov))

    # --------------Return appropriate parameters--------------------------------
    if fit_type == 'power':
        return {'a': popt.x[0], 'aUnc': stderr[0], 
                'b': popt.x[1], 'bUnc': stderr[1],
                'n': len(x), 'res': np.sum(popt.fun/y[ind])}
    elif fit_type == 'powerC':
        return {'a': popt.x[0], 'aUnc': stderr[0],
                'b': popt.x[1], 'bUnc': stderr[1], 
                'c': popt.x[2], 'cUnc': stderr[2],
                'n': len(x), 'res': np.sum(popt.fun/y[ind])}
    else:
        return {'a': popt.x[0], 'aUnc': stderr[0], 'n': len(x), 'res': np.sum(popt.fun/y[ind])}


def plot_fits(a, b, c=5, fit_type='power', axes=None, **kwargs):
    """Plot with coefficients a, b, and c for either a line or power law.

    Coefficients follow

    Args:
        a (float):
        b (float):
        c (float):
        fit_type (str): defaults to power fit, choose from power, powerC, or linear
        axes (obj): the axis being resized
        **kwargs: additional keywords for matplotlib plotting
    """
    def callback(ax):
        """Provides the mechanism for recalculating the plots if they are resized.

        ax (obj): axis that is being resized

        """
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        new_x = np.linspace(low, high, 10000)
        if fit_type == 'power':
            ax.plot(new_x, power_law(new_x, a, b, 0), **kwargs)
            ax.plot(-new_x, -power_law(new_x, a, b, 0), **kwargs)
        elif fit_type == 'powerC':
            ax.plot(new_x, power_law(new_x, a, b, c), **kwargs)
        elif fit_type == 'linear':
            ax.plot(new_x, linear_law(new_x, a, b), **kwargs)
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)


def hist_axis(bl, disk_limits=None, **kwargs):
    """Returned parameters for multiple histograms."""
    b = kwargs.pop('binCount', 100)
    constant = kwargs.pop('const', 'width')
    axes_swap = kwargs.pop('axes_swap', False)
    if axes_swap:
        x, y, da = quad.extract_valid_points(bl)
    else:
        y, x, da = quad.extract_valid_points(bl)

    minimum = np.nanmin(x)
    maximum = np.nanmax(x)
    if disk_limits is not None:
        ind = np.where(np.logical_and(
            np.logical_and(np.isfinite(x), np.isfinite(y)), np.logical_and(da > disk_limits[0], da < disk_limits[1])))
        x = x[ind]
        y = y[ind]
    else:
        ind = np.where(np.logical_and(np.isfinite(x), np.isfinite(y)))
        x = x[ind]
        y = y[ind]
    hist_list = []

    # Split by constant bin width
    if constant == 'width':
        xedges = np.linspace(minimum, maximum, b)
        for i in range(1, xedges.shape[0] - 1):
            inds = np.where(
                np.logical_and(x < xedges[i], x >= xedges[i-1]))
            if len(inds[0]) < 20:
                continue
            slice_med = (xedges[i] + xedges[i-1]) / 2
            hist_list.append(get_hist_info(y[inds], xedges, slice_med))
    # Split by constant bin count
    else: 
        sort_ind = np.argsort(x)
        y = y[sort_ind]
        s = 0
        e = 100
        for i in range(0, len(y), b):
            slice_med = np.median(x[s:e])
            hist_list.append(get_hist_info(y[s:e], 10, slice_med))
            s += b
            e += b

    return hist_list, x, y


def get_hist_info(data, b, sM):
    """For use with hist_axis in getting histogram data."""
    med = np.median(data)
    mea = np.mean(data)
    std = np.std(data)
    return {'med': med, 'mea': mea, 'std': std, 'data': data,
            'b': b, 'sliceMed': sM}


def scatter_plot(dict1, dict2, separate=False):
    """Plots different blocked n values. Deprecated."""
    i = 1
    for n in sorted([x for x in dict1.keys() if x != 'instr']):
        if separate:
            plt.subplot(len(dict1), 1, i)
            plt.xlabel(dict1['instr'])
            plt.ylabel(dict2['instr'])
            plt.title('Number of blocks: {}'.format(n))
        plt.plot([np.nanmin(dict1[n]), np.nanmax(dict1[n])], 
                 [np.nanmin(dict1[n]), np.nanmax(dict1[n])], 'k-', lw=2)
        plt.plot(dict1[n], dict2[n], '.')
        i += 1
    plt.show()


def hist2d(x, y, ax, edges=None, noise=26, lim=1000):
    """Plots a hist2d of the points on a given axis.

    Args:
        x (array): x data to plot
        y (array): y data to plot
        ax (obj): the matplotlib axis object to plot on
        edges (array): the edges for the histograms
        noise (float): the level of field noise you want to ignore
        lim (float): axis limits

    Returns:
        h2d: the hist2d axis
    """
    plt.rc('text', usetex=True)

    if noise != 0:
        bins = (lim // noise)
        edges = np.arange((-bins * noise) + noise, (bins * noise) + noise, noise)
    else:
        edges = 100
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    ind = (np.abs(x) > noise) * (np.abs(y) > noise) * np.isfinite(x) * np.isfinite(y)

    h2d = ax.hist2d(x[ind], y[ind], cmap='inferno', norm=colors.LogNorm(), bins=edges, zorder=1)

    # ------------- Set Plot Properties ----------------------------
    add_identity(ax, color='.3', ls='-', zorder=2)
    ax.axis([-lim, lim, -lim, lim])
    ax.set_facecolor('black')
    ax.set(adjustable='box-forced', aspect='equal')

    return h2d


def scatter_density(x, y, ax, lim=1000, log_vmax=600, cmap='inferno', null_cond=0):
    """Plots a scatter density of the points on a given axis.

    This is useful for high density scatter plots if you don't want to deal with histogram binning.

    Args:
        x (array): x data to plot
        y (array): y data to plot
        ax (obj): the matplotlib axis object to plot on
        lim (float): axis limits, defaults to 1000
        log_vmax (float): the maximum saturation for the log scale on the density, defaults to 600
        cmap (obj): a colormap to use for the density, defaults to inferno
        null_cond (float): a number used to ignore noise in the data below a certain threshold, defaults to 0

    """

    plt.rc('text', usetex=True)
    log_norm = ImageNormalize(vmin=0., vmax=log_vmax, stretch=LogStretch())
    ind = np.isfinite(x) * np.isfinite(y) * (np.abs(x) > null_cond) * (np.abs(y) > null_cond)
    ax.scatter_density(x[ind], y[ind], norm=log_norm, cmap=cmap)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set(adjustable='box-forced', aspect='equal')

def box_plot(bl, disk_limits, ax, clr='blue', corrections=False, **kwargs):
    """Creates a box plot and sets properties."""
    hl, x, y = hist_axis(bl, disk_limits, **kwargs)
    y2 = np.array([t['med'] for t in hl])
    x2 = np.array([s['sliceMed'] for s in hl])
    lim = max(np.max(np.abs(x2)), np.max(np.abs(y2)))*1.1
    box_list = [s['data'] for s in hl]
    box_widths = .4*(max(x2) - min(x2))/len(y2)
    box = ax.boxplot(box_list, widths=box_widths, positions=x2, manage_xticks=False,
                     whis=0, sym="", showcaps=False, patch_artist=True)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set(adjustable='box-forced', aspect='equal')
    add_identity(ax, color='.3', ls='-', linewidth=2, zorder=1)

    for bx in box['boxes']:
        bx.set(edgecolor=clr, facecolor=clr, alpha=.75)

    if corrections:
        total_kernel = gaussian_kde(y)
        xspace = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
        total_kernel_array = total_kernel.evaluate(xspace)
        gaussian_means = []
        gaussian_stds = []
        for bin in hl:
            bin_kernel = gaussian_kde(bin['data'])
            bin_kernel_array = bin_kernel.evaluate(xspace)
            divided_dist = bin_kernel_array/np.sqrt(total_kernel_array)
            max_ind = np.argmax(bin_kernel_array)
            ind_valid = (divided_dist < 1e10)
            try:
                popt = scipy.optimize.least_squares(gaussian_func,
                                                    [np.abs(bin['sliceMed']), np.abs(bin['sliceMed'] / 5),
                                                     bin['sliceMed']],
                                                    args=(xspace[ind_valid], divided_dist[ind_valid]),
                                                    jac='3-point', x_scale='jac', loss='soft_l1', f_scale=.1).x
            except ValueError:
                popt = np.array([np.nan, np.nan, np.nan])
            gaussian_means.append(popt[2])
            gaussian_stds.append(popt[1])
        ax.plot(x2, gaussian_means, color='black', linestyle='None', marker='.', ms=15, alpha=.6, zorder=10)
        for std, mean, x_pos in zip(gaussian_stds, gaussian_means, x2):
            ax.plot([x_pos, x_pos], [mean + std, mean - std], 'k-', alpha=.6, zorder=10)

    return hl


def box_grid(bl, diskCuts=[0, 30, 45, 70], show_fits=True, **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    i1 = bl['i1']
    i2 = bl['i2']
    t_diff = str(round(bl['timeDifference'].total_seconds()/3600, 1)) + ' hours'
    disk_cut_data = []
    f, grid = plt.subplots(1, len(diskCuts) - 1, figsize=(18, 9))
    f.subplots_adjust(wspace=0)
    clrs = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    # -----------------------------Extract box data------------------------------
    for i in range(len(diskCuts)-1):
        disk_cut_data.append(box_plot(bl, diskCuts[i:i+2], grid[i], clr=clrs[i], **kwargs))
        disk_cut_data[i][0]['c'] = clrs[i]

    # --------------------------Setting plot parameters--------------------------
    max_field = np.max([np.max(np.abs(xy)) for ax in grid for xy in (ax.get_ylim(), ax.get_xlim())])
    for plot in grid:
        plot.set_xlim(-max_field, max_field)
        plot.set_ylim(-max_field, max_field)

    grid[1].xaxis.set_ticks_position('top')
    grid[0].set_ylabel(r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()), labelpad=-.75)
    grid[-1].set_ylabel(r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()), labelpad=25,
                        rotation=270)
    grid[1].get_yaxis().set_ticks([])
    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')
    f.text(.5, .17, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i2.upper()),
           horizontalalignment='center')
    fig_title = "Time Difference Between Magnetograms: " + t_diff  # + '\n' + 'n = ' + str(bl['n'])
    f.suptitle(fig_title, y=.85, fontsize=30, fontweight='bold')

    if show_fits:
        lines = ["-", "--", ":"]
        linecycler = cycle(lines)
        colorcycler = cycle(clrs)
        fit_parameters = []
        for h in disk_cut_data:
            fit_parameters.append(fit_xy(np.abs(np.array([s['sliceMed'] for s in h])),
                                         np.abs(np.array([s['med'] for s in h])), 'power'))
        for g in grid:
            for p in fit_parameters:
                plot_fits(p['a'], p['b'], ax=g, color=next(colorcycler), linewidth=3, linestyle=next(linecycler),
                          zorder=1)

    add_box_legend(grid, diskCuts)


def corrected_box_plot(bl, dL, ax, clr='blue', **kwargs):
    """Creates a box plot and sets properties."""
    hl, x, y = hist_axis(bl, dL, **kwargs)
    totalKernel = gaussian_kde(y)
    xspace = np.linspace(np.nanmin(x), np.nanmax(x), 3000)
    totalKernelArray = totalKernel.evaluate(xspace)
    gaussianStds = []
    bxpstats = []

    for bin in hl:
        binKernel = gaussian_kde(bin['data'])
        binKernelArray = binKernel.evaluate(xspace)
        D = binKernelArray/np.sqrt(totalKernelArray)
        indValid = ((D < 1e7))
        try:
            popt = scipy.optimize.least_squares(gaussian_func, 
                    [np.abs(bin['sliceMed']), np.abs(bin['sliceMed']/5), bin['sliceMed']], 
                    args=(xspace[indValid], D[indValid]), jac='3-point', x_scale='jac', 
                    loss='arctan', f_scale=.1).x
        except ValueError:
            try:
                popt = scipy.optimize.least_squares(gaussian_func, 
                    [np.abs(bin['sliceMed']), np.abs(bin['sliceMed']/5), bin['sliceMed']], 
                    args=(xspace[indValid], D[indValid]),
                    loss='soft_l1', f_scale=.1).x
            except ValueError:
                popt = np.array([np.nan, np.nan, np.nan])
        bin['correctedMed'] = popt[2]
        bin['correctedStd'] = popt[1]
        gaussianStds.append(popt[1])
        q1 = bin['correctedMed'] - popt[1]
        q3 = bin['correctedMed'] + popt[1]
        bxpstats.append({'med': bin['correctedMed'], 'q1': q1, 'q3': q3, 'whislo': q1, 'whishi': q3, 'fliers': [np.nan]})

    x = [s['sliceMed'] for s in hl]
    y = [s['correctedMed'] for s in hl]
    boxWidths = .4*(max(x) - min(x))/len(x)
    box = ax.bxp(bxpstats, positions=x, widths=boxWidths, manage_xticks=False, 
            whiskerprops=dict(linestyle='-', color=clr), 
            medianprops=dict(linewidth=3.0), patch_artist=True, 
            showmeans=False, showfliers=False, showcaps=False)

    # for std, mean, x in zip(gaussianStds, y, x):
    #     ax.plot([x, x], [mean+std, mean-std], color=clr, linestyle='-', alpha=.6)

    for bx in box['boxes']:
        bx.set(edgecolor=clr, facecolor=clr, alpha=.75)

    add_identity(ax, color='.3', ls='-', linewidth=3.0, zorder=1)

    return hl


def corrected_box_grid(bl, diskCuts=[0, 30, 45, 70], fullSectorData=None, **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    axis_font = {'size':'30', 'horizontalalignment': 'center', 'verticalalignment': 'center'}

    i1 = bl['i1']
    i2 = bl['i2']
    l = len(diskCuts) - 1
    tDiff = str(round(bl['timeDifference'].total_seconds()/3600, 1)) + ' hours'
    # -----------------------------Extract box data------------------------------
    if l == 3:
        f, grid = plt.subplots(1, 3, sharey=True, figsize=(24, 13))
        f.subplots_adjust(left=.05, right=.94, bottom=.20, top=.75, wspace=0)
        nLims = {'400': 2500, '399': 2250, '300': 2000, '200': 1500,
                '150': 1000, '100': 750, '75': 500, '50': 400, '25': 75}
    elif l == 5:
        f, grid = create_5_plots()
        
    colors = [(80/255, 60/255, 0),
              (81/255, 178/255, 76/255),
              (114/255, 178/255, 229/255),
              (111/255, 40/255, 124/255),
              (255/255, 208/255, 171/255)]
    # -----------------------------Extract box data------------------------------
    if fullSectorData is None:
        diskCutData = []
        for i in range(l):
            diskCutData.append(corrected_box_plot(bl, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))
            diskCutData[i][0]['c'] = colors[i]
    else:
        diskCutData = fullSectorData
    # --------------------------Setting plot parameters--------------------------
    
    if l == 3:
        lim = nLims[str(bl['n'])]
        grid[0].set_ylim(-lim, lim)
        for i, plot in enumerate(grid):
            plot.set_xlim(grid[0].get_ylim())
            plot.set(adjustable='box-forced', aspect='equal')
            if i % 2 == 1: #even number
                plot.xaxis.set_ticks_position('top')
        grid[0].set_ylabel(
            r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
            labelpad=-.75)
        grid[-1].set_ylabel(
            r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
            labelpad=25, rotation=270)
        grid[-1].yaxis.set_ticks_position('right')
        grid[-1].yaxis.set_label_position('right')
        f.text(.40, .13, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i2.upper()), **axis_font)
    elif l == 5:
        lim = np.max(np.array([(max(abs(x['sliceMed']), abs(x['correctedMed'])) + abs(x['correctedStd'])) for sec in diskCutData for x in sec]))*1.05
        for i, plot in enumerate(grid):
            plot.set_xlim(-lim, lim)
            plot.set_ylim(-lim, lim)
            plot.set(adjustable='box-forced', aspect='equal')
        f.text(.17, .5, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()), rotation=90, **axis_font)
        f.text(.83, .5, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()), rotation=270, **axis_font)
        f.text(.5, .065, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i2.upper()), **axis_font)

    fig_title = "Time Difference Between Magnetograms: " + tDiff + \
        '\n' + 'n = ' + str(bl['n'])
    print(fig_title)
    f.suptitle(fig_title, y=.98, fontsize=30, fontweight='bold')

    lines = ["--",":"]
    linecycler = cycle(lines)
    colorcycler = cycle(colors)
    fitParametersPower = []
    fitParametersLinear = []

    for sec in diskCutData:
        x = np.abs(np.array([s['sliceMed'] for s in sec]))
        y = np.abs(np.array([s['correctedMed'] for s in sec]))
        fitParametersPower.append(fit_xy(x, y, 'power'))
        fitParametersLinear.append(fit_xy(x, y, 'linear'))

    for i, g in enumerate(grid):
        c = next(colorcycler)
        plot_fits(fitParametersPower[i]['a'], fitParametersPower[i]['b'], ax=g, color=c,
            linewidth=2, linestyle=next(linecycler), zorder=1)
        plot_fits(fitParametersLinear[i]['a'], 0, fit_type='linear', ax=g, color=c,
                  linewidth=2, linestyle=next(linecycler), zorder=1)

    for ax, letter in zip(grid, 'abcde'):
        ax.annotate(
                '({0})'.format(letter),
                xy=(0,1), xycoords='axes fraction',
                xytext=(5, -24), textcoords='offset points',
                ha='left', va='bottom', fontsize=19)

    add_box_legend(grid, diskCuts)
    add_equations(grid, fitParametersPower, 'power', 'bot')
    add_equations(grid, fitParametersLinear, 'linear', 'top')

    return diskCutData


def violin_plot(bl, dL, ax, clr='blue', alpha=.75, percentiles=[25, 75], corrections=False, **kwargs):
    """Creates a violin plot and sets properties."""
    hl, x, y = hist_axis(bl, dL, **kwargs)
    y2 = np.array([t['med'] for t in hl])
    x2 = np.array([s['sliceMed'] for s in hl])
    ind = ((y > 0) & (x > 0)) | ((y < 0) & (x < 0))
    y = y[ind]
    x = x[ind]
    lim = max(np.max(np.abs(x2)), np.max(np.abs(y2)))*1.1
    dataset = [s['data'] for s in hl]
    p_data = [np.percentile(s, percentiles) for s in dataset]
    refined_dataset = [x[((x > p[0]) & (x < p[1]))] for x, p in zip(dataset, p_data)]
    violin_widths = .6*(max(x2) - min(x2))/len(y2)

    violins = ax.violinplot(refined_dataset, widths=violin_widths, showmedians=True, showextrema=False, positions=x2)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set(adjustable='box-forced', aspect='equal')
    add_identity(ax, color='.3', ls='-', linewidth=2, zorder=1)

    for vio in violins['bodies']:
        vio.set(facecolor=clr, alpha=alpha)

    violins['cmedians'].set(edgecolor='red')
    violins['cmedians'].set_linewidth(2.5)

    if corrections:
        total_kernel = gaussian_kde(y)
        xspace = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
        total_kernel_array = total_kernel.evaluate(xspace)
        gaussian_means = []
        gaussian_stds = []
        for bin in hl:
            bin_kernel = gaussian_kde(bin['data'])
            bin_kernel_array = bin_kernel.evaluate(xspace)
            divided_dist = bin_kernel_array/np.sqrt(total_kernel_array)
            max_ind = np.argmax(bin_kernel_array)
            ind_valid = (divided_dist < 1e10)
            # ind_valid = (divided_dist < divided_dist[max_ind])
            try:
                popt = scipy.optimize.least_squares(gaussian_func,
                                                    [np.abs(bin['sliceMed']), np.abs(bin['sliceMed'] / 5),
                                                     bin['sliceMed']],
                                                    args=(xspace[ind_valid], divided_dist[ind_valid]),
                                                    jac='3-point', x_scale='jac', loss='soft_l1', f_scale=.1).x
                # popt, pcov = curve_fit(gaussian, xspace[ind_valid], divided_dist[ind_valid],
                #     p0=[np.abs(bin['sliceMed']), np.abs(bin['sliceMed']/5), bin['sliceMed']], maxfev=10000)
            except:
                popt = np.array([np.nan, np.nan, np.nan])
            gaussian_means.append(popt[2])
            gaussian_stds.append(popt[1])
        ax.plot(x2, gaussian_means, color='black', linestyle='None', marker='.', ms=15, alpha=.6, zorder=10)
        for std, mean, x_pos in zip(gaussian_stds, gaussian_means, x2):
            ax.plot([x_pos, x_pos], [mean + std, mean - std], 'k-', alpha=.6, zorder=10)

    return hl


def violin_grid(bl, diskCuts=[0, 30, 45, 70], show_fits=True, **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    i1 = bl['i1']
    i2 = bl['i2']
    t_diff = str(round(bl['timeDifference'].total_seconds()/3600, 1)) + ' hours'
    disk_cut_data = []
    f, grid = plt.subplots(1, len(diskCuts) - 1, figsize=(18, 9))
    f.subplots_adjust(wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    # -----------------------------Extract box data------------------------------
    for i in range(len(diskCuts)-1):
        disk_cut_data.append(
            violin_plot(bl, diskCuts[i:i + 2], grid[i], alpha=.4, percentiles=[0, 100], clr=colors[i], **kwargs))
        disk_cut_data.append(violin_plot(bl, diskCuts[i:i + 2], grid[i], alpha=.5, percentiles=[25, 75], clr=colors[i], **kwargs))
        disk_cut_data.append(violin_plot(bl, diskCuts[i:i + 2], grid[i], alpha=.75, percentiles=[37.5, 62.5], clr=colors[i], **kwargs))
        disk_cut_data[i][0]['c'] = colors[i]

    # --------------------------Setting plot parameters--------------------------
    max_field = np.max([np.max(np.abs(xy)) for ax in grid for xy in (ax.get_ylim(), ax.get_xlim())])
    for plot in grid:
        plot.set_xlim(-max_field, max_field)
        plot.set_ylim(-max_field, max_field)

    grid[1].xaxis.set_ticks_position('top')
    grid[0].set_ylabel(
        r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
        labelpad=-.75)
    grid[-1].set_ylabel(
        r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
        labelpad=25, rotation=270)
    grid[1].get_yaxis().set_ticks([])
    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')
    f.text(.5, .17, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i2.upper()),
           horizontalalignment='center')
    fig_title = "Time Difference Between Magnetograms: " + t_diff  # + \
    # '\n' + 'n = ' + str(bl['n'])
    f.suptitle(fig_title, y=.85, fontsize=30, fontweight='bold')

    if show_fits:
        lines = ["-", "--", ":"]
        linecycler = cycle(lines)
        colorcycler = cycle(colors)
        fit_parameters = []
        for h in disk_cut_data:
            fit_parameters.append(fit_xy(np.abs(np.array([s['sliceMed'] for s in h])),
                                         np.abs(np.array([s['med'] for s in h])), 'power'))
        for g in grid:
            for p in fit_parameters:
                plot_fits(p['a'], p['b'], ax=g, color=next(colorcycler), linewidth=3, linestyle=next(linecycler),
                          zorder=1)

    add_box_legend(grid, diskCuts)


def variance_grid(bl, fullSectorData=None, diskCuts=[0, 20, 45, 90], **kwargs):
    #----------Initialize Tex and data if not passed----------------------------
    plt.rc('text', usetex=True)
    if fullSectorData is None:
        fullSectorData = corrected_box_grid(bl, diskCuts, **kwargs)

    i1 = bl['i1']
    i2 = bl['i2']
    tDiff = str(round(bl['timeDifference'].total_seconds()/3600, 1)) + ' hours'
    f, grid = plt.subplots(1, len(diskCuts) - 1, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.94, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    nxLims = {'400': 2500, '300': 2000, '200': 1500, '100': 750, '75': 600, '50': 400, '25': 75}
    nyLims = {'400': 900, '300': 500, '200': 500, '100': 275, '75': 150, '50': 75, '25': 25}
    fitParametersC = []
    fitParameters = []
    colorcycler = cycle(colors)

    for i, g in enumerate(grid):
        sectorData = fullSectorData[i]
        medians = np.array([x['correctedMed'] for x in sectorData])
        stdevs = np.array([x['correctedStd'] for x in sectorData])
        g.scatter(np.abs(medians), np.abs(stdevs), 
                color=colors[i], **kwargs)
        g.set_xlim(0, nxLims[str(bl['n'])])
        g.set_ylim(0, nyLims[str(bl['n'])])
        p = fit_xy(np.abs(medians), np.abs(stdevs), fit_type='powerC')
        fitParameters.append(p)
        c = next(colorcycler)
        plot_fits(p['a'], p['b'], p['c'], fit_type='powerC', ax=g, color=c,
                  linewidth=3, linestyle='--', zorder=1)

    #--------------------------Setting plot parameters--------------------------
    grid[1].xaxis.set_ticks_position('top')
    grid[0].set_ylabel(
        r'$\mathrm{{{0}\ Standard\ Errors\ (Mx/cm^2)}}$'.format(i1.upper()),
        labelpad=-.75)
    grid[2].set_ylabel(
        r'$\mathrm{{{0}\ Standard\ Errors\ (Mx/cm^2)}}$'.format(i1.upper()),
        labelpad=25, rotation=270)
    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')
    f.text(.40, .17, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i2.upper()))
    fig_title = "Time Difference Between Magnetograms: " + tDiff + \
        '\n' + 'n = ' + str(bl['n'])
    f.suptitle(fig_title, y=.85, fontsize=30, fontweight='bold')

    add_equations(grid, fitParameters, 'powerC', 'bot')
    add_box_legend(grid, diskCuts)
    return fitParameters


def extract_fitting_data(iP):
    """Takes the completed box and raw data and extracts fitting parameters."""
    n = []
    i1 = []
    i2 = []
    t = []
    outerA = []
    middleA = []
    centerA = []
    outerB = []
    middleB = []
    centerB = []
    outerC = []
    middleC = []
    centerC = []

    iKey = {'f': '512', 's': 'spmg', 'm': 'mdi', 'h': 'hmi'}
    for key, pairData in iP.items():
        for nValue in [25, 50, 75, 100, 150, 200, 300, 400]:
            try:
                print(key)
                fitParameters = variance_grid(pairData['{0}_data'.format(nValue)], 
                                            pairData['{0}_boxes'.format(nValue)])
                n.append(nValue)
                i1.append(iKey[key[0]])
                i2.append(iKey[key[1]])
                t.append(int(key[2:len(key)]))
                for i, loc in enumerate([(centerA, centerB, centerC), (middleA, middleB, middleC), (outerA, outerB, outerC)]):
                    loc[0].append(fitParameters[i]['a'])
                    loc[1].append(fitParameters[i]['b'])
                    loc[2].append(fitParameters[i]['c'])
            except:
                continue

    a = {'n': n, 'i1': i1, 'i2': i2, 'tDiff': t, 'center': centerA, 'middle': middleA, 'outer': outerA}
    b = {'n': n, 'i1': i1, 'i2': i2, 'tDiff': t, 'center': centerB, 'middle': middleB, 'outer': outerB}
    c = {'n': n, 'i1': i1, 'i2': i2, 'tDiff': t, 'center': centerC, 'middle': middleC, 'outer': outerC}
    dfA = pd.DataFrame(a)
    dfB = pd.DataFrame(b)
    dfC = pd.DataFrame(c)

    return dfA, dfB, dfC


def plot_variance_ns(data):
    d = data
    dt = [1, 3, 12, 24, 36, 48]
    n = [25, 50, 75, 100, 150, 200, 300, 400]

    d['mm1_400'][0][(d['mm1_400'][0] > 4000)] = np.nan
    d['mm1_400'][1][(d['mm1_400'][0] > 4000)] = np.nan
    
    res_sums = 0
    for nVal in n:
        plt.figure(nVal)
        f = plt.gca()
        for time in dt:
            x = d['mm{0}_{1}'.format(time, nVal)][0]
            y = d['mm{0}_{1}'.format(time, nVal)][1]
            scattered = f.plot(x, y, '.', ms=10, label='{0} hrs'.format(time))
            c = scattered[0].get_color()
            params = fit_xy(x, y, 'powerC')
            res_sums += params['res']
            xx = np.linspace(0, np.nanmax(x), 1000)
            yy = power_law(xx, params['a'], params['b'], params['c'])
            if nVal==400 and time==1:
                print(params)
            f.plot(xx, yy, color=c)
        plt.legend()
    print(res_sums)
    plt.show()


def plot_variance_coefficients(df, coeff='a', yAxis='center', xAxis='n', zAxis='tDiff', i1='mdi', i2='mdi'):
    """Takes a dataframe containing fit coefficients for variance grids and
    plots them as a function of yaxis.

    coeff:      what coefficient that is being plotted
    yAxis:      which coefficient (center, middle, outer disk) to plot
    xAxis:      the independent variable to plot_fits
    cAxis:      secondary variable to separate data sets
    i1:         reference instrument
    i2:         secondary instrument
    """

    x = df[xAxis][(df['i1'] == i1) & (df['i2'] == i2)].as_matrix()
    y = df[yAxis][(df['i1'] == i1) & (df['i2'] == i2)].as_matrix()
    z = df[zAxis][(df['i1'] == i1) & (df['i2'] == i2)].as_matrix().astype(np.float32)
    z[(z < 1)] = .25
    norm1 = matplotlib.colors.LogNorm(vmin=np.min(z), vmax=np.max(z))
    norm2 = matplotlib.colors.LogNorm(vmin=np.min(x), vmax=np.max(x))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    for segment in sorted(set(z)):
    
        ind = (z == segment)
        ax1.scatter(x[ind], y[ind], c=cm.viridis(norm1(z[ind][0])), 
                edgecolor='face', cmap='viridis')
        ax1.plot(x[ind], y[ind], 
            c=cm.viridis(norm1(z[ind][0])), 
            label='{0} hr'.format(segment))

    for segment in sorted(set(x)):
        ind = (x == segment)
        ax2.scatter(z[ind], y[ind], c=cm.viridis(norm2(x[ind][0])), 
                edgecolor='face', cmap='viridis')
        ax2.plot(z[ind][np.argsort(z[ind])], y[ind][np.argsort(z[ind])], 
            c=cm.viridis(norm2(x[ind][0])),
            label='n = {0}'.format(segment))

    f = plt.gcf()
    fig_title = "{0}/{1}".format(i1.upper(), i2.upper())
    f.suptitle(fig_title, y=.95, fontsize=30, fontweight='bold')
    ax1.set_ylabel('{0} Disk {1} Coefficient'.format(yAxis.title(), coeff))
    ax1.set_xlabel('{0}'.format(xAxis))
    ax2.set_xlabel('{0}'.format(zAxis))
    plt.legend(loc=4)


def add_box_legend(axes, cuts):
    """
    Takes in a set of mpl axes and adds a legend depicting disk location."""
    plt.rc('text', usetex=True)
    colors =  [(80, 60, 0), (81, 178, 76), (114, 178, 229), 
               (111, 40, 124), (255, 208, 171)]
    colors = [tuple(map(lambda x: x/255, c)) for c in colors]

    for i, plot in enumerate(axes):
        plot.legend(loc=2, 
            handles=[matplotlib.patches.Patch(
                color=colors[i],
                label=r'${:d}^{{\circ}} - {:d}^{{\circ}}$'.format(cuts[i], cuts[i+1]))],
            frameon=False)


def add_equations(axes, fits, fitType = 'power', loc='top'):
    """Formats latex equations for curve fitting and displays them on plots."""

    plt.rc('text', usetex=True)
    if loc=='top': l = 30
    else: l = 5
    if fitType=='powerC':
        baseEqStr = r'$\cdots\ y=({0}\pm {1})x^{{({2}\pm {3})}} + ({4}\pm {5})$'
    if fitType=='power':
        baseEqStr = r'-\ -\ -\ $y=({0}\pm {1})x^{{({2}\pm {3})}}$'
    elif fitType=='linearC':
        baseEqStr = r'$y=({0}\pm {1})x + ({2}\pm {3})$'
    elif fitType=='linear':
        baseEqStr = r'$\cdots\ y=({0}\pm {1})x$'

    if fitType=='power':
        for i, ax in enumerate(axes):
            # aUncSF = -int(np.floor(np.log10(np.abs(fits[i]['aUnc']))))
            # bUncSF = -int(np.floor(np.log10(np.abs(fits[i]['bUnc']))))
            #if aUncSF==0: aUncSF = None
            #if bUncSF==0: bUncSF = None
            ax.annotate(
                baseEqStr.format(
                round(float(fits[i]['a']), 2), 
                round(float(fits[i]['aUnc']), 2), 
                round(float(fits[i]['b']), 2), 
                round(float(fits[i]['bUnc']), 2)), 
                xy=(1,0), xycoords='axes fraction',
                xytext=(-5, l), textcoords='offset points',
                ha='right', va='bottom')
    elif fitType=='powerC':
        for i, ax in enumerate(axes):
            # aUncSF = -int(np.floor(np.log10(np.abs(fits[i]['aUnc']))))
            # bUncSF = -int(np.floor(np.log10(np.abs(fits[i]['bUnc']))))
            # cUncSF = -int(np.floor(np.log10(np.abs(fits[i]['cUnc']))))
            #if aUncSF==0: aUncSF = None
            #if bUncSF==0: bUncSF = None
            #if cUncSF==0: cUncSF = None
            ax.annotate(
                baseEqStr.format(
                round(float(fits[i]['a']), 2), 
                round(float(fits[i]['aUnc']), 2), 
                round(float(fits[i]['b']), 2), 
                round(float(fits[i]['bUnc']), 2),
                round(float(fits[i]['c']), 2), 
                round(float(fits[i]['cUnc']), 2)), 
                xy=(1,0), xycoords='axes fraction',
                xytext=(-5, l), textcoords='offset points',
                ha='right', va='bottom')
    elif fitType=='linearC':
        for i, ax in enumerate(axes):
            # aUncSF = -int(np.floor(np.log10(np.abs(fits[i]['aUnc']))))
            # bUncSF = -int(np.floor(np.log10(np.abs(fits[i]['bUnc']))))
            #if aUncSF==0: aUncSF = None
            ax.annotate(
                baseEqStr.format(
                round(float(fits[i]['a']), 2), 
                round(float(fits[i]['aUnc']), 2), 
                round(float(fits[i]['b']), 2),
                round(float(fits[i]['bUnc']), 2)), 
                xy=(1,0), xycoords='axes fraction',
                xytext=(-5, l), textcoords='offset points',
                ha='right', va='bottom')
    elif fitType=='linear':
        for i, ax in enumerate(axes):
            # aUncSF = abs(int(np.floor(np.log10(np.abs(fits[i]['aUnc'])))))
            ax.annotate(
                baseEqStr.format(
                round(fits[i]['a'], 2), round(fits[i]['aUnc'], 2)), 
                xy=(1,0), xycoords='axes fraction',
                xytext=(-5, l), textcoords='offset points',
                ha='right', va='bottom')


def plot_block_parameters(bl, save=False):
    """
    Accepts any number of p-tuples and creates scatter plots.

    p-tuples take the form of (p_i1, p_i2) where the p values
    for each instrument are calculated from the quadrangles
    module.
    """
    f, ax = plt.subplots(1,2, num=1, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.89, bottom=.08, top=.96, wspace=.10)
    co = 'viridis'
    plt.rc('text', usetex=True)
    i1 = bl['i1']
    i2 = bl['i2']
    tDiff = str(round(bl['timeDifference'].total_seconds()/3600, 1)) + ' hours'
    
    y, x, da = quad.extract_valid_points(bl)

    if not np.sum(np.isfinite(y) & np.isfinite(x)):
        raise ValueError("No suitable points to plot.")

    sortedInds = np.argsort(da)
    f_i1 = y[sortedInds]
    f_i2 = x[sortedInds]
    da = da[sortedInds]

    #--------------------------Mean Field Plot----------------------------------
    ax[0].scatter(f_i2, f_i1, cmap=co, c=da,
            vmin=0, vmax=90, edgecolors='face', zorder=2)
    maxField = max(np.nanmax(f_i1), np.nanmax(f_i2))
    ax[0].set_ylim(-maxField, maxField)
    ax[0].set_xlim(ax[0].get_ylim())

    #-----------------------Mean Field Plot Log Scale---------------------------
    x1 = np.log10(np.abs(f_i2))
    y1 = np.log10(np.abs(f_i1))
    s = (np.abs(f_i1) > 1) & (np.abs(f_i2) > 1)
    plots = ax[1].scatter(x1[s]*np.sign(f_i2[s]), y1[s]*np.sign(f_i1[s]),
        cmap=co, c=da[s], vmin=0, vmax=90, edgecolors='face', zorder=2)
    maxLogField = max(np.nanmax(x1[s]), np.nanmax(y1[s]))
    ax[1].set_ylim(-maxLogField, maxLogField)
    ax[1].set_xlim(ax[1].get_ylim())

    #------------------------Finish Plot Properties-----------------------------
    add_identity(ax[0], color='.3', ls='-', zorder=1)
    add_identity(ax[1], color='.3', ls='-', zorder=1)
    cbar_ax = f.add_axes([.90, .15, .03, .74])
    f.colorbar(plots, cax=cbar_ax)
    ax[0].set(aspect='equal', xlabel=r'$\mathrm{{Linear\ (Mx/cm^2)}}$', 
                ylabel=r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()))
    ax[1].set(aspect='equal', 
                xlabel=r'$\mathrm{{Logarithmic\ (sign*log(abs(Mx/cm^2))}}$')
    f.text(.40, .07, 
        r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ }}$'.format(i2.upper())
        )
    fig_title = "Time Difference Between Magnetograms: " + tDiff + \
        '\n' + 'n = ' + str(bl['n'])
    f.suptitle(fig_title, y=.95, fontsize=30, fontweight='bold')

    return f


def plot_coefficients():
    """Deprecated"""
    f, ax = plt.subplots(1,3, num=1, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.92, bottom=.20, top=.75, wspace=0)

    fitParameters = pd.read_csv('fitParameters.csv')
    i1 = np.array(fitParameters['Instrument 1'])
    i2 = np.array(fitParameters['Instrument 2'])
    a1 = np.array(fitParameters['a1'])
    a2 = np.array(fitParameters['a2'])
    a3 = np.array(fitParameters['a3'])
    t = np.array(fitParameters['Time Difference'])
    n = np.array(fitParameters['n'])

    markerMap = dict(zip(set(zip(i1,i2)), ['o', 'v', '8', '*', 'D']))
    markers = np.array([markerMap[x] for x in zip(i1, i2)])

    lines = ["-","--", ":"]
    colors = [(255/255, 208/255, 171/255), 
            (114/255, 178/255, 229/255), 
            (81/255, 178/255, 76/255),
            (111/255, 40/255, 124/255),
            (80/255, 60/255, 0/255)]
    linecycler = cycle(lines)
    colorcycler = cycle(colors)

    for pair in markerMap:
        instrMask = (pair == np.array(list(zip(i1,i2))))
        instrMask = np.array([x[0] & x[1] for x in instrMask])
        c = next(colorcycler)
        i = 0
        for uniqueTime in set(t.tolist()):
            l = next(linecycler)
            timeMask = (uniqueTime == t)
            mask = timeMask & instrMask
            if not np.sum(mask): continue
            i += 1
            ax[0].scatter(n[mask], a1[mask], s=75, c=c, edgecolor='face', marker=markers[mask][0])
            ax[0].plot(n[mask], a1[mask], color=c, linestyle=l, alpha=.5, label=uniqueTime)
            ax[1].scatter(n[mask], a2[mask], s=75, c=c, edgecolor='face', marker=markers[mask][0])
            ax[1].plot(n[mask], a2[mask], color=c, linestyle=l, alpha=.5)
            ax[2].scatter(n[mask], a3[mask], s=75, c=c, edgecolor='face', marker=markers[mask][0], label="{}, {}".format(pair[0], pair[1]) if i==1 else "_nolegend_")
            ax[2].plot(n[mask], a3[mask], color=c, linestyle=l, alpha=.5)
    

    black_24 = mlines.Line2D([], [], color='black', label=r'$24$ hours')
    black_1 = mlines.Line2D([], [], color='black', linestyle='--', label=r'$< 1$ hour')
    black_48 = mlines.Line2D([], [], color='black', linestyle=':', label=r'$48$ hours')
    ax[0].legend(handles=[black_24, black_1, black_48], bbox_to_anchor=(0., 1.1, 1., .102), loc=8,
           ncol=1, borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(0., 1.1, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

    #--------------------------Setting plot parameters--------------------------
    ax[1].xaxis.set_ticks_position('top')
    ax[0].set_ylabel('Variance Coefficient a', labelpad=-.75)
    ax[2].set_ylabel('Variance Coefficient a', labelpad=25, rotation=270)
    ax[2].yaxis.set_ticks_position('right')
    ax[2].yaxis.set_label_position('right')
    f.text(.4, .14, 'Fragmentation Parameter (n)')
    fig_title = "Variance Fit Slopes versus Fragmentation Parameter"
    f.suptitle(fig_title, y=.97, fontsize=30, fontweight='bold')

    return fitParameters


def to_matlab(iP):
    d = {}
    for key, pairData in iP.items():
        for nValue in [25, 50, 75, 100, 150, 200, 300, 400]:
            try:
                medians = np.abs(np.array([x['correctedMed'] for x in 
                                    pairData['{0}_boxes'.format(str(nValue))][0]]))
                stdevs = np.abs(np.array([x['correctedStd'] for x in 
                                    pairData['{0}_boxes'.format(str(nValue))][0]]))
                dict_str = key + '_' + str(nValue)
                d[dict_str] = np.array([medians, stdevs])
            except: continue
    return d


def create_5_plots():
    yPad = .12
    rowPad = 0.0
    x = 24
    y = 12
    aspectRatio = y/x
    size = (1 - yPad*2 - rowPad*2)/2
    xPad = (1 - size*aspectRatio*3)/2
    grid = []

    fig = plt.figure(figsize=(x, y))
    grid.append(fig.add_axes([xPad, .5 + rowPad, size*aspectRatio, size]))
    grid.append(fig.add_axes([xPad + (size*aspectRatio) - .001, .5 + rowPad, size*aspectRatio, size]))
    grid.append(fig.add_axes([xPad + (size*aspectRatio)*2 - .002, .5 + rowPad, size*aspectRatio, size]))
    grid.append(fig.add_axes([xPad + size*aspectRatio/2, yPad, size*aspectRatio, size]))
    grid.append(fig.add_axes([xPad + size*aspectRatio*3/2 - .001, yPad, size*aspectRatio, size]))

    grid[0].xaxis.set_ticks_position('top')
    grid[0].xaxis.set_label_position('top')
    grid[1].yaxis.set_ticklabels('')
    grid[1].xaxis.set_ticks_position('top')
    grid[2].yaxis.set_ticks_position('right')
    grid[2].xaxis.set_ticks_position('top')
    grid[4].yaxis.set_ticks_position('right')

    for x in grid:
        x.xaxis.set_major_locator(MaxNLocator(nbins=7, prune='both'))
        x.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='both'))


    return fig, grid
