import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib
import copy
import itertools
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import scipy
import scipy.interpolate as interpolate
from scipy.odr import *
from itertools import cycle, islice
from mpl_toolkits.axes_grid1 import AxesGrid
import random
import ransac
import quadrangles as quad

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

matplotlib.rcParams.update({'font.size': 22})

def add_identity(axes, *line_args, **line_kwargs):
    """Plots the identity line on the specified axis."""
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)

def power_law(x, a, b, c):
    return (a*(x**b)) + c

def power_func(p, x, y):
    return (p[0]*(x**p[1])) + 5 - y

def power_func_intercept(p, x, y):
    return (p[0]*(x**p[1])) + p[2] - y

def linear_law(x, a, b):
    return a*x + b

def linear_func(p, x, y):
    return (p*x) - y

def gaussian(x, A, sigma, mu):
    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))

def gaussian_func(p, x, y):
    return p[0]/(np.sqrt(2*np.pi)*p[1])*np.exp(-(x-p[2])**2/(2.*p[1]**2)) - y

def fit_medians(h, fitType='power', corrected=False):
    """Deprecated"""
    if corrected:
        medy = [s['correctedMed'] for s in h]
    else:
        medy = [s['med'] for s in h]
    medx = [s['sliceMed'] for s in h]
    ind = (np.isfinite(medx) & np.isfinite(medy))
    if fitType=='power':
        popt = scipy.optimize.least_squares(power_func, [1, 1], args=(np.abs(medx)[ind], np.abs(medy)[ind]), loss='soft_l1', f_scale=.1)
    elif fitType=='linear':
        popt = scipy.optimize.least_squares(linear_func, [1], args=(np.abs(medx)[ind], np.abs(medy)[ind]), loss='soft_l1', f_scale=.1)
    _, s, VT = np.linalg.svd(popt.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(popt.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    stderr = np.sqrt(np.diag(pcov))
    if fitType=='power':
        return {'a': popt.x[0], 'b': popt.x[1], 'aUnc': stderr[0], 'bUnc': stderr[1], 'sse': np.sum(popt.fun**2), 'n': len(medx), 'sst': np.sum((medy-np.mean(medy))**2)}
    elif fitType=='linear':
        return {'a': popt.x[0], 'aUnc': stderr[0], 'sse': np.sum(popt.fun**2), 'n': len(medx), 'sst': np.sum((medy-np.mean(medy))**2)}

def fit_xy(x, y, fitType='power'):
    ind = (np.isfinite(x) & np.isfinite(y))
    if fitType=='power':
        popt = scipy.optimize.least_squares(power_func, [1, 1],
                        args=(x[ind], y[ind]), bounds=([0,0], [np.inf, np.inf]),
                        ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale='jac',
                        jac='3-point', loss='soft_l1', f_scale=.1)
    elif fitType=='powerC':
        popt = scipy.optimize.least_squares(power_func_intercept, [1, .5, 1], 
                        args=(x[ind], y[ind]), max_nfev=10000,
                        bounds=([0,0,0], [np.inf, np.inf,np.inf]), 
                        ftol=1e-3, xtol=1e-3, gtol=1e-3, x_scale='jac', 
                        jac='3-point', loss='soft_l1', f_scale=.5)
    elif fitType=='linear':
        popt = scipy.optimize.least_squares(linear_func, [1],
                        args=(x[ind], y[ind]), loss='huber', f_scale=.1)
    #---------------Acquire fit uncertainties-----------------------------------
    _, s, VT = np.linalg.svd(popt.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(popt.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    stderr = np.sqrt(np.diag(pcov))

    #--------------Return appropriate parameters--------------------------------
    if fitType=='power':
        return {'a': popt.x[0], 'aUnc': stderr[0], 
                'b': popt.x[1], 'bUnc': stderr[1],
                'n': len(x), 'res': np.sum(popt.fun/y[ind])}
    elif fitType=='powerC':
        return {'a': popt.x[0], 'aUnc': stderr[0], 
                'b': popt.x[1], 'bUnc': stderr[1], 
                'c': popt.x[2], 'cUnc': stderr[2],
                'n': len(x), 'res': np.sum(popt.fun/y[ind])}
    elif fitType=='linear':
        return {'a': popt.x[0], 'aUnc': stderr[0], 'n': len(x), 'res': np.sum(popt.fun/y[ind])}

def plot_fits(a, b, c=5, fitType='power', ax=None, **kwargs):
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        new_x = np.linspace(low, high, 10000)
        if fitType == 'power':
            ax.plot(new_x, power_law(new_x, a, b, 0), **kwargs)
            ax.plot(-new_x, -power_law(new_x, a, b, 0), **kwargs)
        elif fitType=='powerC':
            ax.plot(new_x, power_law(new_x, a, b, c), **kwargs)
        elif fitType == 'linear':
            ax.plot(new_x, linear_law(new_x, a, b), **kwargs)
    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)

def hist_axis(bl, diskLimits=[0,90], **kwargs):
    """Returned parameters for multiple histograms."""
    b = kwargs.pop('binCount', 100)
    constant = kwargs.pop('const', 'width')

    y, x, da = quad.extract_valid_points(bl)

    minimum = np.nanmin(x)
    maximum = np.nanmax(x)
    ind = np.where(np.logical_and(
            np.logical_and(np.isfinite(x), np.isfinite(y)),
            np.logical_and(da > diskLimits[0], da < diskLimits[1])
            )
        )
    x = x[ind]
    y = y[ind]
    histList = []

    #Split by constant bin width
    if constant == 'width':
        xedges = np.linspace(minimum, maximum, b)
        for i in range(1, xedges.shape[0] - 1):
            inds = np.where(
                np.logical_and(
                    x < xedges[i], x >= xedges[i-1]))
            if len(inds[0]) < 20: continue
            sliceMed = (xedges[i] + xedges[i-1]) / 2
            histList.append(get_hist_info(y[inds], xedges, sliceMed))
    #Split by constant bin count
    else: 
        sortInd = np.argsort(x)
        y = y[sortInd]
        s = 0
        e = 100
        for i in range(0, len(y), b):
            sliceMed = np.median(x[s:e])
            histList.append(get_hist_info(y[s:e], 10, sliceMed))
            s += b
            e += b

    return histList, x, y

def get_hist_info(data, b, sM):
    """Deprecated. For use with hist_axis in getting histogram data."""
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

def box_plot(bl, dL, ax, clr='blue', **kwargs):
    """Creates a box plot and sets properties."""
    hl, x, y = hist_axis(bl, dL, **kwargs)
    totalKernel = gaussian_kde(y)
    xspace = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    totalKernelArray = totalKernel.evaluate(xspace)

    gaussianMeans = []
    gaussianStds = []
    for bin in hl:
        binKernel = gaussian_kde(bin['data'])
        binKernelArray = binKernel.evaluate(xspace)
        D = binKernelArray/np.sqrt(totalKernelArray)
        maxInd = np.argmax(binKernelArray)
        indValid = ((D < D[maxInd]))
        try:
            popt, pcov = curve_fit(gaussian, xspace[indValid], 
                D[indValid], 
                p0=[np.abs(bin['sliceMed']), np.abs(bin['sliceMed']/5), bin['sliceMed']], 
                maxfev=10000)
        except:
            popt = np.array([np.nan, np.nan, np.nan])
        gaussianMeans.append(popt[2])
        gaussianStds.append(popt[1])


    y = np.array([x['med'] for x in hl])
    x = [s['sliceMed'] for s in hl]
    boxList = [s['data'] for s in hl]
    lim = max(abs(x[0]*1.10), abs(x[-1]*1.10))
    boxWidths = .4*(max(x) - min(x))/len(y)
    box = ax.boxplot(boxList, widths=boxWidths, positions=x, manage_xticks=False,
            whis=[10, 90], sym="", showcaps=False, showmeans=True,
            whiskerprops=dict(linestyle='-', color=clr), patch_artist=True)
    #ax.plot(x, gaussianMeans, color='black', linestyle='None', marker='.', ms=15, alpha=.6)
    
    ax.set_xlim(-lim, lim)
    ax.set(aspect='equal')
    add_identity(ax, color='.3', ls='--', zorder=1)

    for std, mean, xPos in zip(gaussianStds, gaussianMeans, x):
        ax.plot([xPos, xPos], [mean+std, mean-std], 'k-', alpha=.6)

    for bx in box['boxes']:
        bx.set(edgecolor=clr, facecolor=clr, alpha=.75)

    return hl

def box_grid(bl, diskCuts=[0, 30, 45, 70], **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    i1 = bl['i1']
    i2 = bl['i2']
    tDiff = str(round(bl['timeDifference'].total_seconds()/3600, 1)) + ' hours'
    diskCutData = []
    f, grid = plt.subplots(1, len(diskCuts) - 1, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.94, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    #-----------------------------Extract box data------------------------------
    for i in range(len(diskCuts)-1):
        diskCutData.append(box_plot(bl, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))
        diskCutData[i][0]['c'] = colors[i]

    #--------------------------Setting plot parameters--------------------------
    maxField = max([s['med'] for hl in diskCutData for s in hl])
    grid[0].set_ylim(-maxField, maxField)

    for plot in grid:
        plot.set_xlim(grid[0].get_ylim())

    grid[1].xaxis.set_ticks_position('top')
    grid[0].set_ylabel(i1 + ' Field (G)', labelpad=-.75)
    grid[2].set_ylabel(i1 + ' Field (G)', labelpad=25, rotation=270)
    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')
    f.text(.45, .17, i2 + ' Field (G)')
    fig_title = "Time Difference Between Magnetograms: " + tDiff + \
        '\n' + 'n = ' + str(bl['n'])
    f.suptitle(fig_title, y=.85, fontsize=30, fontweight='bold')

    lines = ["-","--", ":"]
    linecycler = cycle(lines)
    colorcycler = cycle(colors)
    fitParameters = []

    for h in diskCutData:
        fitParameters.append(fit_xy(
            np.abs(np.array([s['sliceMed'] for s in h])), 
            np.abs(np.array([s['med'] for s in h])),
            'power'))

    for g in grid:
        for p in fitParameters:
            plot_fits(p['a'], p['b'], ax=g, color=next(colorcycler),
                linewidth=3, linestyle=next(linecycler), zorder=1)

    add_box_legend(grid, diskCuts)
    #f.savefig('pict.png', bbox_inches='tight', pad_inches = 0.1)

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
                    args=(xspace[indValid], D[indValid]), 
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

    for std, mean, x in zip(gaussianStds, y, x):
        ax.plot([x, x], [mean+std, mean-std], color=clr, linestyle='-', alpha=.6)

    for bx in box['boxes']:
        bx.set(edgecolor=clr, facecolor=clr, alpha=.75)

    ax.set(aspect='equal')
    add_identity(ax, color='.3', ls='-', linewidth=3.0, zorder=1)


    return hl

def corrected_box_grid(bl, diskCuts=[0, 30, 45, 70], fullSectorData=None, return_fits=False, **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    i1 = bl['i1']
    i2 = bl['i2']
    l = len(diskCuts) - 1
    tDiff = str(round(bl['timeDifference'].total_seconds()/3600, 1)) + ' hours'
    diskCutData = []
    if l == 3:
        rows = 1
        columns = 3
        f, grid = plt.subplots(rows, columns, sharey=True, figsize=(24, 13))
        f.subplots_adjust(left=.05, right=.94, bottom=.20, top=.75, wspace=0)
    elif l == 5:
        f = plt.figure(figsize=(24, 13))
        rows = 2
        columns = 3
        grid = []
        grid.append(plt.subplot2grid((2,4), (0,0)))
        grid.append(plt.subplot2grid((2,4), (0,1), colspan=2, sharey=grid[0]))
        grid.append(plt.subplot2grid((2,4), (0,3), sharey=grid[0]))
        grid.append(plt.subplot2grid((2,4), (1,0), colspan=2, sharey=grid[0]))
        grid.append(plt.subplot2grid((2,4), (1,2), colspan=2, sharey=grid[3]))
        f.subplots_adjust(left=.03, right=.94, bottom=.06, top=.77, hspace=.15, wspace=0)

    colors =   [(80/255, 60/255, 0), 
                (81/255, 178/255, 76/255),
                (114/255, 178/255, 229/255),
                (111/255, 40/255, 124/255),
                (255/255, 208/255, 171/255)]
    #-----------------------------Extract box data------------------------------
    if fullSectorData is None:
        for i in range(l):
            diskCutData.append(corrected_box_plot(bl, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))
            diskCutData[i][0]['c'] = colors[i]
    else:
        diskCutData = fullSectorData
    #--------------------------Setting plot parameters--------------------------
    nLims = {'400': 2500, '399': 2250, '300': 2000, '200': 1500, '150': 1000, '100': 750, '75': 500, '50': 400, '25': 75}
    lim = nLims[str(bl['n'])]
    grid[0].set_ylim(-lim, lim)

    for i, plot in enumerate(grid):
        plot.set_xlim(grid[0].get_ylim())
        plot.set(adjustable='box-forced', aspect='equal')
        if i % 2 == 1: #even number
            plot.xaxis.set_ticks_position('top')

    #grid[1].xaxis.set_ticks_position('top')
    grid[0].set_ylabel(
        r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
        labelpad=-.75)
    grid[-1].set_ylabel(
        r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
        labelpad=25, rotation=270)
    grid[-1].yaxis.set_ticks_position('right')
    grid[-1].yaxis.set_label_position('right')
    f.text(.40, .13, r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i2.upper()))
    fig_title = "Time Difference Between Magnetograms: " + tDiff + \
        '\n' + 'n = ' + str(bl['n'])
    f.suptitle(fig_title, y=.95, fontsize=30, fontweight='bold')

    lines = ["--",":"]
    linecycler = cycle(lines)
    colorcycler = cycle(colors)
    fitParametersPower = []
    fitParametersLinear = []

    for sec in diskCutData:
        x = np.abs(np.array([s['sliceMed'] for s in sec]))
        y= np.abs(np.array([s['correctedMed'] for s in sec]))
        fitParametersPower.append(fit_xy(x, y, 'power'))
        fitParametersLinear.append(fit_xy(x, y, 'linear'))

    for i, g in enumerate(grid):
        c = next(colorcycler)
        plot_fits(fitParametersPower[i]['a'], fitParametersPower[i]['b'], ax=g, color=c,
            linewidth=2, linestyle=next(linecycler), zorder=1)
        plot_fits(fitParametersLinear[i]['a'], 0, fitType='linear', ax=g, color=c,
            linewidth=2, linestyle=next(linecycler), zorder=1)

    add_box_legend(grid, diskCuts)
    add_equations(grid, fitParametersPower, 'power', 'bot')
    add_equations(grid, fitParametersLinear, 'linear', 'top')

    if return_fits:
        return diskCutData, fitParametersPower, fitParametersLinear
    else:
        return diskCutData

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
        p = fit_xy(np.abs(medians), np.abs(stdevs), fitType='powerC')
        fitParameters.append(p)
        c = next(colorcycler)
        plot_fits(p['a'], p['b'], p['c'], fitType='powerC', ax=g, color=c,
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

    if  not np.sum(np.isfinite(y) & np.isfinite(x)):
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
