import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
import copy
import itertools
from scipy.optimize import curve_fit
from scipy.odr import *
from itertools import cycle, islice
from mpl_toolkits.axes_grid1 import AxesGrid
import random

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
    return axes

def power_law(x, a, b):
    """Used in fitting the mean fields without the vertical shift."""
    return (a*(x**b))

def power_law_c(x, a, b, c):
    """Used in fitting the mean fields with the vertical shift."""
    return (a*(x**b)) + c

def power_law_odr(p, x):
    """Used in fitting the mean fields."""
    a, b = p
    return a*(x**b)

def fit_mean_field(axes, p):
    """Takes all p values and attempts to fit the fields with a power law."""
    y, x = combine_field_arrays(p)
    ma = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[ma]
    y = y[ma]
    mask = np.sign(x) == np.sign(y)
    popt, pcov = curve_fit(power_law, np.abs(x[mask]), np.abs(y[mask]), maxfev=100000)
    a, b, c = popt
    print("a: {}, b: {}, c: {}".format(a, b, c))
    new_x = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    axes.plot(new_x, power_law(new_x, a, b, c))
    axes.plot(-new_x, -power_law(new_x, a, b, c))

def fit_mean_field_unc(axes, p):
    """
    Takes all p values and attempts to fit the fields with a
    power law with uncertainties.
    """
    y, x, yUnc, xUnc = combine_field_arrays(p, unc=True)
    mask = np.sign(x) == np.sign(y)
    mod = Model(power_law_odr)
    data = RealData(np.abs(x[mask]), np.abs(y[mask]), sx=xUnc[mask], sy=yUnc[mask])
    odr = ODR(data, mod, beta0=[1., 1.])
    out = odr.run()
    out.pprint()
    p = out.beta
    new_x = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    print(p)
    axes.plot(new_x, power_law(new_x, p[0], p[1]))
    axes.plot(-new_x, -power_law(new_x, p[0], p[1]))

def fit_medians(ax, h, **kwargs):
    medy = [s['med'] for s in h]
    medx = [s['sliceMed'] for s in h]
    popt, pcov = curve_fit(power_law, np.abs(medx), np.abs(medy))
    a, b = popt
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        new_x = np.linspace(low, high, 10000)
        ax.plot(new_x, power_law(new_x, a, b), **kwargs)
        ax.plot(-new_x, -power_law(new_x, a, b), **kwargs)
    callback(ax)
    print("a: {}, b: {}".format(a, b))
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)

def fit_xy(x, y, ax, **kwargs):
    popt, pcov = curve_fit(power_law_c, x, y, p0=(1,-1,1), maxfev=10000)
    a, b, c = popt
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        new_x = np.linspace(low, high, 10000)
        ax.plot(new_x, power_law_c(new_x, a, b, c), **kwargs)
    callback(ax)
    print("a: {}, b: {}, c: {}".format(a, b, c))
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)

def hist(p):
    """Deprecated. Calculates a 2D histogram and plots it."""
    x, y = combine_field_arrays(p)
    ind = np.where(np.logical_and(np.isfinite(x), np.isfinite(y)))
    H, xedges, yedges = np.histogram2d(x[ind], y[ind], bins=1000)
    Hmasked = np.ma.masked_where(H==0, H)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.pcolormesh(xedges, yedges, Hmasked, cmap='plasma')
    plt.show()

def hist_colored(p):
    """Deprecated. Calculates a 2D histogram and plots it."""
    x, y = combine_field_arrays(p)
    ind = np.where(np.logical_and(np.isfinite(x), np.isfinite(y)))
    hist, xedges, yedges = np.histogram2d(x[ind], y[ind], bins=1000)
    xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
    c = hist[xidx, yidx]
    sortedInds = np.argsort(c)
    plt.scatter(x[sortedInds], y[sortedInds], c=c[sortedInds],
            edgecolors='face', cmap='plasma')

    plt.show()

def hist_axis(p, diskLimits=[0,90], **kwargs):
    """Returned parameters for multiple histograms."""
    b = kwargs.pop('binCount', 100)
    constant = kwargs.pop('const', 'width')

    y, x = combine_field_arrays(p)
    minimum = np.nanmin(x)
    maximum = np.nanmax(x)
    c, c2 = combine_field_arrays(p, option='color')
    ind = np.where(np.logical_and(
            np.logical_and(np.isfinite(x), np.isfinite(y)),
            np.logical_and(c > diskLimits[0], c < diskLimits[1])
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
            if len(inds[0]) < 10: continue
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

def hist_plot(histList, y, skip, ax):
    """Deprecated"""
    minVal = np.min(histList[0]['data'])
    maxVal = np.max(histList[-1]['data'])
    medians = np.array([x['med'] for x in histList])
    lims = max(abs(minVal*1.10), abs(maxVal*1.10))
    print(lims)
    norm = colors.Normalize(int(medians.min()), int(medians.max()))
    lines = ["-","--"]
    linecycler = cycle(lines)
    for i in range(0, len(histList), skip):
        d = histList[i]['data']
        b = histList[i]['b']
        ax.hist(d, bins=b, histtype='step', normed=True,
                orientation='horizontal', color=cm.viridis(norm(medians[i])), 
                linewidth=3, linestyle=next(linecycler))
        ax.hist(d, bins=b, histtype='stepfilled', normed=True,
                orientation='horizontal', color=cm.viridis(norm(medians[i])), 
                alpha=.15)
        try:
            #ax.set_ylim(-lims, lims)
            pass
        except IndexError:
            pass
            #ax.set_ylim(xedges[minInd], xedges[-1])
    #plt.xscale('log', nonposy='clip')

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

def box_plot(p, dL, ax, clr='blue', **kwargs):
    """Creates a box plot and sets properties."""
    hl, x, y = hist_axis(p, dL, **kwargs)
    medians = np.array([x['med'] for x in hl])
    xPos = [s['sliceMed'] for s in hl]
    boxList = [s['data'] for s in hl]
    lims = max(abs(xPos[0]*1.10), abs(xPos[-1]*1.10))
    box = ax.boxplot(boxList, widths=5, positions=xPos, manage_xticks=False,
            whis=[2.5, 97.5], sym="", showcaps=False,
            whiskerprops=dict(linestyle='-', color=clr), patch_artist=True)
    ax.set(aspect='equal')
    add_identity(ax, color='.3', ls='--', zorder=1)
    i = 0
    for bx in box['boxes']:
        bx.set(edgecolor=clr, facecolor=clr, alpha=.75)
        i += 1

    return hl

def box_grid(p, bl, d, diskCuts=[0, 30, 45, 70], **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    i1 = bl[0][0][0].mgnt.im_raw.instrument
    i2 = bl[0][0][0].mgnt.im_raw.instrument
    tDiff = str(abs((d[0][0] - d[0][1]).total_seconds()//3600)) + ' hours'
    hl = []
    f, grid = plt.subplots(1, 3, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.92, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    grid[0].set_ylim(-200, 200)
    for i in range(len(diskCuts)-1):
        hl.append(box_plot(p, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))
        hl[i][0]['c'] = colors[i]
        grid[i].set_xlim(grid[0].get_ylim())
        grid[i].set_aspect('equal')

    grid[1].xaxis.set_ticks_position('top')
    grid[0].set_ylabel(i1 + ' Field (G)', labelpad=-.75)
    grid[2].set_ylabel(i1 + ' Field (G)', labelpad=25, rotation=270)
    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')
    f.text(.45, .17, i2 + ' Field (G)')
    fig_title = "Time Difference Between Magnetograms: " + tDiff
    f.suptitle(fig_title, y=.83, fontsize=30, fontweight='bold')

    lines = ["-","--", ":"]
    linecycler = cycle(lines)

    for ax, h in itertools.product(grid, hl):
        fit_medians(ax, h, color=h[0]['c'], linewidth=3, linestyle=next(linecycler), zorder=1)

    add_box_legend(grid, diskCuts)
    f.savefig('pict.png', bbox_inches='tight', pad_inches = 0.1)

def variance_plot(p, dL, ax, clr='blue', **kwargs):
    hl, x, y = hist_axis(p, dL, **kwargs)
    medians = np.array([x['med'] for x in hl])
    stdevs = np.array([s['std'] for s in hl])

    box = ax.plot(np.abs(medians), np.abs(stdevs/medians), '.', color=clr, markersize=10, **kwargs)

    return hl

def variance_grid(p, bl, d, diskCuts=[0, 20, 45, 90], **kwargs):
    i1 = bl[0][0][0].mgnt.im_raw.instrument
    i2 = bl[0][0][0].mgnt.im_raw.instrument
    tDiff = str(abs((d[0][0] - d[0][1]).total_seconds()//3600)) + ' hours'
    hl = []
    f, grid = plt.subplots(1, 3, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.92, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    grid[0].set_ylim(0, 1.5)
    for i in range(len(diskCuts)-1):
        hl.append(variance_plot(p, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))  
        hl[i][0]['c'] = colors[i]
        grid[i].set_xlim(0, 200)
        #grid[i].set_aspect('equal')

    grid[0].set_ylabel(i1 + ' Field (G)', labelpad=-.75)
    grid[2].set_ylabel(i1 + ' Field (G)', labelpad=25, rotation=270)
    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')

    fig_title = "Time Difference Between Magnetograms: " + tDiff
    f.suptitle(fig_title, y=.83, fontsize=30, fontweight='bold')

    lines = ["-","--", ":"]
    linecycler = cycle(lines)

    for ax, h in itertools.product(grid, hl):
        x = np.array([x['med'] for x in h])
        y = np.array([s['std'] for s in h])/x
        fit_xy(np.abs(x), np.abs(y), ax, color=h[0]['c'], linewidth=3, linestyle=next(linecycler), zorder=1)

    add_box_legend(grid, diskCuts)

def var2_grid(p, bl, d, p2, diskCuts=[0, 20, 45, 90], **kwargs):
    i1 = bl[0][0][0].mgnt.im_raw.instrument
    i2 = bl[0][0][0].mgnt.im_raw.instrument
    tDiff = str(abs((d[0][0] - d[0][1]).total_seconds()//3600)) + ' hours'
    hl1 = []
    hl2 = []
    f, grid = plt.subplots(1, 3, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.92, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    grid[0].set_ylim(0, 1.5)
    for i in range(len(diskCuts)-1):
        hl1.append(variance_plot(p, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))
        hl2.append(variance_plot(p2, diskCuts[i:i+2], grid[i], clr=colors[i], marker='*', **kwargs))
        grid[i].set_xlim(0, 200)
        #grid[i].set_aspect('equal')

    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')

    lines = ["-","--", ":"]
    linecycler = cycle(lines)

    # for ax, h in itertools.product(grid, hl):
    #     x = np.array([x['med'] for x in h])
    #     y = np.array([s['std'] for s in h])/x
    #     fit_xy(np.abs(x), np.abs(y), ax, color=h[0]['c'], linewidth=3, linestyle=next(linecycler), zorder=1)

    add_box_legend(grid, diskCuts)


def add_box_legend(axes, cuts):
    c1 = cuts[0]
    brown_patch = matplotlib.patches.Patch(color=(80/255, 60/255, 0), 
            label=r'$0\degree-20\degree$')
    green_patch = matplotlib.patches.Patch(color=(81/255, 178/255, 76/255), 
            label=r'$20\degree-45\degree$')
    blue_patch = matplotlib.patches.Patch(color=(114/255, 178/255, 229/255), 
            label=r'$45\degree-90\degree$')
    axes[0].legend(loc=2, handles=[brown_patch], frameon=False)
    axes[1].legend(loc=2, handles=[green_patch], frameon=False)
    axes[2].legend(loc=2, handles=[blue_patch], frameon=False)


def plot_block_parameters(*args):
    """
    Accepts any number of p-tuples and creates scatter plots.

    p-tuples take the form of (p_i1, p_i2) where the p values
    for each instrument are calculated from the quadrangles
    module.
    """
    f, ax = plt.subplots(1,3, num=1)
    co = 'viridis'
    plt.rc('text', usetex=True)
    ar_i1, ar_i2 = combine_field_arrays(args, option='area')
    f_i1, f_i2 = combine_field_arrays(args, option='field')
    c, c2 = combine_field_arrays(args, option='color')
    
    ind = np.where(np.abs((ar_i1 - ar_i2)/ar_i1) < .10)
    c_valid = c[ind]
    sortedInds = np.argsort(c_valid)
    #ar_i1 = ar_i1[ind][sortedInds]
    #ar_i2 = ar_i2[ind][sortedInds]
    f_i1 = f_i1[ind][sortedInds]
    f_i2 = f_i2[ind][sortedInds]

    f1_valid = (np.abs(f_i1) > 1)
    f2_valid = (np.abs(f_i2) > 1)
    s = f1_valid & f2_valid

    #-----------------------------Area Plot-------------------------------------
    ax[0].scatter(ar_i2, ar_i1, cmap=co,
            vmin=0, vmax=90, edgecolors='face', zorder=2)
    ax[0].set_xlim(ax[0].get_ylim())

    #--------------------------Mean Field Plot----------------------------------
    ax[1].scatter(f_i2, f_i1, cmap=co, c=c_valid,
            vmin=0, vmax=90, edgecolors='face', zorder=2)
    maxField = max(np.nanmax(f_i1), np.nanmax(f_i2))
    ax[1].set_ylim(-maxField, maxField)
    ax[1].set_xlim(ax[1].get_ylim())

    #-----------------------Mean Field Plot Log Scale---------------------------
    x1 = np.log10(np.abs(f_i2[s]))
    y1 = np.log10(np.abs(f_i1[s]))
    plots = ax[2].scatter(x1*np.sign(f_i2[s]), y1*np.sign(f_i1[s]),
        cmap=co, c=c_valid[s], vmin=0, vmax=90, edgecolors='face', zorder=2)
    maxLogField = max(np.nanmax(x1), np.nanmax(y1))
    ax[2].set_ylim(-maxLogField, maxLogField)
    ax[2].set_xlim(ax[2].get_ylim())

    #------------------------Finish Plot Properties-----------------------------
    set_p_plot_properties(ax)
    #fit_mean_field(ax[1], args)
    #fit_mean_field_unc(ax[1], args)
    #cbar.ax.set_ylabel('Degrees from disk center')
    f.subplots_adjust(left=.05, right=.89, bottom=.29, top=.71, wspace=.15)
    cbar_ax = f.add_axes([.90, .29, .03, .42])
    f.colorbar(plots, cax=cbar_ax)
    return f

def n_plot(n_dict):
    """Plots a dictionary of n values"""
    NList = []
    TList = []
    for key, value in n_dict.items():
        NList.append(key)
        TList.append(len(value))
    N = np.array(NList)
    T = np.array(TList)
    plt.plot(N, T, '.')

    plt.show()

def block_plot(*args, overlay=True):
    """Given a list of blocks, will plot a nice image differentiating them.

    --Optional arguments--
    overlay: Toggle for quadrangles if you want to show over magnetogram."""
    n = len(args)
    print(n)
    rows = int(round(np.sqrt(n)))
    cols = int(np.ceil(np.sqrt(n)))
    im = {}
    ax = {}
    for i in range(len(args)):
        print(i)
        if isinstance(args[i], type(list())):
            im[i] = args[i][0].mgnt.lonh.v.copy()
            im[i][:] = np.nan
            for x in args[i]:
                im[i][x.indices] = x.pltColor
            ax[i] = plt.subplot2grid((rows, cols), (i%rows, i//rows))
            if overlay:
                ax[i].imshow(args[i][0].mgnt.im_raw.data, cmap='binary')
                ax[i].imshow(im[i], vmin=0, vmax=1, alpha=.2)
            else:
                ax[i].imshow(im[i], vmin=0, vmax=1)
            title = "{}: {}".format(args[i][0].mgnt.im_raw.instrument, 
                    args[i][0].mgnt.im_raw.date.isoformat())
            ax[i].set_title(title)
        else:
            im[i] = args[i]
            ax[i] = plt.subplot2grid((rows, cols), (i%rows, i//rows))
            ax[i].imshow(im[i], cmap='binary')

    return ax

def combine_field_arrays(p, unc=False, option='field'):
    """Takes in a list of p-values and returns a concatenated numpy array."""
    x = np.array([])
    xUnc = np.array([])
    y = np.array([])
    yUnc = np.array([])
    if option == 'area':
        i = 0
    elif option == 'field':
        i =  1
    elif option == 'color':
        i = 2
    for s in range(len(p)):
        x = np.append(x, p[s][0][i])
        y = np.append(y, p[s][1][i])
        if unc:
            xUnc = np.append(xUnc, p[s][0][i + 3])
            yUnc = np.append(yUnc, p[s][1][i + 3])

    if unc:
        return x, y, xUnc, yUnc
    else:
        return x, y

def set_p_plot_properties(ax):
    """Sets the scatter plot properties."""
    add_identity(ax[0], color='.3', ls='--', zorder=1)
    add_identity(ax[1], color='.3', ls='--', zorder=1)
    add_identity(ax[2], color='.3', ls='--', zorder=1)

    ax[0].set_title('Block Areas')
    ax[1].set_title('Mean Field')
    ax[0].set(aspect='equal', xlabel=r'Area (cm^2)',
            ylabel=r'Reference Area (cm^2)')
    ax[1].set(aspect='equal', xlabel=r'B Field (G)',
            ylabel=r'Reference B Field (G)')
    ax[2].set_title('Logarithm Space of Fields')
    ax[2].set(aspect='equal', xlabel=r'Logarithmic B Field (log(B))',
            ylabel=r'Reference Logarithmic B Field (log(B))')