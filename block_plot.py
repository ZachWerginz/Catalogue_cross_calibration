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
    medians = np.array([x['med'] for x in hl])
    xPos = [s['sliceMed'] for s in hl]
    boxList = [s['data'] for s in hl]
    lims = max(abs(xPos[0]*1.10), abs(xPos[-1]*1.10))
    box = ax.boxplot(boxList, widths=5, positions=xPos, manage_xticks=False,
            whis=[10, 90], sym="", showcaps=False,
            whiskerprops=dict(linestyle='-', color=clr), patch_artist=True)
    ax.set(aspect='equal')
    add_identity(ax, color='.3', ls='--', zorder=1)
    i = 0
    for bx in box['boxes']:
        bx.set(edgecolor=clr, facecolor=clr, alpha=.75)
        i += 1

    return hl

def box_grid(bl, diskCuts=[0, 30, 45, 70], **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    i1 = bl['i1']
    i2 = bl['i2']
    tDiff = str(bl['timeDifference'].total_seconds()/3600) + ' hours'
    hl = []
    f, grid = plt.subplots(1, 3, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.92, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    grid[0].set_ylim(-200, 200)
    for i in range(len(diskCuts)-1):
        hl.append(box_plot(bl, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))
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

def violin_plot(bl, dL, ax, clr='blue', **kwargs):
    """Creates a box plot and sets properties."""
    hl, x, y = hist_axis(bl, dL, **kwargs)
    medians = np.array([x['med'] for x in hl])
    xPos = [s['sliceMed'] for s in hl]
    boxList = [s['data'] for s in hl]
    lims = max(abs(xPos[0]*1.10), abs(xPos[-1]*1.10))
    vio = ax.violinplot(boxList, widths=5, positions=xPos,
            showmeans=True)
    ax.set(aspect='equal')
    add_identity(ax, color='.3', ls='--', zorder=1)
    i = 0
    # for bx in box['boxes']:
    #     bx.set(edgecolor=clr, facecolor=clr, alpha=.75)
    #     i += 1

    return 

def violin_grid(bl, diskCuts=[0, 30, 45, 70], **kwargs):
    """Splits box plots into sections of degrees from disk center."""

    i1 = bl[0][0].i1
    i2 = bl[0][0].i2
    tDiff = str(abs((bl[0][0].date1 - bl[0][0].date2).total_seconds()//3600)) + ' hours'
    hl = []
    f, grid = plt.subplots(1, 3, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.92, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    grid[0].set_ylim(-200, 200)
    for i in range(len(diskCuts)-1):
        hl.append(box_plot(bl, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))
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

def variance_plot(bl, dL, ax, clr='blue', **kwargs):
    hl, x, y = hist_axis(bl, dL, **kwargs)
    medians = np.array([x['med'] for x in hl])
    stdevs = np.array([s['std'] for s in hl])

    variance = ax.plot(np.abs(medians), np.abs(stdevs/medians), '.', color=clr, markersize=10, **kwargs)

    return hl

def variance_grid(bl, diskCuts=[0, 20, 45, 90], **kwargs):
    i1 = bl['i1']
    i2 = bl['i2']
    tDiff = str(bl['timeDifference'].total_seconds()/3600) + ' hours'
    hl = []
    f, grid = plt.subplots(1, 3, sharey=True, figsize=(24, 13))
    f.subplots_adjust(left=.05, right=.92, bottom=.20, top=.75, wspace=0)
    colors = [(80/255, 60/255, 0), (81/255, 178/255, 76/255), (114/255, 178/255, 229/255)]
    grid[0].set_ylim(0, 1.5)
    for i in range(len(diskCuts)-1):
        hl.append(variance_plot(bl, diskCuts[i:i+2], grid[i], clr=colors[i], **kwargs))  
        hl[i][0]['c'] = colors[i]
        grid[i].set_xlim(0, 200)
        #grid[i].set_aspect('equal')

    grid[2].yaxis.set_ticks_position('right')
    grid[2].yaxis.set_label_position('right')

    fig_title = "Time Difference Between Magnetograms: " + tDiff
    f.suptitle(fig_title, y=.83, fontsize=30, fontweight='bold')

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

def plot_block_parameters(bl):
    """
    Accepts any number of p-tuples and creates scatter plots.

    p-tuples take the form of (p_i1, p_i2) where the p values
    for each instrument are calculated from the quadrangles
    module.
    """
    f, ax = plt.subplots(1,2, num=1)
    co = 'viridis'
    plt.rc('text', usetex=True)
    
    y, x, da = quad.extract_valid_points(bl)
    sortedInds = np.argsort(da)
    f_i1 = y[sortedInds]
    f_i2 = x[sortedInds]

    f1LogValid = (np.abs(y) > 1)
    f2LogValid = (np.abs(x) > 1)
    s = f1LogValid & f2LogValid

    #--------------------------Mean Field Plot----------------------------------
    ax[0].scatter(f_i2, f_i1, cmap=co, c=da,
            vmin=0, vmax=90, edgecolors='face', zorder=2)
    maxField = max(np.nanmax(f_i1), np.nanmax(f_i2))
    ax[0].set_ylim(-maxField, maxField)
    ax[0].set_xlim(ax[0].get_ylim())

    #-----------------------Mean Field Plot Log Scale---------------------------
    x1 = np.log10(np.abs(f_i2[s]))
    y1 = np.log10(np.abs(f_i1[s]))
    plots = ax[1].scatter(x1*np.sign(f_i2[s]), y1*np.sign(f_i1[s]),
        cmap=co, c=da[s], vmin=0, vmax=90, edgecolors='face', zorder=2)
    maxLogField = max(np.nanmax(x1), np.nanmax(y1))
    ax[1].set_ylim(-maxLogField, maxLogField)
    ax[1].set_xlim(ax[1].get_ylim())

    #------------------------Finish Plot Properties-----------------------------
    set_p_plot_properties(ax)
    f.subplots_adjust(left=.05, right=.89, bottom=.08, top=.96, wspace=.10)
    cbar_ax = f.add_axes([.90, .29, .03, .42])
    f.colorbar(plots, cax=cbar_ax)
    return f

def set_p_plot_properties(ax):
    """Sets the scatter plot properties."""
    add_identity(ax[0], color='.3', ls='--', zorder=1)
    add_identity(ax[1], color='.3', ls='--', zorder=1)

    ax[0].set_title('Mean Field')
    ax[0].set(aspect='equal', xlabel=r'B Field (G)',
            ylabel=r'Reference B Field (G)')
    ax[1].set_title('Logarithm Space of Fields')
    ax[1].set(aspect='equal', xlabel=r'Logarithmic B Field (log(B))',
            ylabel=r'Reference Logarithmic B Field (log(B))')