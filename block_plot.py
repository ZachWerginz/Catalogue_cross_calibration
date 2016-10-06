import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
from scipy.optimize import curve_fit

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
    """Used in fitting the mean fields."""
    return a*(x**b)

def fit_mean_field(axes, p):
    """Takes all p values and attempts to fit the fields with a power law."""
    xArr = np.array([])
    yArr = np.array([])
    for s in range(len(p)):
        xArr = np.append(xArr, p[s][0][2])
        yArr = np.append(yArr, p[s][1][2])
    x = xArr
    y = yArr
    ma = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[ma]
    y = y[ma]
    mask = np.sign(x) == np.sign(y)
    popt, pcov = curve_fit(power_law, np.abs(x[mask]), np.abs(y[mask]), maxfev=100000)
    a, b = popt
    print("a: {}, b: {}".format(a, b))
    new_x = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    axes.plot(new_x, power_law(new_x, a, b))
    axes.plot(-new_x, -power_law(new_x, a, b))
    return x, y

def hist(x, y):
    """Calculates a 2D histogram and plots it."""
    H, xedges, yedges = np.histogram2d(x, y, bins=100)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)
    plt.show()


def scatter_plot(dict1, dict2, separate=False):
    """Plots different blocked n values. Deprecated."""
    i = 1
    for n in sorted([x for x in dict1.keys() if x != 'instr']):
        if separate:
            plt.subplot(len(dict1), 1, i)
            plt.xlabel(dict1['instr'])
            plt.ylabel(dict2['instr'])
            plt.title('Number of blocks: {}'.format(n))
        plt.plot([np.nanmin(dict1[n]),np.nanmax(dict1[n])],[np.nanmin(dict1[n]),np.nanmax(dict1[n])], 'k-', lw=2)
        plt.plot(dict1[n], dict2[n], '.')
        i += 1
    plt.show()

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

    for p in args:
        p1, p2 = p
        ar_i1 = p1[0]
        ar_i2 = p2[0]
        ind = np.where(np.abs((ar_i1 - ar_i2)/ar_i1) < .10)
        c_valid = p1[3][ind]
        sortedInds = np.argsort(c_valid)
        ar_i1 = ar_i1[ind][sortedInds]
        ar_i2 = ar_i2[ind][sortedInds]
        f_i1 = p1[2][ind][sortedInds]
        f_i2 = p2[2][ind][sortedInds]

        f1_valid = (np.abs(f_i1) > 1)
        f2_valid = (np.abs(f_i2) > 1)
        s = f1_valid & f2_valid
        #Area plot
        ax[0].scatter(ar_i1, ar_i2, cmap=co, c=c_valid,
                vmin=0, vmax=90, edgecolors='face', zorder=2)

        #Mean Field plot
        ax[1].scatter(f_i1, f_i2, cmap=co, c=c_valid,
                vmin=0, vmax=90, edgecolors='face', zorder=2)
        ax[1].set_ylim(-400, 400)
        ax[1].set_ylim(-400, 400)
    
        #Mean Field plot log scale
        x1 = np.log10(np.abs(f_i1[s]))*np.sign(f_i1[s])
        y1 = np.log10(np.abs(f_i2[s]))*np.sign(f_i2[s])
        plots = ax[2].scatter(x1, y1, cmap=co, c=c_valid[s],
                vmin=0, vmax=90, edgecolors='face', zorder=2)

    add_identity(ax[0], color='.3', ls='--', zorder=1)
    add_identity(ax[1], color='.3', ls='--', zorder=1)
    add_identity(ax[2], color='.3', ls='--', zorder=1)

    ax[0].set_title('Block Areas')
    ax[1].set_title('Mean Field')
    ax[0].set(aspect='equal', xlabel=r'Area (cm^2)', ylabel=r'Area (cm^2)')
    ax[1].set(aspect='equal', xlabel=r'B Field (G)', ylabel=r'B Field (G)')
    ax[1].set_xlim(ax[1].get_ylim())
    #fit_mean_field(ax[1], args)
    ax[2].set_title('Logarithm Space of Fields')
    ax[2].set(aspect='equal', xlabel=r'Logarithmic B Field (log(B))',
            ylabel=r'Logarithmic B Field (log(B))')
    ax[2].set_ylim(-3, 3)
    ax[2].set_xlim(ax[2].get_ylim())
    
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
            title = "{}: {}".format(args[i][0].mgnt.im_raw.instrument, 
                    args[i][0].mgnt.im_raw.date.isoformat())
            ax[i].set_title(title)
        else:
            im[i] = args[i]
            ax[i] = plt.subplot2grid((rows, cols), (i%rows, i//rows))
            ax[i].imshow(im[i], cmap='binary')

    return ax