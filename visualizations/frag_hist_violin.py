import cross_calibration as c
import quadrangles_keep_inds as temp_q
import matplotlib.pyplot as plt
import visualizations.cc_plot as ccplot
import copy
import random
import numpy as np
import util as u

def data(raw_remap=False):
    f1 = "test_mgnts/spmg_eo100_C1_19920424_1430.fits"
    f2 = "test_mgnts/spmg_eo100_C1_19920425_1540.fits"

    if raw_remap:
        m1, m2 = c.fix_longitude(f1, f2, raw_remap=True)
    else:
        m1, m2 = c.fix_longitude(f1, f2)
    return m1, m2


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


def block_plot(m1, blocks, ax1):
    """Given a list of blocks, will plot a nice image differentiating them."""
    im1 = m1.lonh.v.copy()
    im1[:] = np.nan
    for x in blocks:
        r = random.random()
        im1[x.indices] = r
    ax1.imshow(m1.im_raw.data, cmap='binary', vmin=-700, vmax=700, zorder=1)
    ax1.imshow(im1, vmin=0, vmax=1, alpha=.4, cmap='jet', zorder=2)


def hexbin(points, ax):
    y, x, da = temp_q.extract_valid_points(points)

    co = 'inferno'
    plt.rc('text', usetex=True)
    i1 = points['i1']
    i2 = points['i2']

    sorted_inds = np.argsort(da)
    f_i1 = y[sorted_inds]
    f_i2 = x[sorted_inds]
    da = da[sorted_inds]

    xmin = np.nanmin(f_i2)
    xmax = np.nanmax(f_i2)
    ymin = np.nanmin(f_i1)
    ymax = np.nanmax(f_i1)
    lim = min(abs(xmin), abs(xmax), abs(ymin), abs(ymax))

    hb = ax.hexbin(f_i2, f_i1, cmap=co, bins='log', gridsize=100, zorder=1)

    # ------------- Set Plot Properties ----------------------------
    add_identity(ax, color='.3', ls='-', zorder=1)
    ax.axis([-lim, lim, -lim, lim])
    ax.set(adjustable='box-forced', aspect='equal')

    return hb


def plot_row(f, m1, blocks, results):
    f.subplots_adjust(left=.1, right=.9, wspace=0)
    ax1 = f.add_subplot(131, projection=m1.im_raw.wcs)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)

    color = (81 / 255, 178 / 255, 76 / 255)

    block_plot(m1, blocks, ax1)
    hb = hexbin(results, ax2)
    ccplot.violin_plot(results, [0, 70], ax3, alpha=.4, percentiles=[0, 100], clr=color, corrections=False)
    ccplot.violin_plot(results, [0, 70], ax3, alpha=.5, percentiles=[25, 75], clr=color, corrections=False)
    ccplot.violin_plot(results, [0, 70], ax3, alpha=.75, percentiles=[37.5, 62.5], clr=color, corrections=False)

    for co in ax1.coords:
        co.set_ticklabel_visible(False)
        co.set_ticks_visible(False)

    ax2.tick_params(axis='y', direction='in', left='off', right='on', labelleft='off', labelright='on', colors='white',
                    pad=-10)
    ax2.tick_params(axis='x', direction='in', colors='white', pad=-25)
    ax3.tick_params(axis='both', direction='in', left='off', right='on', labelleft='off', labelright='on', pad=-10)
    ax3.tick_params(axis='x', direction='in', pad=-25)


def main():
    plt.ion()
    axis_font = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
    plt.rc('text', usetex=True)
    m1, m2 = data()

    blocks_25 = temp_q.fragment_multiple(m1, m2, 25)
    blocks_50 = temp_q.fragment_multiple(m1, m2, 50)
    blocks_100 = temp_q.fragment_multiple(m1, m2, 100)

    r_25 = u.download_cc_data('spmg', 'spmg', 25, '23 hours', '25 hours')
    r_50 = u.download_cc_data('spmg', 'spmg', 50, '23 hours', '25 hours')
    r_100 = u.download_cc_data('spmg', 'spmg', 100, '23 hours', '25 hours')
    #  --------------------Plot first set of panels of n = 25------------------
    f1 = plt.figure(1)
    plot_row(f1, m1, blocks_25, r_25)
    ax1, ax2, ax3 = f1.get_axes()
    ax1.set_ylabel(r'$\mathrm{{n = {0}}}$'.format('25'))
    ax2.set_yticks([-200, 0, 200])
    ax2.set_yticklabels([r'$-200$', r'$0$', r'$200$'], ha='right')
    ax3.set_yticks([-50, 0, 50])
    ax3.set_yticklabels([r'$-50$', r'$0$', r'$50$'], ha='right')

    #  --------------------Plot first set of panels of n = 25------------------
    f2 = plt.figure(2)
    plot_row(f2, m1, blocks_50, r_50)
    ax1, ax2, ax3 = f2.get_axes()
    ax1.set_ylabel(r'$\mathrm{{n = {0}}}$'.format('50'))
    ax2.set_xticks([-600, 0, 600])
    ax2.set_yticks([-600, 0, 600])
    ax2.set_yticklabels([r'$-600$', r'$0$', r'$600$'], ha='right')
    ax3.set_yticks([-200, 0, 200])
    ax3.set_yticklabels([r'$-200$', r'$0$', r'$200$'], ha='right')

    #  --------------------Plot first set of panels of n = 25------------------
    f3 = plt.figure(3)
    plot_row(f3, m1, blocks_100, r_100)
    ax1, ax2, ax3 = f3.get_axes()
    ax1.set_ylabel(r'$\mathrm{{n = {0}}}$'.format('100'))
    ax2.set_yticks([-1500, 0, 1500])
    ax2.set_xticks([-1500, 0, 1500])
    ax2.set_yticklabels([r'$-1500$', r'$0$', r'$1500$'], ha='right')
    ax3.set_yticks([-500, 0, 500])
    ax3.set_yticklabels([r'$-500$', r'$0$', r'$500$'], ha='right')

    # ax2.set_xlabel(r'$\mathrm{{Magnetic\ Flux\ Density\ (Mx/cm^2)}}$', labelpad=20, size=23, **axis_font)
    # ax3.set_ylabel(r'$\mathrm{{Magnetic\ Flux\ Density\ (Mx/cm^2)}}$', labelpad=20, rotation=270, size=23, **axis_font)
    # f4.suptitle("Fragmentation with n=50 and Flux Scatter Plot", y=.77, fontsize=30, fontweight='bold')

    plt.draw()
    plt.show()

    axes = []
    for f in [f1, f2, f3]:
        axes.extend(f.get_axes())

    for ax, letter in zip(axes, 'abcdefghi'):
        if letter in 'beh':
            color = 'white'
        else:
            color = 'black'
        ax.annotate(
                '{0}'.format(letter),
                xy=(0,1), xycoords='axes fraction',
                xytext=(7, -25), textcoords='offset points',
                ha='left', va='bottom', fontsize=19, color=color, family='serif')


if __name__ == '__main__':
    main()