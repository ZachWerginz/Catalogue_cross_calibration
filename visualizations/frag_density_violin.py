"""This script was meant to creates visualization plots for fragmentation.

In order to show fragmentation of the solar disk and preliminary results, we plot three axes on a figure. One contains
the solar disk with overlaid colored blocks as a visual aid to show how the solar disk is being fragmented. The other
two panels contain a two dimensional histogram as well as violin plots.

"""

import random

import mpl_scatter_density
import matplotlib.pyplot as plt
import numpy as np

import cross_calibration as c
import quadrangles_keep_inds as temp_q
import util as u
import visualizations.cc_plot as ccplot


def data(raw_remap=False):
    """Load sample data into memory and prepare it for fragmentation.

    Args:
        raw_remap (bool): defaults to False, uses raw fiels instead of corrected

    Returns:
        m1: reference magnetogram CRD object
        m2: secondary magnetogram CRD object

    """
    f1 = "test_mgnts/spmg_eo100_C1_19920424_1430.fits"
    f2 = "test_mgnts/spmg_eo100_C1_19920425_1540.fits"

    m1, m2 = c.prepare_magnetograms(f1, f2, raw_remap)
    return m1, m2


def block_plot(m1, blocks, ax1):
    """Given a list of blocks, will plot a nice image differentiating them.

    Args:
        m1 (obj): CRD object who's image data will be shown
        blocks (list): list of quadrangles to be plotted as blocks
        ax1 (obj): matplotlib axis object to plot on

    """
    im1 = m1.lonh.v.copy()
    im1[:] = np.nan
    for x in blocks:
        r = random.random()
        im1[x.indices] = r
    ax1.imshow(m1.im_raw.data, cmap='binary', vmin=-700, vmax=700, zorder=1)
    ax1.imshow(im1, vmin=0, vmax=1, alpha=.4, cmap='jet', zorder=2)


def plot_row(f, m1, blocks, results, hist_lim):
    """Plots three axes on a figure - the fragmentation, a hist2d plot, and a violin plot.

    Args:
        f (obj): figure to plot on
        m1 (obj): CRD object who's image data will be shown
        blocks (list): list of quadrangles to show fragmentation
        results (dict): points to provide for the hist2d
        hist_lim (float): axis limits for 2D histogram
    """
    f.subplots_adjust(left=.1, right=.9, wspace=0)
    ax1 = f.add_subplot(131, projection=m1.im_raw.wcs)
    ax2 = f.add_subplot(132, projection='scatter_density')
    ax3 = f.add_subplot(133)

    color = (81 / 255, 178 / 255, 76 / 255)

    block_plot(m1, blocks, ax1)
    # hb = ccplot.hist2d(results['reference_fd'], results['secondary_fd'], ax2, noise=0, lim=hist_lim)
    plot = ccplot.scatter_density(results['reference_fd'], results['secondary_fd'], ax2, lim=hist_lim)
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
    """Fragments two magnetograms at a level of 25, 50, and 100 and shows example plots of summary data."""
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
    fig1 = plt.figure(figsize=(16, 16 / 3))
    plot_row(fig1, m1, blocks_25, r_25, 250)
    ax1, ax2, ax3 = fig1.get_axes()
    ax1.set_ylabel(r'$\mathrm{{n = {0}}}$'.format('25'))
    ax2.set_yticks([-200, 0, 200])
    ax2.set_xticks([-200, 0, 200])
    ax2.set_yticklabels([r'$-200$', r'$0$', r'$200$'], ha='right')
    ax3.set_yticks([-50, 0, 50])
    ax3.set_yticklabels([r'$-50$', r'$0$', r'$50$'], ha='right')

    #  --------------------Plot first set of panels of n = 50------------------
    fig2 = plt.figure(figsize=(16, 16 / 3))
    plot_row(fig2, m1, blocks_50, r_50, 650)
    ax1, ax2, ax3 = fig2.get_axes()
    ax1.set_ylabel(r'$\mathrm{{n = {0}}}$'.format('50'))
    ax2.set_xticks([-600, 0, 600])
    ax2.set_yticks([-600, 0, 600])
    ax2.set_yticklabels([r'$-600$', r'$0$', r'$600$'], ha='right')
    ax3.set_yticks([-200, 0, 200])
    ax3.set_yticklabels([r'$-200$', r'$0$', r'$200$'], ha='right')

    #  --------------------Plot first set of panels of n = 100------------------
    fig3 = plt.figure(figsize=(16, 16 / 3))
    plot_row(fig3, m1, blocks_100, r_100, 1600)
    ax1, ax2, ax3 = fig3.get_axes()
    ax1.set_ylabel(r'$\mathrm{{n = {0}}}$'.format('100'))
    ax2.set_yticks([-1500, 0, 1500])
    ax2.set_xticks([-1500, 0, 1500])
    ax2.set_yticklabels([r'$-1500$', r'$0$', r'$1500$'], ha='right')
    ax3.set_yticks([-500, 0, 500])
    ax3.set_yticklabels([r'$-500$', r'$0$', r'$500$'], ha='right')

    plt.draw()
    plt.show()

    axes = []
    for f in [fig1, fig2, fig3]:
        axes.extend(f.get_axes())

    for ax, letter in zip(axes, 'abcdefghi'):
        if letter in 'beh':
            color = 'white'
        else:
            color = 'black'
        ax.annotate(
                '{0}'.format(letter),
                xy=(0, 1), xycoords='axes fraction',
                xytext=(7, -25), textcoords='offset points',
                ha='left', va='bottom', fontsize=19, color=color, family='serif')


if __name__ == '__main__':
    main()
