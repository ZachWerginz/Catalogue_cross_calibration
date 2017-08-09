import cross_calibration as c
import quadrangles_keep_inds as temp_q
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors
import copy
import random
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter


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


def data(raw_remap=False):
    f1 = "test_mgnts/spmg_eo100_C1_19920424_1430.fits"
    f2 = "test_mgnts/spmg_eo100_C1_19920425_1540.fits"

    if raw_remap:
        m1, m2 = c.fix_longitude(f1, f2, raw_remap=True)
    else:
        m1, m2 = c.fix_longitude(f1, f2)
    return m1, m2


def block_plot(m1, blocks, ax1):
    """Given a list of blocks, will plot a nice image differentiating them."""
    im1 = m1.lonh.v.copy()
    im1[:] = np.nan
    for x in blocks:
        r = random.random()
        im1[x.indices] = r
    ax1.imshow(m1.im_raw.data, cmap='binary', vmin=-700, vmax=700, zorder=1)
    ax1.imshow(im1, vmin=0, vmax=1, alpha=.4, cmap='jet', zorder=2)


def plot_axis(f, firstImage, secondImage, wcs, orig):
    ax1 = f.add_subplot(131, projection=wcs)
    ax2 = f.add_subplot(132, projection=wcs)
    ax3 = f.add_subplot(133, projection=wcs)

    for ax in (ax1, ax2, ax3):
        ax.imshow(orig, cmap=colors.ListedColormap([np.array([90/255, 90/255, 106/255])]))

    ax1.imshow(firstImage, cmap='binary', vmin=-700, vmax=700)
    ax2.imshow(secondImage, cmap='bwr', vmin=-700, vmax=700)
    ax3.imshow(firstImage, cmap='binary', vmin=-700, vmax=700)
    norm = mpl.colors.Normalize(vmin=-700, vmax=700)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
    rgb = scalarMap.to_rgba(secondImage)
    alpha_m2 = np.abs(secondImage)
    alpha_m2[alpha_m2 > 400] = 400
    alpha_m2 = (alpha_m2/400)**.33
    rgb[:, :, 3] = alpha_m2
    ax3.imshow(rgb)

    f.subplots_adjust(left=.1, right=.9, wspace=0)
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax3.set_facecolor('black')
    for ax in (ax1, ax2, ax3):
        for co in ax.coords:
            co.set_ticklabel_visible(False)
            co.set_ticks_visible(False)

    plt.draw()


def main(m1=None, m2=None):
    if m1 is None or m2 is None:
        m1, m2 = data()
    axis_font = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
    plt.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [
        r'\usepackage{cmbright}']
    plt.ion()

    m2max = max(np.nanmin(m2.im_raw.data), np.nanmax(m2.im_raw.data))
    m1max = max(np.nanmin(m1.im_raw.data), np.nanmax(m1.im_raw.data))

    sun_orig = copy.deepcopy(m1.im_raw.data)
    sun_orig[m1.rg < m1.par['rsun']] = 500000

    # ------------------Raw Before Rot/Interp-------------
    f1 = plt.figure(1) 
    plot_axis(f1, m1.im_raw.data, m2.im_raw.data, m1.im_raw.wcs, sun_orig)
    f1.suptitle("Raw Data Before Rotation", y=.785, fontsize=30, fontweight='bold')
    f1.text(.235, .74, "SPMG {0}".format(m1.im_raw.date), fontsize=21, **axis_font)
    f1.text(.5, .74, "SPMG {0}".format(m2.im_raw.date), fontsize=21, **axis_font)

    # --------------Radial Correction Before Rot/Interp-------------
    f2 = plt.figure(2)
    plot_axis(f2, m1.im_corr.v, m2.im_corr.v, m1.im_raw.wcs, sun_orig)
    f2.suptitle("Radially Corrected Data Before Rotation", y=.76, fontsize=30, fontweight='bold')

    # --------------Radial Correction After Rot/Interp-------------
    f3 = plt.figure(3)
    plot_axis(f3, m1.im_corr.v, m2.remap, m1.im_raw.wcs, sun_orig)
    f3.suptitle("Radially Corrected Data After Rotation and Interpolation", y=.76, fontsize=30, fontweight='bold')

    blocks_n = temp_q.fragment_multiple(m1, m2, 50)

    # ----------Plot last segmented panels because its special----
    f4 = plt.figure(4)
    f4.subplots_adjust(left=.1, right=.9, wspace=0)
    ax1 = f4.add_subplot(131, projection=m1.im_raw.wcs)
    ax2 = f4.add_subplot(132)
    ax3 = f4.add_subplot(133)

    ax1.imshow(sun_orig, cmap=colors.ListedColormap([np.array([90/255, 90/255, 106/255])]))
    block_plot(m1, blocks_n, ax1)
    x_data = np.array([block.fluxDensity for block in blocks_n])
    y_data = np.array([block.fluxDensity2 for block in blocks_n])
    da = np.array([block.diskAngle.v for block in blocks_n])
    ax2.scatter(x_data, y_data, cmap='viridis', c=da, vmin=0, vmax=90, edgecolors='face', zorder=10)

    # ----------------------Box plot setup -------------------------

    disk_limits = [0, 90]
    minimum = np.nanmin(x_data)
    maximum = np.nanmax(x_data)
    ind = np.where(np.logical_and(np.logical_and(np.isfinite(x_data), np.isfinite(y_data)),
                                  np.logical_and(da > disk_limits[0], da < disk_limits[1])))
    x = x_data[ind]
    y = y_data[ind]
    hist_list = []
    xedges = np.linspace(minimum, maximum, 100)
    for i in range(1, xedges.shape[0] - 1):
        inds = np.where(np.logical_and(x < xedges[i], x >= xedges[i - 1]))
        slice_med = (xedges[i] + xedges[i - 1]) / 2
        med = np.median(y[inds])
        mea = np.mean(y[inds])
        std = np.std(y[inds])
        hist_list.append({'med': med, 'mea': mea, 'std': std, 'data': y[inds], 'b': xedges, 'sliceMed': slice_med})

    y_bins = np.array([x['med'] for x in hist_list])
    x_bins = np.array([s['sliceMed'] for s in hist_list])
    lim = max(np.max(np.abs(x_bins)), np.max(np.abs(y_bins))) * 1.1
    box_list = np.array([s['data'] for s in hist_list])
    # lim = max(abs(x[0]*1.10), abs(x[-1]*1.10))
    box_widths = .4 * (max(x_bins) - min(x_bins)) / len(y_bins)
    box = ax3.boxplot(box_list, widths=box_widths, positions=x_bins, manage_xticks=False,
                     whis=0, sym="", showcaps=False, patch_artist=True)

    clr = (81/255, 178/255, 76/255)
    for bx in box['boxes']:
        bx.set(edgecolor=clr, facecolor=clr, alpha=.75)

    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set(aspect='equal')
    add_identity(ax3, color='.3', ls='-', linewidth=2, zorder=1)

    ax2.set(aspect='equal')
    ax2.axis('square')
    ax1.set_facecolor('black')
    ax3.yaxis.set_ticks_position('right')
    ax3.yaxis.set_label_position('right')
    # ax3.tick_params(axis='both', pad=0)

    def math_formatter(x, pos):
        return "%i" % x

    ax2.xaxis.set_major_formatter(FuncFormatter(math_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(math_formatter))
    ax3.xaxis.set_major_formatter(FuncFormatter(math_formatter))
    ax3.yaxis.set_major_formatter(FuncFormatter(math_formatter))
    ax2.set_xlabel(r'$\mathrm{{Magnetic\ Flux\ Density\ (Mx/cm^2)}}$', labelpad=20, size=23, **axis_font)
    ax3.set_ylabel(r'$\mathrm{{Magnetic\ Flux\ Density\ (Mx/cm^2)}}$', labelpad=20, rotation=270, size=23, **axis_font)
    ax2.tick_params(width=5, labelsize=25)
    ax3.tick_params(width=5, labelsize=25)
    maxlimax2 = max(*ax2.get_xlim(), *ax2.get_ylim())
    ax2.set_xlim(-maxlimax2, maxlimax2)
    ax2.set_ylim(-maxlimax2, maxlimax2)
    for co in ax1.coords:
        co.set_ticklabel_visible(False)
        co.set_ticks_visible(False)

    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=7, prune='both'))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='both'))
    add_identity(ax2, color='.3', ls='-', linewidth=3.0, zorder=1)
    add_identity(ax3, color='.3', ls='-', linewidth=3.0, zorder=1)
    f4.suptitle("Fragmentation with n=50 and Flux Scatter Plot", y=.77, fontsize=30, fontweight='bold')

    axesList = []
    for f in [f1, f2, f3]:
        axesList.extend(f.get_axes())

    for ax, letter in zip(axesList, 'abcdefghijkl'):
        ax.annotate(
                '{0}'.format(letter),
                xy=(0,1), xycoords='axes fraction',
                xytext=(7, -25), textcoords='offset points',
                ha='left', va='bottom', fontsize=19, color='white', family='serif')

    for ax, letter in zip(f4.get_axes(), 'abc'):
        if letter == 'c' or letter == 'b':
            color = 'black'
        else:
            color = 'white'
        ax.annotate(
                '{0}'.format(letter),
                xy=(0,1), xycoords='axes fraction',
                xytext=(7, -25), textcoords='offset points',
                ha='left', va='bottom', fontsize=19, color=color, family='serif')

    plt.draw()

    plt.show()

if __name__ =='__main__':
    main()