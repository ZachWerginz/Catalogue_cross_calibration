import cross_calibration as c
from scipy.interpolate import griddata
import quadrangles_keep_inds as temp_q
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors
from zaw_coord import CRD
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
    f1 = "spmg_eo100_C1_19920424_1430.fits"
    f2 = "spmg_eo100_C1_19920425_1540.fits"

    if raw_remap:
        m1, m2 = c.fix_longitude(f1, f2, raw_remap=True)
    else:
        m1, m2 = c.fix_longitude(f1, f2)
    return m1, m2

def block_plot(m1, m2, blocks, ax1, ax2):
    """Given a list of blocks, will plot a nice image differentiating them."""
    im1 = m1.lonh.v.copy()
    im2 = m2.lonh.v.copy()
    im1[:] = np.nan
    im2[:] = np.nan
    for x in blocks:
        r = random.random()
        im1[x.indices] = r
        im2[x.indices] = r
    ax1.imshow(m1.im_raw.data, cmap='binary', vmin=-700, vmax=700, zorder=1)
    ax2.imshow(m2.remap, cmap='binary', vmin=-700, vmax=700, zorder=1)
    ax1.imshow(im1, vmin=0, vmax=1, alpha=.4, cmap='jet', zorder=2)
    ax2.imshow(im2, vmin=0, vmax=1, alpha=.4, cmap='jet', zorder=2)

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
    rgb[:,:,3] = alpha_m2
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
    axis_font = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
    plt.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [
        r'\usepackage{cmbright}']
    plt.ion()

    m2max = max(np.nanmin(m2.im_raw.data), np.nanmax(m2.im_raw.data))
    m1max = max(np.nanmin(m1.im_raw.data), np.nanmax(m1.im_raw.data))

    sun_orig = copy.deepcopy(m1.im_raw.data)
    sun_orig[m1.rg < m1.rsun] = 500000

    #------------------Raw Before Rot/Interp-------------
    f1 = plt.figure(1) 
    plot_axis(f1, m1.im_raw.data, m2.im_raw.data, m1.im_raw.wcs, sun_orig)
    f1.suptitle("Raw Data Before Rotation", y=.785, fontsize=30, fontweight='bold')
    f1.text(.235, .74, "SPMG {0}".format(m1.im_raw.date), fontsize=21, **axis_font)
    f1.text(.5, .74, "SPMG {0}".format(m2.im_raw.date), fontsize=21, **axis_font)

    #--------------Radial Correction Before Rot/Interp-------------
    f2 = plt.figure(2)
    plot_axis(f2, m1.im_corr.v, m2.im_corr.v, m1.im_raw.wcs, sun_orig)
    f2.suptitle("Radially Corrected Data Before Rotation", y=.76, fontsize=30, fontweight='bold')

    #--------------Radial Correction After Rot/Interp-------------
    f3 = plt.figure(3)
    plot_axis(f3, m1.im_corr.v, m2.remap, m1.im_raw.wcs, sun_orig)
    f3.suptitle("Radially Corrected Data After Rotation and Interpolation", y=.76, fontsize=30, fontweight='bold')

    blocks_n = temp_q.fragment_multiple(m1, m2, 25)

    #----------Plot last segmented panels because its special----
    f4= plt.figure(4)
    ax1 = f4.add_subplot(131, projection=m1.im_raw.wcs)
    ax2 = f4.add_subplot(132, projection=m1.im_raw.wcs)
    ax3 = f4.add_subplot(133)
    for ax in (ax1, ax2):
        ax.imshow(sun_orig, cmap=colors.ListedColormap([np.array([90/255, 90/255, 106/255])]))
    block_plot(m1, m2, blocks_n, ax1, ax2)
    xData = [block.fluxDensity for block in blocks_n]
    yData = [block.fluxDensity2 for block in blocks_n]
    da = [block.diskAngle.v for block in blocks_n]
    ax3.scatter(xData, yData, cmap='viridis', c=da, vmin=0, vmax=90, edgecolors='face', zorder=10)
    ax3.set(aspect='equal')
    ax3.axis('square')
    f4.subplots_adjust(left=.1, right=.9, wspace=0)
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax3.yaxis.set_ticks_position('right')
    ax3.yaxis.set_label_position('right')
    #ax3.tick_params(axis='both', pad=0)
    def math_formatter(x, pos):
        return "%i" %x
    ax3.xaxis.set_major_formatter(FuncFormatter(math_formatter))
    ax3.yaxis.set_major_formatter(FuncFormatter(math_formatter))
    ax3.set_xlabel(r'$\mathrm{{Magnetic\ Flux\ Density\ (Mx/cm^2)}}$', labelpad=0)
    ax3.set_ylabel(r'$\mathrm{{Magnetic\ Flux\ Density\ (Mx/cm^2)}}$', labelpad=20, rotation=270)
    maxlimAx3 = max(*ax3.get_xlim(), *ax3.get_ylim())
    ax3.set_xlim(-maxlimAx3, maxlimAx3)
    ax3.set_ylim(-maxlimAx3, maxlimAx3)

    for ax in (ax1, ax2):
        for co in ax.coords:
            co.set_ticklabel_visible(False)
            co.set_ticks_visible(False)

    ax3.xaxis.set_major_locator(MaxNLocator(nbins=7, prune='both'))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='both'))
    add_identity(ax3, color='.3', ls='-', linewidth=3.0, zorder=1)
    f4.suptitle("Fragmentation with n=25 and Flux Scatter Plot", y=.76, fontsize=30, fontweight='bold')

    axesList = []
    for f in [f1,f2,f3,f4]:
        axesList.extend(f.get_axes())

    for ax, letter in zip(axesList, 'abcdefghijklmno'):
        if letter=='o':
            c = 'black'
        else:
            c = 'white'
        ax.annotate(
                '{0}'.format(letter),
                xy=(0,1), xycoords='axes fraction',
                xytext=(7, -25), textcoords='offset points',
                ha='left', va='bottom', fontsize=19, color=c, family='serif')

    plt.draw()

    plt.show()

if __name__ == '__main__':
    main()