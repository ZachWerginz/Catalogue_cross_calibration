import cross_calibration as c
import quadrangles_keep_inds as temp_q
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors
import copy
import random
import numpy as np
import util as u
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

def data(raw_remap=False):
    f1 = "test_mgnts/spmg_eo100_C1_19920424_1430.fits"
    f2 = "test_mgnts/spmg_eo100_C1_19920425_1540.fits"

    if raw_remap:
        m1, m2 = c.fix_longitude(f1, f2, raw_remap=True)
    else:
        m1, m2 = c.fix_longitude(f1, f2)
    return m1, m2


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


def plot_axis_base(f, firstImage, secondImage):
    ax1 = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)

    ax1.imshow(firstImage, cmap='binary', vmin=-100, vmax=100)
    ax2.imshow(secondImage, cmap='bwr', vmin=-100, vmax=100)
    ax3.imshow(firstImage, cmap='binary', vmin=-100, vmax=100)
    # norm = mpl.colors.Normalize(vmin=-700, vmax=700)
    # scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
    # rgb = scalarMap.to_rgba(secondImage)
    # alpha_m2 = np.abs(secondImage)
    # alpha_m2[alpha_m2 > 400] = 400
    # alpha_m2 = (alpha_m2/400)**.33
    # rgb[:, :, 3] = alpha_m2
    ax3.imshow(secondImage, cmap='bwr', vmin=-100, vmax=100, alpha=.7)


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
    #f1 = plt.figure(1)
    #plot_axis(f1, m1.im_raw.data, m2.im_raw.data, m1.im_raw.wcs, sun_orig)
    #f1.suptitle("Raw Data Before Rotation", y=.785, fontsize=30, fontweight='bold')


    # --------------Radial Correction Before Rot/Interp-------------
    f2 = plt.figure(2)
    plot_axis(f2, m1.im_corr.v, m2.im_corr.v, m1.im_raw.wcs, sun_orig)
    f2.text(.235, .745, "SPMG {0}".format(m1.im_raw.date), fontsize=21, **axis_font)
    f2.text(.5, .745, "SPMG {0}".format(m2.im_raw.date), fontsize=21, **axis_font)
    f2.suptitle("Radially Corrected Data Before Rotation", y=.79, fontsize=30, fontweight='bold')

    # --------------Radial Correction After Rot/Interp-------------
    f3 = plt.figure(3)
    plot_axis(f3, m1.im_corr.v, m2.remap, m1.im_raw.wcs, sun_orig)
    f3.suptitle("Radially Corrected Data After Rotation and Interpolation", y=.77, fontsize=30, fontweight='bold')

    axes = []
    for f in [f2, f3]:
        axes.extend(f.get_axes())

    for ax, letter in zip(axes, 'abcdefghijkl'):
        ax.annotate(
                '{0}'.format(letter),
                xy=(0,1), xycoords='axes fraction',
                xytext=(7, -25), textcoords='offset points',
                ha='left', va='bottom', fontsize=19, color='white', family='serif')

    plt.draw()
    plt.show()

if __name__ == '__main__':
    main()
