"""This script takes magnetograms and compares them pixel for pixel.

This also includes funcitonality for finding optimum rotation among magnetograms using the golden search method. It was
determined that finding the optimal rotation among two magnetograms produces minimal benefits.

"""

import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import cross_calibration as c
import visualizations.cc_plot as ccplot


def get_pcoeff(increment, m1=None, file2=None, cond=None):
    """Originally used to find pearson correlation coefficient but not used in practice."""
    print('Increment: {}'.format(increment))
    m2 = CRD(file2, rotate=increment)
    m2.magnetic_flux()
    rotation = u.diff_rot(m1, m2)
    m2.lonhRot = m2.lonh + rotation.value
    c.interpolate_remap(m1, m2)
    y = m1.im_corr.v.ravel()
    x = m2.remap.ravel()
    ind = (np.abs(x) > cond) * (np.abs(y) > cond) * np.isfinite(x) * np.isfinite(y)
    corr_coeff = sp.stats.pearsonr(x[ind], y[ind])
    print('Pearson Coefficient: {}'.format(corr_coeff[0]))
    return -corr_coeff[0]


def rotate_p(file1, file2, instr='hmi'):
    """Originally used to find optimal rotation but not used in practice."""
    if instr == 'mdi':
        cond = 26
    else:
        cond = 20

    m1 = CRD(file1)
    m1.magnetic_flux()
    y = m1.im_corr.v.ravel()
    best_p0_diff = 0
    best_corr = 0
    best_p0_diff = sciop.minimize_scalar(get_pcoeff, method='golden', tol=.05,
                                         args=(m1, file2, cond), bracket=(-1, -.4, 1))

    print('Best p_angle shift: {}'.format(best_p0_diff))

    m2 = CRD(file2, rotate=best_p0_diff.x)
    m2.magnetic_flux()
    rotation = u.diff_rot(m1, m2)
    m2.lonhRot = m2.lonh + rotation.value
    c.interpolate_remap(m1, m2)
    x = m2.remap.ravel()
    ind = (np.abs(x) > cond) * (np.abs(y) > cond) * np.isfinite(x) * np.isfinite(y)
    edges = np.arange(-987.5, 987.5, 25)
    f = plt.figure(1)
    ax = f.add_subplot(111)
    ax.hist2d(x[ind], y[ind], cmap='inferno', norm=colors.LogNorm(), bins=edges)
    ax.set_facecolor('black')
    ax.set(adjustable='box-forced', aspect='equal')

    return best_p0_diff


def create_variable_time_plot(instr):
    """Input an instrument and plot sample data at different time scales pixel for pixel.

    This will compare two magnetograms in time pixel for pixel with a 2D histogram shown on a log scale.

    Args:
        instr (str): the instrument to plot

    Returns:
        object: a tuple of the figure and axis grid

    """
    if instr == 'mdi':
        files = glob.glob('test_mgnts/fd*')
        f, grid = plt.subplots(1, 3, figsize=(16, 16 / 3), sharex='col', sharey='row',
                               gridspec_kw={'wspace': 0, 'hspace': 0},
                               subplot_kw={'projection': 'scatter_density'})
        grid[0].set_ylabel('MDI', fontsize=30)
        cond = 26
        times = ['1.6 hrs', '6.4 hrs', '24 hrs']
    elif instr == 'hmi':
        files = glob.glob('test_mgnts/hmi*')
        f, grid = plt.subplots(2, 3, figsize=(16, 16/1.5), sharex='col', sharey='row',
                               gridspec_kw={'wspace': 0, 'hspace': 0},
                               subplot_kw={'projection': 'scatter_density'})
        f.text(0.07, 0.5, 'HMI', ha='center', va='center', rotation='vertical', fontsize=30)
        cond = 20
        times = ['12 min', '24 min', '48 min', '1.6 hrs', '6.4 hrs', '24 hrs']
    elif instr == 'sim':
        files = glob.glob('test_mgnts/*.dat')
        cond = 20
        times = ['2 hrs', '6 hrs', '24 hrs']
        f, grid = plt.subplots(1, 3, figsize=(16, 16 / 3), sharex='col', sharey='row',
                               gridspec_kw={'wspace': 0, 'hspace': 0},
                               subplot_kw={'projection': 'scatter_density'})
        grid[0].set_ylabel('AFT Simulation', fontsize=30)
    lim = 1000

    for file, ax in zip(files[1:], grid.flatten()):
        if instr == 'sim':
            m1, m2 = c.prepare_simulation(files[0], file)
        else:
            m1, m2 = c.prepare_magnetograms(files[0], file)
        x = m2.remap.ravel()
        y = m1.im_corr.v.ravel()
        ccplot.scatter_density(x, y, ax, lim=lim, null_cond=cond, log_vmax=200)
        ccplot.add_identity(ax, color='.5', ls='-', alpha=.5, linewidth=2, zorder=1)

    for ax, letter in zip(f.get_axes(), times):
        ax.annotate(
            '{0}'.format(letter),
            xy=(0, 1), xycoords='axes fraction',
            xytext=(7, -25), textcoords='offset points',
            ha='left', va='bottom', fontsize=19, color='white', family='serif')

    for ax in grid.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.xaxis.set_tick_params(labelsize=30)
        ax.yaxis.set_tick_params(labelsize=30)

    if instr == 'mdi':
        f.subplots_adjust(top=0.982, bottom=0.075, left=0.1, right=0.977, hspace=0.2, wspace=0.0)
    else:
        f.subplots_adjust(top=0.969, bottom=0.085, left=0.1, right=0.977, hspace=0.2, wspace=0.0)

    return f, grid


def main():
    """Creates all three pixel-pixel comparison plots for MDI, HMI, and AFT simulations."""
    create_variable_time_plot('mdi')
    create_variable_time_plot('hmi')
    create_variable_time_plot('sim')


if __name__ == '__main__':
    main()
