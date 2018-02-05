import matplotlib.pyplot as plt
import glob
import cross_calibration as c
from coord import CRD
import matplotlib.colors as colors
import numpy as np
import scipy as sp
import scipy.optimize as sciop
from matplotlib.ticker import MaxNLocator
import util as u


def get_pcoeff(increment, m1=None, file2=None, cond=None):
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


def plot_hist2d(m1, m2, cond):
    x = m2.remap.ravel()
    y = m1.im_corr.v.ravel()
    edges = np.arange(-987.5, 987.5, 25)

    f = plt.figure()
    ax = f.add_subplot(111)
    ind = (np.abs(x) > cond) * (np.abs(y) > cond) * np.isfinite(x) * np.isfinite(y)

    ax.hist2d(x[ind], y[ind], cmap='inferno', norm=colors.LogNorm(), bins=edges)
    ax.set_facecolor('black')
    ax.set(adjustable='box-forced', aspect='equal')


def export_to_standard_form(x, y):
    bl = {'referenceFD': y, 'secondaryFD': x, 'diskangle': None}
    return bl


def create_sim_variable_times():
    cond = 20
    times = ['2 hrs', '6 hrs', '24 hrs']

    files = glob.glob('test_mgnts/*.dat')
    f, grid = plt.subplots(1, 3, figsize=(16, 16/3), sharex='col', sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
    lim = 1000
    edges = np.arange(-987.5, 987.5, 25)

    for file, ax in zip(files[1:], grid.flatten()):
        m1, m2 = c.sim_compare(files[0], file)
        x = m2.remap.ravel()
        y = m1.im_corr.v.ravel()
        ind = (np.abs(x) > cond) * (np.abs(y) > cond) * np.isfinite(x) * np.isfinite(y)
        ax.hist2d(x[ind], y[ind], cmap='inferno', norm=colors.LogNorm(), bins=edges)
        ax.axis([-lim, lim, -lim, lim])
        ax.set_facecolor('black')

    for ax, letter in zip(f.get_axes(), times):
        ax.annotate(
            '{0}'.format(letter),
            xy=(0, 1), xycoords='axes fraction',
            xytext=(7, -25), textcoords='offset points',
            ha='left', va='bottom', fontsize=19, color='white', family='serif')

    for ax in grid.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)

    grid[0].set_ylabel('AFT Simulation', fontsize=30)


def create_variable_time_plot(instr):
    if instr == 'mdi':
        files = glob.glob('test_mgnts/fd*')
        f, grid = plt.subplots(1, 3, figsize=(16, 16 / 3), sharex='col', sharey='row',
                               gridspec_kw={'wspace': 0, 'hspace': 0})
        grid[0].set_ylabel('MDI', fontsize=30)
        cond = 26
        times = ['1.6 hrs', '6.4 hrs', '24 hrs']
    elif instr == 'hmi':
        files = glob.glob('test_mgnts/hmi*')
        f, grid = plt.subplots(2, 3, figsize=(16, 16/1.5), sharex='col', sharey='row',
                               gridspec_kw={'wspace': 0, 'hspace': 0})
        #grid[0].set_ylabel('HMI', fontsize=30)
        f.text(0.07, 0.5, 'HMI', ha='center', va='center', rotation='vertical', fontsize=30)
        cond = 20
        times = ['12 min', '24 min', '48 min', '1.6 hrs', '6.4 hrs', '24 hrs']
    lim = 1000
    edges = np.arange(-987.5, 987.5, 25)

    for file, ax in zip(files[1:], grid.flatten()):
        m1, m2 = c.fix_longitude(files[0], file)
        x = m2.remap.ravel()
        y = m1.im_corr.v.ravel()
        ind = (np.abs(x) > cond) * (np.abs(y) > cond) * np.isfinite(x) * np.isfinite(y)
        ax.hist2d(x[ind], y[ind], cmap='inferno', norm=colors.LogNorm(), bins=edges)
        ax.axis([-lim, lim, -lim, lim])
        ax.set_facecolor('black')

    for ax, letter in zip(f.get_axes(), times):
        ax.annotate(
            '{0}'.format(letter),
            xy=(0, 1), xycoords='axes fraction',
            xytext=(7, -25), textcoords='offset points',
            ha='left', va='bottom', fontsize=19, color='white', family='serif')

    for ax in grid.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)

    return f, grid


def main():
    create_variable_time_plot('mdi')


if __name__ == '__main__':
    main()
