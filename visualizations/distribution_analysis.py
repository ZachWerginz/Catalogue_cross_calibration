import matplotlib.pyplot as plt
import util as u
import numpy as np
import quadrangles as quad
import matplotlib.pyplot as plt
import visualizations.cc_plot as ccplot
from scipy.stats import gaussian_kde
import scipy


def gaussian(x, A, sigma, mu):
    """Define a gaussian function."""
    return A / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def main():
    plt.rc('text', usetex=True)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.ion()
    color = color = (81 / 255, 178 / 255, 76 / 255)

    r_100 = u.download_cc_data('spmg', 'spmg', 100, '23 hours', '25 hours')

    fig = plt.figure(figsize=(19, 19 / 3))
    fig.subplots_adjust(top=0.940, bottom=0.11, left=0.125, right=0.89, hspace=0.2, wspace=0.0)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ccplot.violin_plot(r_100, None, ax1, clr=color, percentiles=[0, 100], axes_swap=False)
    ccplot.violin_plot(r_100, None, ax2, clr=color, percentiles=[0, 100], axes_swap=True)
    ax1.set_ylabel(r'$\mathrm{{Reference\ Flux\ Density\ (Mx/cm^2)}}$')
    ax2.set_xlabel(r'$\mathrm{{Magnetic\ Flux\ Density\ (Mx/cm^2)}}$')
    ax1.annotate(r'$\mathrm{{+24\ hr}}$', xy=(0.5, 1), xycoords='axes fraction', ha='center', color='black')
    ax2.annotate(r'$\mathrm{{-24\ hr}}$', xy=(0.5, 1), xycoords='axes fraction', ha='center', color='black')
    ax2.tick_params(axis='y', left='off', right='off', labelleft='off', labelright='off')
    ax3.yaxis.tick_right()

    for ax, letter in zip(fig.get_axes(), 'abc'):
        ax.annotate(
            '{0}'.format(letter),
            xy=(0, 1), xycoords='axes fraction',
            xytext=(7, -25), textcoords='offset points',
            ha='left', va='bottom', fontsize=19, color='black', family='serif')

    hl, x, y = ccplot.hist_axis(r_100, None)
    total_kernel = gaussian_kde(y)
    xspace = np.linspace(np.nanmin(x), np.nanmax(x), 3000)
    total_kernel_array = total_kernel.evaluate(xspace)
    bin = hl[12]
    bin_kernel = gaussian_kde(bin['data'])
    bin_kernel_array = bin_kernel.evaluate(xspace)
    d = bin_kernel_array/np.sqrt(total_kernel_array)
    max_ind = np.argmax(bin_kernel_array)
    ind_valid = (d < 2*d[max_ind])
    popt = scipy.optimize.least_squares(ccplot.gaussian_func,
                                        [np.abs(bin['sliceMed']), np.abs(bin['sliceMed'] / 5),
                                         bin['sliceMed']],
                                        args=(xspace[ind_valid], d[ind_valid]),
                                        jac='3-point', x_scale='jac', loss='soft_l1', f_scale=.1).x

    xs = xspace[ind_valid]
    y1 = total_kernel_array[ind_valid]/np.max(total_kernel_array[ind_valid])
    y2 = bin_kernel_array[ind_valid]/np.max(bin_kernel_array[ind_valid])
    y3 = d[ind_valid]/np.max(d[ind_valid])
    y4 = gaussian(xspace, *popt)[ind_valid]/np.max(gaussian(xspace, *popt)[ind_valid])

    cdf1 = np.nancumsum(bin_kernel_array)*(xspace[-1] - xspace[0])/3000
    ind1 = np.argmin(cdf1 - np.nanmax(cdf1))
    cdf2 = np.nancumsum(d)*(xspace[-1] - xspace[0])/3000
    ind2 = np.argmin(cdf2 - np.nanmax(cdf2))

    ax3.plot(xs, y1, label='Total Sun Distribution', color='blue')
    ax3.plot(xs, y2, label='Bin Distribution', color='orange')
    ax3.plot(xs, y3, label='Divided Distribution', color='green')
    ax3.axvline(x=xspace[ind1], color='orange', ls='-', zorder=1, label='Bin Median')
    ax3.axvline(x=xspace[ind2], color='green', ls='-', zorder=1, label='Divided Distribution Median')
    ax3.axvline(x=bin['sliceMed'], color='.2', ls='--', zorder=1, label='Location of Bin')
    ax3.set_ylabel(r'$\mathrm{{Probability Density}}$', rotation=270, labelpad=20)
    ax3.yaxis.set_label_position('right')
    ax3.set_xlim([0, 300])


if __name__ == '__main__':
    main()
