import util as z
import visualizations.cc_plot as b
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import scipy
import matplotlib.pyplot as plt


def gaussian_func(p, x, y):
    """Define a gaussian function for use with scipy.optimize."""
    return p[0] / (np.sqrt(2 * np.pi) * p[1]) * np.exp(-(x - p[2]) ** 2 / (2. * p[1] ** 2)) - y


def main():
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Download data
    i1 = '512'
    i2 = 'spmg'
    r = z.download_cc_data(i1, i2, 50, '0 days', '2 days')

    hl, x, y = b.hist_axis(r, None)
    # y = np.abs(y)
    # x = np.abs(x)
    total_kernel = gaussian_kde(y)
    xspace = np.linspace(np.nanmin(x), np.nanmax(x), 3000)
    total_kernel_array = total_kernel.evaluate(xspace)
    bin = hl[12]
    bin_kernel = gaussian_kde(bin['data'])
    bin_kernel_array = bin_kernel.evaluate(xspace)
    d = bin_kernel_array/np.sqrt(total_kernel_array)
    max_ind = np.argmax(bin_kernel_array)
    ind_valid = (d < 2*d[max_ind])
    #popt, pcov = curve_fit(gaussian, xspace[ind_valid], d[ind_valid],
    #                       p0=[np.abs(bin['sliceMed']), np.abs(bin['sliceMed']/5), bin['sliceMed']], maxfev=10000)
    ind_valid = (d < 1e10)
    popt = scipy.optimize.least_squares(gaussian_func,
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
    new_arr1 = np.full(cdf1.shape, np.nanmax(cdf1) / 2)
    ind1 = np.isclose(new_arr1, cdf1, rtol=.005)
    cdf2 = np.nancumsum(d)*(xspace[-1] - xspace[0])/3000
    half_max = np.nanmax(cdf2)/2
    new_arr2 = np.full(cdf2.shape, np.nanmax(cdf2)/2)
    ind2 = np.isclose(new_arr2, cdf2, rtol=.005)
    print(ind1.sum())
    print(ind2.sum())

    bin_max = np.argmax(y2)
    d_max = np.argmax(y4)

    f1 = plt.figure(100, figsize=(17, 11))
    ax1 = f1.add_subplot(111)
    ax1.plot(xs, y1, label='Total Sun Distribution')
    ax1.plot(xs, y2, label='Bin Distribution')
    ax1.plot(xs, y3, label='bin/sqrt(total)')
    ax1.set(xlabel=r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
            ylabel='Probability Density Normalized to Mode')
    f1.suptitle('Flux Density Distributions', y=.92, weight='bold')
    plt.legend(frameon=False, framealpha=0)
    ax1.set_xlim([0, 300])

    f2 = plt.figure(200, figsize=(17, 11))
    ax2 = f2.add_subplot(111)
    ax2.plot(xs, y1, label='Total Sun Distribution', color='blue')
    ax2.plot(xs, y2, label='Bin Distribution', color='orange')
    ax2.plot(xs, y3, label='Divided Distribution', color='green')
    ax2.axvline(x=xspace[ind1], color='orange', ls='-', zorder=1, label='Bin Median')
    ax2.axvline(x=xspace[ind2], color='green', ls='-', zorder=1, label='Divided Distribution Median')
    ax2.axvline(x=bin['sliceMed'], color='.2', ls='--', zorder=1, label='Location of Bin')
    ax2.set(xlabel=r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
            ylabel='Probability Density')
    f2.suptitle('Flux Density Distributions With Medians', y=.92, weight='bold')
    ax2.set_xlim([0, 300])

    plt.legend(frameon=False, framealpha=0)
    f1.savefig('flux_density_distributions.pdf', bbox_inches='tight', pad_inches=.1)
    f2.savefig('flux_density_dist_with_gaussians.pdf', bbox_inches='tight', pad_inches=.1)

    plt.show()


if __name__ == '__main__':
    main()
