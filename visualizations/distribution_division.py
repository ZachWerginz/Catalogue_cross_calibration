import util as z
import visualizations.cc_plot as b
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def gaussian(x, A, sigma, mu):
    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))


def main():
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Download data
    i1 = '512'
    i2 = 'spmg'
    r = z.download_cc_data(i1, i2, 50, '0 days', '2 days')

    hl, x, y = b.hist_axis(r, [0, 30, 45, 79])
    total_kernel = gaussian_kde(y)
    xspace = np.linspace(np.nanmin(x), np.nanmax(x), 3000)
    total_kernel_array = total_kernel.evaluate(xspace)
    bin = hl[11]
    bin_kernel = gaussian_kde(bin['data'])
    bin_kernel_array = bin_kernel.evaluate(xspace)
    d = bin_kernel_array/np.sqrt(total_kernel_array)
    max_ind = np.argmax(bin_kernel_array)
    ind_valid = (d < 2*d[max_ind])
    popt, pcov = curve_fit(gaussian, xspace[ind_valid], d[ind_valid],
        p0=[np.abs(bin['sliceMed']), np.abs(bin['sliceMed']/5), bin['sliceMed']],
        maxfev=10000)

    xs = xspace[ind_valid]
    y1 = total_kernel_array[ind_valid]/np.max(total_kernel_array[ind_valid])
    y2 = bin_kernel_array[ind_valid]/np.max(bin_kernel_array[ind_valid])
    y3 = d[ind_valid]/np.max(d[ind_valid])
    y4 = gaussian(xspace, *popt)[ind_valid]/np.max(gaussian(xspace, *popt)[ind_valid])

    bin_max = np.argmax(y2)
    d_max = np.argmax(y4)

    f1 = plt.figure(1, figsize=(17, 11))
    ax1 = f1.add_subplot(111)
    ax1.plot(xs, y1, label='Total Sun Distribution')
    ax1.plot(xs, y2, label='Bin Distribution')
    ax1.plot(xs, y3, label='bin/sqrt(total)')
    ax1.set(xlabel=r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
            ylabel='Probability Density')
    f1.suptitle('Flux Density Distributions', y=.92, weight='bold')
    plt.legend(frameon=False, framealpha=0)

    f2 = plt.figure(2, figsize=(17, 11))
    ax2 = f2.add_subplot(111)
    ax2.plot(xs, y1, label='Total Sun Distribution')
    ax2.plot(xs, y2, label='Bin Distribution')
    ax2.plot(xs, y4, label='Gaussian Fit')
    ax2.axvline(x=xspace[bin_max], color='.3', ls='--', zorder=1, label='Bin Peak')
    ax2.axvline(x=xspace[d_max], color='.3', ls='-', zorder=1, label='Divided Distribution Peak')
    ax2.set(xlabel=r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
            ylabel='Probability Density')
    f2.suptitle('Flux Density Distributions With Fitted Gaussian', y=.92, weight='bold')

    plt.legend(frameon=False, framealpha=0)
    f1.savefig('flux_density_distributions.pdf', bbox_inches='tight', pad_inches=.1)
    f2.savefig('flux_density_dist_with_gaussians.pdf', bbox_inches='tight', pad_inches=.1)

    plt.show()


if __name__ == '__main__':
    main()
