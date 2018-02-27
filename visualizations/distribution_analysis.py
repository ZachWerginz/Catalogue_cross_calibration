import matplotlib.pyplot as plt
import util as u
import numpy as np
import quadrangles as quad
import matplotlib.pyplot as plt
import visualizations.cc_plot as ccplot
from scipy.stats import gaussian_kde
import scipy

def main():
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.ion()
    color = color = (81 / 255, 178 / 255, 76 / 255)

    r_100 = u.download_cc_data('spmg', 'spmg', 100, '23 hours', '25 hours')

    fig = plt.figure(figsize=(17, 11))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ccplot.violin_plot(r_100, None, ax1, clr=color, percentiles=[0,100], axes_swap=False)
    ccplot.violin_plot(r_100, None, ax2, clr=color, percentiles=[0,100], axes_swap=True)
    ax1.set_ylabel(r'$\mathrm{{Reference\ Flux\ Density\ (Mx/cm^2)}}$')
    ax1.set_xlabel(r'$\mathrm{{+24\ hr\ Magnetogram Flux\ Density\ (Mx/cm^2)}}$')
    ax2.set_ylabel(r'$\mathrm{{Reference\ Flux\ Density\ (Mxs/cm^2)}}$')
    ax2.set_xlabel(r'$\mathrm{{-24\ hr\ Magnetogram\ Flux\ Density\ (Mx/cm^2)}}$')
    #f1.suptitle("Reversed Axis Binning Box Plots", y=.85, fontsize=30, fontweight='bold')

    for ax, letter in zip(fig.get_axes(), 'ab'):
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
    # new_arr1 = np.full(cdf1.shape, np.nanmax(cdf1) / 2)
    # ind1 = np.isclose(new_arr1, cdf1, rtol=.005)
    ind1 = np.argmin(cdf1 - np.nanmax(cdf1))
    cdf2 = np.nancumsum(d)*(xspace[-1] - xspace[0])/3000
    # half_max = np.nanmax(cdf2)/2
    # new_arr2 = np.full(cdf2.shape, np.nanmax(cdf2)/2)
    # ind2 = np.isclose(new_arr2, cdf2, rtol=.005)
    ind2 = np.argmin(cdf2 - np.nanmax(cdf2))

    ax3.plot(xs, y1, label='Total Sun Distribution', color='blue')
    ax3.plot(xs, y2, label='Bin Distribution', color='orange')
    ax3.plot(xs, y3, label='Divided Distribution', color='green')
    ax3.axvline(x=xspace[ind1], color='orange', ls='-', zorder=1, label='Bin Median')
    ax3.axvline(x=xspace[ind2], color='green', ls='-', zorder=1, label='Divided Distribution Median')
    ax3.axvline(x=bin['sliceMed'], color='.2', ls='--', zorder=1, label='Location of Bin')
    ax3.set(xlabel=r'$\mathrm{{{0}\ Magnetic\ Flux\ Density\ (Mx/cm^2)}}$'.format(i1.upper()),
            ylabel='Probability Density')
    # f2.suptitle('Flux Density Distributions With Medians', y=.92, weight='bold')
    ax3.set_xlim([0, 300])


if __name__ == '__main__':
    main()
