import util as u
import numpy as np
import quadrangles as quad
import matplotlib.pyplot as plt
import visualizations.cc_plot as ccplot

#
# def add_identity(axes, *line_args, **line_kwargs):
#     """Plots the identity line on the specified axis."""
#     identity, = axes.plot([], [], *line_args, **line_kwargs)
#
#     def callback(axes):
#         low_x, high_x = axes.get_xlim()
#         low_y, high_y = axes.get_ylim()
#         low = max(low_x, low_y)
#         high = min(high_x, high_y)
#         identity.set_data([low, high], [low, high])
#     callback(axes)
#     axes.callbacks.connect('xlim_changed', callback)
#     axes.callbacks.connect('ylim_changed', callback)

#
# def scatter(points, ax, axes_swap=False):
#     if axes_swap:
#         x, y, da = quad.extract_valid_points(points)
#     else:
#         y, x, da = quad.extract_valid_points(points)
#
#     co = 'viridis'
#     plt.rc('text', usetex=True)
#     i1 = points['i1']
#     i2 = points['i2']
#
#     sorted_inds = np.argsort(da)
#     f_i1 = y[sorted_inds]
#     f_i2 = x[sorted_inds]
#     da = da[sorted_inds]
#
#     ax.scatter(f_i2, f_i1, cmap=co, c=da, vmin=0, vmax=90, edgecolors='face', zorder=2)
#
#     # ------------- Set Plot Properties ----------------------------
#     add_identity(ax, color='.3', ls='-', zorder=1)
#     ax.axis('square')
#     max_field = max(np.nanmax(f_i1), np.nanmax(f_i2))
#     ax.set_ylim(-max_field, max_field)
#     ax.set_xlim(ax.get_ylim())


def main():
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.ion()
    color = color = (81 / 255, 178 / 255, 76 / 255)

    r_100 = u.download_cc_data('spmg', 'spmg', 100, '23 hours', '25 hours')
    fig = plt.figure(figsize=)
    ax1 = f1.add_subplot(121)
    ax2 = f1.add_subplot(122)
    ccplot.violin_plot(r_100, None, ax1, clr=color, percentiles=[0,100], axes_swap=False)
    ccplot.violin_plot(r_100, None, ax2, clr=color, percentiles=[0,100], axes_swap=True)
    ax1.set_ylabel(r'$\mathrm{{Reference\ Flux\ Density\ (Mx/cm^2)}}$')
    ax1.set_xlabel(r'$\mathrm{{+24\ hr\ Magnetogram Flux\ Density\ (Mx/cm^2)}}$')
    ax2.set_ylabel(r'$\mathrm{{Reference\ Flux\ Density\ (Mxs/cm^2)}}$')
    ax2.set_xlabel(r'$\mathrm{{-24\ hr\ Magnetogram\ Flux\ Density\ (Mx/cm^2)}}$')
    f1.suptitle("Reversed Axis Binning Box Plots", y=.85, fontsize=30, fontweight='bold')

    for ax, letter in zip(f1.get_axes(), 'ab'):
        ax.annotate(
            '{0}'.format(letter),
            xy=(0, 1), xycoords='axes fraction',
            xytext=(7, -25), textcoords='offset points',
            ha='left', va='bottom', fontsize=19, color='black', family='serif')
    plt.show()


if __name__ == '__main__':
    main()
