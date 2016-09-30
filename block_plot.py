import numpy as np
import matplotlib.pyplot as plt
import copy

def scatter_plot(dict1, dict2, separate=False):
    i = 1
    for n in sorted([x for x in dict1.keys() if x != 'instr']):
        if separate:
            plt.subplot(len(dict1), 1, i)
            plt.xlabel(dict1['instr'])
            plt.ylabel(dict2['instr'])
            plt.title('Number of blocks: {}'.format(n))
        plt.plot([np.nanmin(dict1[n]),np.nanmax(dict1[n])],[np.nanmin(dict1[n]),np.nanmax(dict1[n])], 'k-', lw=2)
        plt.plot(dict1[n], dict2[n], '.')
        i += 1
    plt.show()

def plot_block_parameters(p1, p2):
    ar_i1 = p1[0]
    ar_i2 = p2[0]
    ind = np.where(np.abs((ar_i1 - ar_i2)/ar_i1) < .10)
    c_valid = p1[3][ind]
    sortedInds = np.argsort(c_valid)
    ar_i1 = ar_i1[ind][sortedInds]
    ar_i2 = ar_i2[ind][sortedInds]
    f_i1 = p1[2][ind][sortedInds]
    f_i2 = p2[2][ind][sortedInds]


    f1_valid = (np.abs(f_i1) > 1)
    f2_valid = (np.abs(f_i2) > 1)
    s = f1_valid & f2_valid
    f, ax = plt.subplots(1,3)
    co = 'viridis'
    ax[0].plot([0, 1.3e19], [0, 1.3e19], 'r-')
    ax[0].scatter(ar_i1, ar_i2, cmap=co, c=c_valid, edgecolors='face')
    ax[0].set_title('Block Areas')
    ax[0].set_aspect('equal')
    ax[1].plot([-400, 400], [-400, 400], 'r-')
    ax[1].scatter(f_i1, f_i2, cmap=co, c=c_valid, edgecolors='face')
    ax[1].set_title('Mean Field')
    ax[1].set_aspect('equal')
    ax[2].plot([-3, 3], [-3, 3], 'r-')
    x1 = np.log10(np.abs(f_i1[s]))*np.sign(f_i1[s])
    y1 = np.log10(np.abs(f_i2[s]))*np.sign(f_i2[s])
    ax[2].scatter(x1, y1, cmap=co, c=c_valid[s], edgecolors='face')
    ax[2].axhline(0, color='black')
    ax[2].axvline(0, color='black')
    ax[2].set_title('Logarithm Space of Fields')
    ax[2].set_aspect('equal')

    return f

def n_plot(n_dict):
    NList = []
    TList = []
    for key, value in n_dict.items():
        NList.append(key)
        TList.append(len(value))
    N = np.array(NList)
    T = np.array(TList)
    plt.plot(N, T, '.')

    plt.show()

def block_plot(*args, overlay=True):
    """Given a list of blocks, will plot a nice image differentiating them.

    --Optional arguments--
    overlay: """
    n = len(args)
    print(n)
    rows = int(round(np.sqrt(n)))
    cols = int(np.ceil(np.sqrt(n)))
    im = {}
    ax = {}
    for i in range(len(args)):
        print(i)
        if isinstance(args[i], type(list())):
            im[i] = args[i][0].mgnt.lonh.v.copy()
            im[i][:] = np.nan
            for x in args[i]:
                im[i][x.indices] = x.pltColor
            ax[i] = plt.subplot2grid((rows, cols), (i%rows, i//rows))
            if overlay:
                ax[i].imshow(args[i][0].mgnt.im_raw.data, cmap='binary')
            ax[i].imshow(im[i], vmin=0, vmax=1, alpha=.2)
            title = "{}: {}".format(args[i][0].mgnt.im_raw.instrument, 
                    args[i][0].mgnt.im_raw.date.isoformat())
            ax[i].set_title(title)
        else:
            im[i] = args[i]
            ax[i] = plt.subplot2grid((rows, cols), (i%rows, i//rows))
            ax[i].imshow(im[i], cmap='binary')

    return ax