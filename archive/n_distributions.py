import util as z
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import cycle

def main():
    instrumentKey = {'512': 1, 'spmg': 2, 'mdi': 3, 'hmi': 4}
    colors = [(255/255, 208/255, 171/255), 
            (114/255, 178/255, 229/255), 
            (81/255, 178/255, 76/255),
            (111/255, 40/255, 124/255),
            (80/255, 60/255, 0/255)]
    colorcycler = cycle(colors)
    i = '512'

    print("Downloading data for {}...".format(i))
    dataDict = {}

    if i == '512':
        for n in [25, 50, 100, 200, 400]:
            dataDict[str(n)] = z.download_cc_data('512', '512', n, '-49 hours', '49 hours')['referenceFD']
            dataDict[str(n)] = np.append(dataDict[str(n)], z.download_cc_data('512', 'spmg', n, '-49 hours', '49 hours')['referenceFD'])
    elif i == 'spmg':
        for n in [25, 50, 100, 200, 400]:
            dataDict[str(n)] = z.download_cc_data('spmg', 'spmg', n, '-49 hours', '49 hours')['referenceFD']
            dataDict[str(n)] = np.append(dataDict[str(n)], z.download_cc_data('spmg', 'mdi', n, '-49 hours', '49 hours')['referenceFD'])
    elif i == 'mdi':
        for n in [25, 50, 100, 200, 400]:
            dataDict[str(n)] = z.download_cc_data('mdi', 'mdi', n, '-49 hours', '49 hours')['referenceFD']
    elif i == 'hmi':
        for n in [25, 50, 100, 200, 400]:
            dataDict[str(n)] = z.download_cc_data('mdi', 'hmi', n, '-49 hours', '49 hours')['referenceFD']

    for k, value in sorted(dataDict.items(), key=lambda x: int(x[0]), reverse=True):
        c = next(colorcycler)
        n1, bins1, patches1 = plt.hist(value, 50, normed=True, histtype='stepfilled', color=c, alpha=1, edgecolor='none', label=k)

    for k, value in sorted(dataDict.items(), key=lambda x: int(x[0]), reverse=True):
        c = next(colorcycler)
        n2, bins2, patches2 = plt.hist(value, bins=bins1, normed=True, histtype='step', color=c, alpha=1)  

    ax = plt.gca()
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Magnetic Flux Density (G)')
    ax.set_yscale('log')
    ax.set_xlim(-4600, 4600)
    f = plt.gcf()
    fig_title = i
    f.suptitle(fig_title, y=.95, fontsize=30, fontweight='bold')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()