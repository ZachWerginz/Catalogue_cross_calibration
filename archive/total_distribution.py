import util as z
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def main():
    n = 400

    print("Downloading data for n = {}...".format(n))

    r1 = z.download_cc_data('512', '512', n, '23 hours', '25 hours')
    r2 = z.download_cc_data('512', '512', n, '47 hours', '49 hours')
    r3 = z.download_cc_data('512', 'spmg', n, '-2 days', '2 days')
    r4 = z.download_cc_data('spmg', 'spmg', n, '23 hours', '25 hours')
    r5 = z.download_cc_data('spmg', 'mdi', n, '0 minutes', '100 minutes')
    r6 = z.download_cc_data('mdi', 'mdi', n, '0 minutes', '100 minutes')
    r7 = z.download_cc_data('mdi', 'mdi', n, '23 hours', '25 hours')
    r8 = z.download_cc_data('mdi', 'mdi', n, '47 hours', '49 hours')
    r9 = z.download_cc_data('mdi', 'hmi', n, '0 minutes', '100 minutes')


    y512 = r1['referenceFD']
    y512 = np.append(y512, r2['referenceFD'])
    y512 = np.append(y512, r3['referenceFD'])

    yspmg = r4['referenceFD']
    yspmg = np.append(yspmg, r5['referenceFD'])

    ymdi = r6['referenceFD']
    ymdi = np.append(ymdi, r7['referenceFD'])
    ymdi = np.append(ymdi, r8['referenceFD'])

    yhmi = r9['referenceFD']


    colors = [(255/255, 208/255, 171/255), 
            (114/255, 178/255, 229/255), 
            (81/255, 178/255, 76/255),
            (111/255, 40/255, 124/255)]

    print("Evaluating histograms...")
    #Stepfilled with equal bins
    n1, bins1, patches1 = plt.hist(y512, 50, normed=True, histtype='stepfilled', color=colors[0], alpha=1, edgecolor = "none", label='512c')
    n3, bins3, patches3 = plt.hist(ymdi, bins=bins1, normed=True, histtype='stepfilled', color=colors[1], alpha=1, edgecolor = "none", label='MDI')
    n2, bins2, patches2 = plt.hist(yspmg, bins=bins1, normed=True, histtype='stepfilled', color=colors[2], alpha=1, edgecolor = "none", label='SPMG')
    n4, bins4, patches4 = plt.hist(yhmi, bins=bins1, normed=True, histtype='stepfilled', color=colors[3], alpha=1, edgecolor = "none", label='HMI')

    #Just a step with equal bins
    n1, bins1, patches1 = plt.hist(y512, 50, normed=True, histtype='step', alpha=1, color=colors[0])
    n3, bins3, patches3 = plt.hist(ymdi, bins=bins1, normed=True, histtype='step', alpha=1, color=colors[1])
    n2, bins2, patches2 = plt.hist(yspmg, bins=bins1, normed=True, histtype='step', alpha=1, color=colors[2])
    n4, bins4, patches4 = plt.hist(yhmi, bins=bins1, normed=True, histtype='step', alpha=1, color=colors[3])

    ax = plt.gca()
    maxlim = max(np.abs(ax.get_xlim()))
    ax.set_xlim(-maxlim, maxlim)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Magnetic Flux Density (G)')
    ax.set_yscale('log')
    f = plt.gcf()
    fig_title = 'n = ' + str(n)
    f.suptitle(fig_title, y=.95, fontsize=30, fontweight='bold')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()