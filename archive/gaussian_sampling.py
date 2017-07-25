import zaw_util as z
import numpy as np
import random
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, A, sigma, mu):
    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))

r = z.download_cc_data('512', '512', 50, '23 hours', '25 hours')
xedges = np.linspace(np.nanmin(r['secondaryFD']), np.nanmax(r['secondaryFD']), 100)

while True:
    try:
        rand = int(random.random()*len(xedges))
        print(xedges[rand])
        ind = np.where( (r['secondaryFD'] < xedges[rand]) & (r['secondaryFD'] > xedges[rand - 1]))
        xspace = np.linspace(xedges[0], xedges[-1], 1000)
        totalKernel = gaussian_kde(r['referenceFD'])
        binKernel = gaussian_kde(r['referenceFD'][ind])
        bK = binKernel.evaluate(xspace)
        tK = totalKernel.evaluate(xspace)
        optimizeInd = (np.isfinite(bK/tK) & (np.abs(bK/tK) < 1e4) & (np.abs(bK/tK) > 1e-4))
        popt, pcov = curve_fit(gaussian, xspace[optimizeInd], (bK/tK)[optimizeInd], p0=[1000, 50, xedges[rand]], maxfev=10000)
        if popt[0]: break
    except:
        continue

f1 = plt.figure(1)
ax1 = f1.add_axes([0,0,1,1])
ax1.plot(xspace, bK, 'b--', xspace, tK,'g--')
popt1, pcov1 = curve_fit(gaussian, xspace, bK)
print(popt1)
ax1.plot(xspace, gaussian(xspace, *popt1), 'b-')
popt2, pcov2 = curve_fit(gaussian, xspace, tK)
print(popt2)
ax1.plot(xspace, gaussian(xspace, *popt2), 'g-')

p = np.corrcoef(bK, tK)[0][1]
print("p = {}".format(p))
t = (popt2[2]*(bK/tK) - popt1[2])/np.sqrt((popt2[1]**2)*((bK/tK)**2) - 2*popt1[1]*popt2[1]*bK/tK*p + (popt[1]**2))
EW = popt2[2]/popt1[2] + (popt1[1]**2)*popt2[2]/(popt1[2]**3) + p*(popt1[1]**2)*(popt2[1])**2/(popt1[2]**2)
print(EW)

f3 = plt.figure(3)
ax3 = f3.add_axes([0,0,1,1])
ax3.plot(xspace, t)

f2 = plt.figure(2)
ax2 = f2.add_axes([0,0,1,1])
popt3, pcov3 = curve_fit(gaussian, xspace[optimizeInd], (bK/tK)[optimizeInd], p0=[1000, 50, xedges[rand]], maxfev=10000)
print(popt3)
plt.plot(xspace, bK/np.nanmax(bK), 'b--', xspace, tK/np.nanmax(tK),'g-')
plt.plot(xspace[optimizeInd], (bK[optimizeInd]/tK[optimizeInd])/np.nanmax(bK[optimizeInd]/tK[optimizeInd]), 'r:')
plt.plot(xspace[optimizeInd], gaussian(xspace[optimizeInd], *popt3)/np.nanmax(bK[optimizeInd]/tK[optimizeInd]), 'r-')
popt4, pcov4 = curve_fit(gaussian, xspace[optimizeInd], (bK/np.sqrt(tK))[optimizeInd], p0=[1000, 50, xedges[rand]], maxfev=10000)
plt.plot(xspace[optimizeInd], gaussian(xspace[optimizeInd], *popt4)/np.nanmax(bK[optimizeInd]/np.sqrt(tK)[optimizeInd]), 'r-')

plt.show()