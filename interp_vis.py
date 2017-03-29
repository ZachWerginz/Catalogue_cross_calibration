import cross_calibration as c
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from zaw_coord import CRD
import numpy as np

f1 = "spmg_eo100_C1_19920424_1430.fits"
f2 = "spmg_eo100_C1_19920425_1540.fits"

m1, m2 = c.fix_longitude(f1, f2)
m2max = max(np.nanmin(m2.im_raw.data), np.nanmax(m2.im_raw.data))

f = plt.figure(1)
ax1 = f.add_subplot(231)
ax1.imshow(m1.im_corr.v, cmap='binary')
ax2 = f.add_subplot(232, sharey=ax1)
ax2.imshow(m2.im_corr.v, cmap='bwr', vmin=-m2max, vmax=m2max)
ax3 = f.add_subplot(233, sharey=ax1)
ax3.imshow(m1.im_corr.v, cmap='binary')
ax3.imshow(m2.im_corr.v, cmap='bwr', vmin=-m2max, vmax=m2max, alpha=.30)

f.subplots_adjust(left=.05, right=.95, wspace=0)

x2 = m2.lonh.v.flatten()
y2 = m2.lath.v.flatten()
v2 = m2.im_corr.v.flatten()
x1 = m1.lonh.v.flatten()
y1 = m1.lath.v.flatten()
dim1 = m1.im_raw.dimensions

latitudeMask = np.where(np.abs(y2) < 50)
minimum = max(np.nanmin(x2[latitudeMask]),np.nanmin(x1[latitudeMask]))
maximum = min(np.nanmax(x2[latitudeMask]), np.nanmax(x1[latitudeMask]))

ind2 = (np.isfinite(x2) * np.isfinite(y2) * np.isfinite(v2) * (x2 > minimum) * (x2 < maximum))
ind1 = (np.isfinite(x1) * np.isfinite(y1) * (x1 > minimum) * (x1 < maximum))

interp_data = griddata((x2[ind2], y2[ind2]), v2[ind2], (x1[ind1], y1[ind1]), method='cubic')
new_m2 = np.full((int(dim1[0].value), int(dim1[1].value)), np.nan)

new_m2.ravel()[ind1] = interp_data

ax1 = f.add_subplot(234)
ax1.imshow(m1.im_corr.v, cmap='binary')
ax2 = f.add_subplot(235, sharey=ax1)
ax2.imshow(new_m2, cmap='bwr', vmin=-m2max, vmax=m2max)
ax3 = f.add_subplot(236, sharey=ax1)
ax3.imshow(m1.im_corr.v, cmap='binary')
ax3.imshow(new_m2, cmap='bwr', vmin=-m2max, vmax=m2max, alpha=.30)

fig2 = plt.figure(2)
ax = fig2.add_axes([0,0,1,1])
ax.imshow(m1.im_corr.v, cmap='binary')
ax.imshow(new_m2, cmap='bwr', vmin=-m2max, vmax=m2max, alpha=.30)


plt.show()