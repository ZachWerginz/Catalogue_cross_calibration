import cross_calibration as c
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from zaw_coord import CRD
import numpy as np

f1 = "fd_M_96m_01d.2226.0002.fits"
f2 = "fd_M_96m_01d.2227.0002.fits"

# f1 = "fd_M_96m_01d.6541.0005.fits"
# f2 = "hmi.M_720s.20101129_214800_TAI.1.magnetogram.fits"

m1, m2 = c.fix_longitude(f1, f2)
m2max = max(np.nanmin(m2.im_raw.data), np.nanmax(m2.im_raw.data))

f = plt.figure(1)
ax1 = f.add_subplot(231)
ax1.imshow(m1.im_raw.data, cmap='binary')
ax2 = f.add_subplot(232, sharey=ax1)
ax2.imshow(m2.im_raw.data, cmap='bwr', vmin=-m2max, vmax=m2max)
ax3 = f.add_subplot(233, sharey=ax1)
ax3.imshow(m1.im_raw.data, cmap='binary')
ax3.imshow(m2.im_raw.data, cmap='bwr', vmin=-m2max, vmax=m2max, alpha=.30)

f.subplots_adjust(left=.05, right=.95, wspace=0)

x = m2.lath.v.flatten()
y = m2.lonh.v.flatten()
values = m2.im_raw.data.flatten()
xi = m1.lath.v.flatten()
yi = m1.lonh.v.flatten()
dim = m1.im_raw.dimensions


ind2 = np.where(np.logical_and(np.isfinite(x), np.isfinite(values)))
#ind1 = np.where(np.logical_and(np.isfinite(xi), (m1.rg.v.flatten() < m1.rsun*np.sin(85.0*np.pi/180))))
ind1 = np.where(np.isfinite(xi))

interp_data = griddata((x[ind2], y[ind2]), values[ind2], (xi[ind1], yi[ind1]), method='cubic')
new_m2 = np.full((int(dim[0].value), int(dim[1].value)), np.nan)

new_m2.ravel()[ind1] = interp_data
new_m2.ravel()[(m1.rg.v.flatten() > m1.rsun*np.sin(75.0*np.pi/180))] = np.nan

ax1 = f.add_subplot(234)
ax1.imshow(m1.im_raw.data, cmap='binary')
ax2 = f.add_subplot(235, sharey=ax1)
ax2.imshow(new_m2, cmap='bwr', vmin=-m2max, vmax=m2max)
ax3 = f.add_subplot(236, sharey=ax1)
ax3.imshow(m1.im_raw.data, cmap='binary')
ax3.imshow(new_m2, cmap='bwr', vmin=-m2max, vmax=m2max, alpha=.30)


plt.show()