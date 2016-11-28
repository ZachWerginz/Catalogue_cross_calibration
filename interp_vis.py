import cross_calibration as c
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from zaw_coord import CRD
import numpy as np

f1 = "fd_M_96m_01d.2226.0002.fits"
f2 = "fd_M_96m_01d.2227.0002.fits"

m1, m2 = c.fix_longitude(f1, f2)
m2max = max(np.nanmin(m2.im_raw.data), np.nanmax(m2.im_raw.data))

f = plt.figure(1)
ax1 = f.add_subplot(131)
ax1.imshow(m1.im_raw.data, cmap='binary')
ax2 = f.add_subplot(132, sharey=ax1)
ax2.imshow(m2.im_raw.data, cmap='bwr', vmin=-m2max, vmax=m2max)
ax3 = f.add_subplot(133, sharey=ax1)
ax3.imshow(m1.im_raw.data, cmap='binary')
ax3.imshow(m2.im_raw.data, cmap='bwr', vmin=-m2max, vmax=m2max, alpha=.30)

f.subplots_adjust(left=.05, right=.95, wspace=0)

interp_data = griddata(points=(m2.lath.v, m2.lonh.v), values=m2.im_raw.data, xi=(m1.lath.v, m1.lonh.v), method='cubic')


plt.show()