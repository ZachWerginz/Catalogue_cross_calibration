import cross_calibration as c
import datetime as dt
from zaw_coord import CRD
import zaw_util as z
import itertools

files1_2 = c.process_instruments('512', 'spmg')
files2_3 = c.process_instruments('spmg', 'mdi')
files3_4 = c.process_instruments('mdi', 'hmi')

f = {}

f['m512'] =  files1_2[0][0]
f['mSPMGa'] = files1_2[0][1]
f['mSPMGb'] = files2_3[0][0]
f['mMDIa'] = files2_3[0][1]
f['mMDIb'] = files3_4[0][0]
f['mHMI'] = files3_4[0][1]

m512 = CRD(f['m512'])
mSPMGa = CRD(f['mSPMGa'])
mSPMGb = CRD(f['mSPMGb'])
mMDIa = CRD(f['mMDIa'])
mMDIb = CRD(f['mMDIb'])
mHMI = CRD(f['mHMI'])
for i in range(0, 6):
    try:
        f['m512_next'] = z.search_file(m512.im_raw.date + dt.timedelta(4 + i), '512')
    except IOError:
        continue

for i in range(0, 6):
    try:
        f['mSPMGa_next'] = z.search_file(mSPMGa.im_raw.date - dt.timedelta(4 + i), 'spmg')
    except IOError:
        continue

for i in range(0, 6):
    try:
        f['mMDIa_next'] = z.search_file(mMDIa.im_raw.date - dt.timedelta(4 + i), 'mdi')
    except IOError:
        continue

for i in range(0, 6):
    try:
        f['mHMI_next'] = z.search_file(mHMI.im_raw.date - dt.timedelta(4 + i), 'hmi')
    except IOError:
        continue

per = itertools.permutations(f.values(), 2)
pairs = []
pairFrags = []
for pair in per:
    p = c.fix_longitude(pair[0], pair[1])
    b = c.fragment_multiple(p[0], p[1], 10)
    try:
        c.block_plot(b[0], b[1])
    except:
        continue
    


