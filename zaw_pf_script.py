# This is a script that calculates flux information for magnetograms
# poles. The poles are based on certain criteria to make sure we
# don't factor in enormously projected areas of the sun.
#
#    Variables:
#               p_px:      Polar cap pixels
#               vp_px:     Valid polar cap pixels
#               posp_px:   Positive flux polar cap pixels
#               negp_px:   Negative flux polar cap pixels
#               swt:       Switch to determine bad polar cap type
#

import numpy as np
import pandas as pd
import datetime as dt
import uncertainty
from uncertainty import Measurement as M
import zaw_util
from collections import OrderedDict
import os.path
import getopt
import sys

# This is the container class for the entire segmented polar data.
class PFData:
    pass

d1 = None       # Start date
d2 = None       # End date
instr = None    # Instrument

# Data dictionary initialization to be stored in an ordered list.
data0 = OrderedDict([('md', 0), ('date', ''), ('meanf_n', M(np.nan, np.nan)),
        ('meanfc_n', M(np.nan, np.nan)), ('sumf_n', M(np.nan, np.nan)), ('sumfc_n', M(np.nan, np.nan)),
        ('unsflux_n', M(np.nan, np.nan)), ('unsfluxc_n', M(np.nan, np.nan)), ('sflux_n', M(np.nan, np.nan)),
        ('sfluxc_n', M(np.nan, np.nan)), ('posfluxc_n', M(np.nan, np.nan)), ('negfluxc_n', M(np.nan, np.nan)),
        ('nvp_px_n', np.nan), ('p_ratio_n', np.nan), ('visarea_n', M(np.nan, np.nan)),
        ('max_pxflux_n', np.nan), ('max_pxf_n', np.nan), ('max_pxfc_n', np.nan),
        ('swt_n', np.nan), ('meanf_s', M(np.nan, np.nan)), ('meanfc_s', M(np.nan, np.nan)),
        ('sumf_s', M(np.nan, np.nan)), ('sumfc_s', M(np.nan, np.nan)), ('unsflux_s', M(np.nan, np.nan)), 
        ('unsfluxc_s', M(np.nan, np.nan)), ('sflux_s', M(np.nan, np.nan)), ('sfluxc_s', M(np.nan, np.nan)), 
        ('posfluxc_s', M(np.nan, np.nan)), ('negfluxc_s', M(np.nan, np.nan)), ('nvp_px_s', np.nan), 
        ('p_ratio_s', np.nan), ('visarea_s', M(np.nan, np.nan)), ('max_pxflux_s', np.nan),
        ('max_pxf_s', np.nan),('max_pxfc_s', np.nan), ('swt_s', np.nan)])

def usage():
    print('Usage: zaw_pf_script.py [-d data-root] [-s start-date] [-e end-date] [-i instrument]')

def parse_args():
    global d1, d2, instr

    try:
        opts, args = getopt.getopt(
                sys.argv[1:],
                "d:s:e:i:", 
                ["data-root=", "date-start=", "date-end=", "instrument=", "debug"])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--data-root"):
            zaw_util.data_root = arg
        elif opt in ("-s", "--date-start"):
            d1 = arg
        elif opt in ("-e", "--date-end"):
            d2 = arg
        elif opt in ("-i", "--instrument"):
            instr = arg
        elif opt in ("--debug"):
            zaw_util.debug = True
        else:
            assert False, "unhandled option"

def init(seg):
    global d1, d2, instr
    seg.pf = []
    seg.meta = {'start_date':dt.date(int(d1[0:4]), int(d1[5:7]), int(d1[8:10])),
              'end_date':dt.date(int(d2[0:4]), int(d2[5:7]), int(d2[8:10])),
              'md_i': None,
              'md_f': None,
              'instrument': instr, 'deg_lim': 65.0, 'inv_px_tol': 0.85}
    seg.meta['md_i'] = zaw_util.date2md(seg.meta['start_date'], seg.meta['instrument'])
    seg.meta['md_f'] = zaw_util.date2md(seg.meta['end_date'], seg.meta['instrument'])

def calc_pol(mgnt, pole, pf_data=None, meta={'deg_lim': 75.0, 'inv_px_tol': .85}):
    if pf_data is None:
        pf_data = data0.copy()
    pole = pole.lower()

    p_px, vp_px, posp_px, negp_px = indices(mgnt, pole, meta['deg_lim'])
    
    swt = validate(p_px, vp_px, posp_px, negp_px, pole, meta['inv_px_tol'])
    # Return with no data if the two previous cases are true.
    if swt == 3:
        pf_data['md'] = mgnt.md
        pf_data['date'] = mgnt.date
        if pole == 'north':
            pf_data['swt_n'] = swt
        else:
            pf_data['swt_s'] = swt

        return pf_data
    # Otherwise calculate data from the poles.
    else:
        meanf = M.nanmean(mgnt.im_raw_u[vp_px])
        meanfc = M.nanmean(mgnt.im_corr[vp_px])
        sumf = M.nansum(mgnt.im_raw_u[vp_px])
        sumfc = M.nansum(mgnt.im_corr[vp_px])
        unsflux = M.nansum(abs(mgnt.mflux_raw[vp_px]))
        unsfluxc = M.nansum(abs(mgnt.mflux_corr[vp_px]))
        sflux = M.nansum(mgnt.mflux_raw[vp_px])
        sfluxc = M.nansum(mgnt.mflux_corr[vp_px])
        posfluxc = M.nansum(mgnt.mflux_corr[posp_px])
        negfluxc = M.nansum(mgnt.mflux_corr[negp_px])
        visarea = M.nansum(mgnt.area[vp_px])
        max_pxflux = M.nanmax(abs(mgnt.mflux_corr[vp_px]))
        max_pxf = M.nanmax(mgnt.im_raw.data[vp_px])
        max_pxfc =  M.nanmax(mgnt.im_corr[vp_px])

        # Corrected field might be zero. Replace entries with NaNs.
        if unsfluxc == 0.0:
            unsfluxc = M(np.nan, np.nan)
            sfluxc = M(np.nan, np.nan)
            posfluxc = M(np.nan, np.nan)
            negfluxc = M(np.nan, np.nan)
            max_pxflux = np.nan
            swt = 4

        var = {'meanf': meanf, 'meanfc': meanfc, 'sumf': sumf, 'sumfc': sumfc,
                'unsflux': unsflux, 'unsfluxc': unsfluxc, 'sflux': sflux,
                'sfluxc': sfluxc, 'posfluxc': posfluxc, 'negfluxc': negfluxc,
                'nvp_px': np.size(vp_px), 'p_ratio': np.size(vp_px)/np.size(p_px),
                'visarea': visarea, 'max_pxflux': max_pxflux, 'max_pxf': max_pxf,
                'max_pxfc': max_pxfc, 'swt': swt}
        # Assign data values per north/south pole.
        for x in var.items():
            if pole == 'north':
                pf_data[x[0] + '_n'] = x[1]
            else:
                pf_data[x[0] + '_s'] = x[1]
        pf_data['md'] = mgnt.md
        pf_data['date'] = mgnt.date     
        return pf_data

def indices(m, pole, dlim=75.0):
    print ("Determining polar cap pixels.")
    if pole == 'north':
        p = np.where((m.lath > dlim) & (m.rg < m.rsun*np.sin(75.0*np.pi/180)))

        vp = np.where((m.rg < m.rsun) & (m.lath > dlim) 
                & M.isfinite(m.im_corr))

        posp = np.where((m.rg < m.rsun) & (m.lath > dlim)
                & (m.im_corr > 0.0))

        negp = np.where((m.rg < m.rsun) & (m.lath > dlim)
                & (m.im_corr < 0.0))
    else:
        p = np.where((m.lath < -dlim) & (m.rg < m.rsun*np.sin(75.0*np.pi/180)))

        vp = np.where((m.rg < m.rsun) & (m.lath < -dlim) 
                & M.isfinite(m.im_corr))

        posp = np.where((m.rg < m.rsun) & (m.lath < -dlim)
                & (m.im_corr > 0.0))

        negp = np.where((m.rg < m.rsun) & (m.lath < -dlim)
                & (m.im_corr < 0.0))

    return p, vp, posp, negp

def validate(p, vp, pos, neg, pole, tol=0.85):
    print ("Validating polar cap.")
    # Search for polarity mixture.
    if (np.size(pos) == 0 or np.size(neg) == 0):
        print ("{} hemisphere has no polarity mixture.".format(pole))
        return 3
    return 0

def process(seg, date):
    while date <= seg.meta['end_date']:
        print(date)
        mgnt = zaw_util.CRD_read(date, seg.meta['instrument'])
        if mgnt != -1:
            print("Calculating polar parameters")
            data = data0.copy()
            calc_pol(data, mgnt, 'north', seg.meta)
            calc_pol(data, mgnt, 'south', seg.meta)
            seg.pf.append(data)
            split_and_export(seg)
            export_to_hdf(seg)
        date = date + dt.timedelta(1)
        

def export_to_csv(seg):
    obj = pd.DataFrame(seg.pf, columns=seg.pf[0].keys())
    out_fn = 'PF_{}_{}.csv'.format(seg.meta['start_date'].isoformat(),
                                    seg.meta['end_date'].isoformat())

    obj.to_csv(out_fn, na_rep='NaN', date_format='%Y,%m,%d,%H,%M,%S')

def export_to_hdf(seg):
    obj = pd.DataFrame(seg.pf, columns=seg.pf[0].keys())
    out_fn = 'PF_{}_{}.h5'.format(seg.meta['start_date'].isoformat(),
                                    seg.meta['end_date'].isoformat())

    obj.to_hdf(out_fn, 'fixed')

def split_and_export(seg):
    df = pd.DataFrame(seg.pf, columns=seg.pf[0].keys())
    new_df = pd.DataFrame()
    new_df['date'] = df['date']
    for array in df.select_dtypes(include=['int64']):
        new_df[array] = df[array]
    for array in df.select_dtypes(include=['object']):
        if array != 'date':
            new_df[array + '_v'] = pd.Series([x.v for x in df[array].values])
            new_df[array + '_u'] = pd.Series([x.u for x in df[array].values])

    out_fn = 'PF_{}_{}.csv'.format(seg.meta['start_date'].isoformat(),
                                    seg.meta['end_date'].isoformat())

    new_df.to_csv(out_fn, na_rep='NaN', date_format='%Y,%m,%d,%H,%M,%S')

def main():
    global d1, d2, instr

    parse_args()

    if    d1 == None:    d1 = input('Enter starting date: ')
    if    d2 == None:    d2 = input('Enter end date: ')
    if instr == None: instr = input('Enter the instrument: ')
    if zaw_util.data_root == '.': zaw_util.data_root = input('Enter data root:')

    # Initialize the segment data first.
    segment = PFData()
    init(segment)

    # Proceed with function calls to read and calculate data.
    date_c = segment.meta['start_date']

    process(segment, date_c) 

    export_to_hdf(segment)    

if __name__ == "__main__":
    main()