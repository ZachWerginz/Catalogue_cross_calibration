# This is a script that is meant to look for magnetograms across instruments
# that are close in time and blocks them into latitude/longitude areas.

import zaw_util as z
import datetime as dt
import itertools
from uncertainty import Measurement as M
from astropy.io import fits
import sunpy.time
import getopt
import sys

i1 = None       #Instrument 1
i2 = None       #Instrument 2
date = None     #Date to be examined

def usage():
    print('Usage: cross_calibration.py [-d data-root] [-f instrument 1] [-s instrument 2] [-i date]')

def parse_args():
    global i1, i2, date

    try:
        opts, args = getopt.getopt(
                sys.argv[1:],
                "d:f:s:i:", 
                ["data-root=", "instrument-1=", "instrument-2=", "date="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--data-root"):
            z.data_root = arg
        elif opt in ("-f", "--first-instr"):
            i1 = arg
        elif opt in ("-s", "--second-instr"):
            i2 = arg
        elif opt in ("-i", "--date"):
            date = sunpy.time.parse_time(arg)
        else:
            assert False, "unhandled option"

def date_defaults(instr):
    if instr == '512':
        return dt.datetime(1976, 1, 5), dt.datetime(1993, 4, 9)
    elif instr == 'spmg':
        return dt.datetime(1992, 4, 21), dt.datetime(1999, 12, 30)
    elif instr == 'mdi':
        return dt.datetime(1996, 4, 15), dt.datetime(2011, 4, 11)
    elif instr == 'hmi':
        return dt.datetime(2010, 4, 8), dt.datetime(2016, 7, 5)
    else:
        raise ValueError('Unrecognized instrument')

def find_match(date, f_list):
    best = f_list[0]
    bestTimeDelta = dt.timedelta(1)  #48 hour max

    for f in f_list:
        timeDelta = date - get_header_date(f)
        if abs(timeDelta) < abs(bestTimeDelta):
            bestTimeDelta = timeDelta
            best = f

    return best

def get_header_date(f):
    hdulist = fits.open(f)

    for hdu in hdulist:
        try:
            time = sunpy.time.parse_time(hdu.header['DATE_OBS'])
        except KeyError:
            try:
                time = sunpy.time.parse_time(hdu.header['DATE-OBS'])
            except KeyError:
                try:
                    time = sunpy.time.parse_time(hdu.header['T_OBS'])
                except:
                    continue
    hdulist.close()

    return time

def process_date(i1, i2, date):
    print(date.isoformat())
    try:
        fList1 = z.search_file(date, i1, auto=False)
        fList2 = z.search_file(date, i2, auto=False)
    except IOError:
        print("No match for date.")
        raise

    return list(itertools.product(fList1, fList2))

def process_instruments(i1, i2):
    start_i1, end_i1 = date_defaults(i1)
    start_i2, end_i2 = date_defaults(i2)

    start = max(start_i1, start_i2)
    end = min(end_i1, end_i2)
    date = start
    good_dates = []

    while date < end:
        try:
            process_date(i1, i2, date)
            good_dates.append(date)
        except IOError:
            continue
        finally:
            date += dt.timedelta(1)

    return good_dates

def lon_lat_indices(m, lat1, lat2, lon1, lon2):
    p = np.where((m.lath > lat1) & (m.lath < lat2)
            & (m.lonh > lon1) & (m.lonh < lon2))

    vp = np.where((m.lath > lat1) & (m.lath < lat2)
            & (m.lonh > lon1) & (m.lonh < lon2) & M.isfinite(m.im_corr))

    posp = np.where((m.lath > lat1) & (m.lath < lat2)
            & (m.lonh > lon1) & (m.lonh < lon2) & (m.im_corr > 0))

    negp = np.where((m.lath > lat1) & (m.lath < lat2)
            & (m.lonh > lon1) & (m.lonh < lon2) & (m.im_corr < 0))

    return p, vp, posp, negp

def main():
    global i1, i2
    parse_args()
    print(i1)
    print(i2)
    print(date)
    if    i1 == None:    i1 = input('Enter first instrument: ')
    if    i2 == None:    i2 = input('Enter second instrument: ')
    if date is not None:
        return process_date(i1, i2, date)
    else:
        #maybe return just a list of paired files, not dates?
        dates = process_instruments(i1, i2)
        return dates

if __name__ == "__main__":
    main()