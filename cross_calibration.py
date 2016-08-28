# This is a script that is meant to look for magnetograms across instruments
# that are close in time and blocks them into latitude/longitude areas.

import zaw_util as z
import datetime as dt
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

def find_match(date, f_list):
    best = f_list[0]
    bestTimeDelta = dt.timedelta(1)  #24 hour max

    for f in f_list:
        timeDelta = date - get_header_date(f)
        if abs(timeDelta) < abs(bestTimeDelta):
            bestTimeDelta = timeDelta
            best = f

    return best


def get_header_date(f):
    return sunpy.time.parse_time(fits.getval(f, 'DATE_OBS'))

def process_date(i1, i2, date):
    try:
        fList1 = z.search_file(date, i1, auto=False)
        fList2 = z.search_file(date, i2, auto=False)
    except IOError:
        print("No match for date.")
        return

    if len(fList1) < len(fList2):
        file1 = fList1[0]
    else:
        file1 = fList2[0]

    file2 = find_match(get_header_date(file1), fList2)

    return file1, file2

def main():
    parse_args()
    print(i1)
    print(i2)
    print(date)
    if date is not None:
        f1, f2 = process_date(i1, i2, date)
    else:
        return
        #process_instruments(i1, i2)
    return

if __name__ == "__main__":
    main()