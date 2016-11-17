"""
Provides the functions necessary for analyzing cross calibration
between instruments.
"""
import zaw_util as z
from zaw_coord import CRD
import quadrangles as quad
import block_plot as b
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from uncertainty import Measurement as M
import random
import sunpy.time
import getopt
import sys
from timeparse import timeparse

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]


i1 = None       #Instrument 1
i2 = None       #Instrument 2

def usage():
    print('Usage: cross_calibration.py [-d data-root] [-f instrument 1] [-s instrument 2] [-t tolerance]')

def parse_args():
    """Sets global variables for scripting purposes."""
    global i1, i2, date, tol

    try:
        opts, args = getopt.getopt(
                sys.argv[1:],
                "d:f:s:t:", 
                ["data-root=", "instrument-1=", "instrument-2=", "tol="])
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
        elif opt in ("-t", "--tol"):
            tol2 = int(arg)
        else:
            assert False, "unhandled option"

def process_date(i1, i2, date, timeTolerance=1):
    """Inputs two instruments and returns unique list of dates for a date."""
    fList1 = []
    fList2 = []

    #Include loop to include 24 hour time tolerance
    for i in range(0 - timeTolerance, 1 + timeTolerance):
        try:
            f1 = z.search_file(date + dt.timedelta(i), i1, auto=False)
            f2 = z.search_file(date, i2, auto=False)
            if f1 == f2 and i1 != 'mdi' and i2 != 'mdi':
                continue
            else:
                fList1.extend(f1)
                fList2.extend(f2)
        except IOError:
            continue
    result = list(set(itertools.product(fList1, fList2)))
    result = [x for x in result if x[0] != x[1]]

    if not result:
        raise IOError

    return result

def process_instruments(i1, i2, par, retDates=False):
    """
    Returns a list of valid file combinations.

    Searches for files within 24 hours of each other between instruments.
    This time default can be changed with the timeTolerance keyword.
    If retDates is set to true, it will also return a list of dates where
    matches were found.
    """
    # if timeTolerance == 1 and retDates:
    #     return z.load_instrument_overlap(i1, i2)
    # elif timeTolerance == 1 and not retDates:
    #     return z.load_instrument_overlap(i1, i2)[0]
    # start_i1, end_i1 = z.date_defaults(i1)
    # start_i2, end_i2 = z.date_defaults(i2)

    # start = max(start_i1, start_i2)
    # end = min(end_i1, end_i2)
    # date = start
    # files = []
    # dates = []

    # while date < end:
    #     try:
    #         files.extend(process_date(i1, i2, date, timeTolerance))
    #         dates.append(date)
    #     except IOError:
    #         continue
    #     finally:
    #         date += dt.timedelta(1)

    # if retDates:
    #     return list(set(files)), dates
    # else:
    #     return list(set(files))
    tol1 = par['t1']
    tol2 = par['t2']

    if i1 == '512': i1 = 'kpvt'

    df = z.load_match_database()
    firstPass = df[
        (df['File 1'].str.contains(i1.upper()) & df['File 2'].str.contains(i2.upper()))
        | (df['File 1'].str.contains(i2.upper()) & df['File 2'].str.contains(i1.upper()))]
    secondPass = firstPass[(firstPass['Time Difference'] < tol2) & (firstPass['Time Difference'] > tol1)]
    res = [(x, y) for x, y in zip(secondPass['File 1'].values, secondPass['File 2'].values)]

    if retDates:
        dates = [x for x in secondPass['Date 1']]
        return res, dates
    else:
        return res

def coordinate_compare(i1, i2):
    """
    A function used to compare two instruments longitude stats.

    Initially used for debugging, but now deprecated."""
    filePairs = process_instruments(i1, i2)
    pairInfo = {}
    infoList = []

    for pair in filePairs:
        print(pair)

    for pair in filePairs:
        instr1 = CRD(pair[0])
        instr2 = CRD(pair[1])
        instr1.heliographic()
        instr2.heliographic()
        ind1 = np.where((instr1.lath < 30) & (instr1.lath > -30))
        ind2 = np.where((instr2.lath < 30) & (instr2.lath > -30))
        pairInfo['Instrument1'] = pair[0]
        pairInfo['Instrument2'] = pair[1]
        pairInfo['timeDelta'] = abs(instr1.im_raw.date - instr2.im_raw.date)
        pairInfo['lonMax1'] = np.nanmax(instr1.lonh.v[ind1])
        pairInfo['lonMin1'] = np.nanmin(instr1.lonh.v[ind1])
        pairInfo['lonMax2'] = np.nanmax(instr2.lonh.v[ind2])
        pairInfo['lonMin2'] = np.nanmin(instr2.lonh.v[ind2])
        pairInfo['deltaLon1'] = pairInfo['lonMax1'] - pairInfo['lonMin1']
        pairInfo['deltaLon2'] = pairInfo['lonMax2'] - pairInfo['lonMin2']
        pairInfo['latMax1'] = np.nanmax(instr1.lath.v)
        pairInfo['latMin1'] = np.nanmin(instr1.lath.v)
        pairInfo['latMax2'] = np.nanmax(instr2.lath.v)
        pairInfo['latMin2'] = np.nanmin(instr2.lath.v)
        pairInfo['deltaLat1'] = pairInfo['latMax1'] - pairInfo['latMin1']
        pairInfo['deltaLat2'] = pairInfo['latMax2'] - pairInfo['latMin2']
        infoList.append(pairInfo.copy())
        if len(infoList) > 100: break

    return infoList

def fix_longitude(f1, f2):
    """
    Will shift the longitude of second magnetogram to match the first.

    Uses differential rotation to update the second magnetogram's longitude.
    Then, afterwards it will shift the longitudes to match the first,
    considering they will only be off by some constant.
    """

    m1 = CRD(f1)
    m2 = CRD(f2)
    print(m1.im_raw.date)
    print(m2.im_raw.date)
    m1.heliographic()
    m2.heliographic()
    # Don't need right away for looking at consistency
    # m1.magnetic_flux()
    # m2.magnetic_flux()
    #Apply differential Rotation
    rotation = z.diff_rot(m1, m2)
    m2.lonhOld = m2.lonh
    m2.lonh = rotation.value + m2.lonh

    return m1, m2

def run_multiple_n(m):
    """Takes mgnt and returns list of different fragmented quadrangles."""
    nList = [i for i in range(10, 3100, 100)]

    n_dict_length = {}

    for n in nList:
        N = quad.fragment_single(m, n)
        n_dict_length[n] = len(N)
    return n_dict_length

def compare_day(i1, i2, par, f1=None, f2=None):
    """
    Fully autonomous magnetogram comparison function.

    Can input two instruments for a random sampling of their matches.
    If date is entered, it will compare magnetograms for that date.
    f1 and f2 are filenames that bypass all of this
    """
    if f1 is not None or f2 is not None:
        files = (f1, f2)
    else:
        files = random.choice(process_instruments(i1, i2, par))

    m1, m2 = fix_longitude(files[0], files[1])
    i1_n, i2_n = quad.fragment_multiple(m1, m2, par['n'])
    p_i1 = quad.calc_block_parameters(m1, i1_n, uncertainty=True)
    p_i2 = quad.calc_block_parameters(m2, i2_n, uncertainty=True)

    return (i1_n, i2_n), (p_i1, p_i2)

def get_instruments():
    global i1, i2
    i1 = input("Enter an instrument: ")
    i2 = input("Enter a second instrument: ")

def select_pair(par):
    """Guides user through valid date combinations and returns filepaths."""
    files, dates = process_instruments(i1, i2, par, True)
    print(set([x.year for x in dates]))
    y = input("Choose a year: ")
    print(set([x.month for x in dates if x.year == int(y)]))
    m = input("Choose a month: ")
    print(set([x.day for x in dates if x.year == int(y) and x.month == int(m)]))
    d = input("Choose a day: ")
    t = [x for x in dates if x.year == int(y) and x.month == int(m) and x.day == int(d)][0]
    files = process_date(i1, i2, t, 1)
    k = 0
    for line in files:
        print("{}: ({}, {})".format(k, line[0].split('\\')[-1], line[1].split('\\')[-1]))
        k += 1
    choice = int(input("Select a pair: "))
    return files[choice]

def del_bad(p, d):
    bad = []
    for i in range(len(p)):
        if len(np.where(p[i][1][2]/p[i][0][2] > 100)[0]) > 10:
            bad.append(i)
            continue
        if len(np.where(p[i][0][2]/p[i][1][2] > 100)[0]) > 10:
            bad.append(i)

    for i in sorted(bad, reverse=True):
        del p[i]
        del d[i]


def main():
    """
    Main loop guiding the user though different cross_calibration
    visualization options. For each magnetogram pair chosen, the
    list of quadrangles and their parameters will be added to 
    bl and p respectively.

    --Options--
    r [num]:    will choose a random magnetogram num times out of the 
                instrument overlap
    s:          user can choose specific date pair
    i:          allows user to switch instruments
    h:          will plot all of the p data held thus filePairs
    e:          exit the program, or return bl and p data if in python.
    """
    parse_args()
    if i1 is None or i2 is None:
        get_instruments()

    p = []
    bl = []
    d = []

    while True:
        try:
            option = input("Choose a function: (r)andom [num], (s)elect date, switch (i)nstruments, s(h)ow plots, (e)xit: ")
            if option=='i':
                get_instruments()
            elif 'r' in option:
                try:
                    passes = int(option.split()[-1])
                except ValueError:
                    passes = 1
                n = int(input("Enter segmentation level: "))
                tol1 = dt.timedelta(seconds=timeparse(input("Enter minimum time: ")))
                tol2 = dt.timedelta(seconds=timeparse(input("Enter maximum time: ")))
                params = {'n': n, 't1': tol1, 't2': tol2}
                for i in range(passes):
                    x, y = compare_day(i1, i2, params)
                    #bl.append(x)
                    try:
                        d.append((x[0][0].mgnt.im_raw.date, x[1][0].mgnt.im_raw.date))
                    except IndexError as e:
                        print(e)
                        continue
                    p.append(y)
                bl.append(x)
            elif 's' in option:
                tol1 = dt.timedelta(seconds=timeparse(input("Enter minimum time: ")))
                tol2 = dt.timedelta(seconds=timeparse(input("Enter maximum time: ")))
                n = int(input("Enter segmentation level: "))
                params = {'n': n, 't1': tol1, 't2': tol2}
                file1, file2 = select_pair(params)
                x, y = compare_day(i1, i2, params, f1=file1, f2=file2)
                bl.append(x), p.append(y)
            elif 'h' in option:
                b.plot_block_parameters(*p)
                if len(p) == 1:
                    b.block_plot()
                plt.show()
            elif 'e' in option:
                break
        except Exception as e:
            print(e)
            continue
    return bl, p, d

if __name__ == "__main__":
    main()