"""
Provides the functions necessary for analyzing cross calibration
between instruments.
"""
import datetime as dt
import getopt
import itertools
import random
import sys

import numpy as np
import psycopg2 as psy
from scipy.interpolate import griddata

import quadrangles as quad
import zaw_util as z
from coord import CRD

psy.extensions.register_adapter(np.float32, psy._psycopg.AsIs)
DEC2FLOAT = psy.extensions.new_type(
    psy.extensions.DECIMAL.values,
    'DEC2FLOAT',
    lambda value, curs: float(value) if value is not None else None)
psy.extensions.register_type(DEC2FLOAT)

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

def process_instruments(i1, i2, par):
    """
    Returns a list of valid file combinations.

    Searches for files within 24 hours of each other between instruments.
    This time default can be changed with the timeTolerance keyword.
    If retDates is set to true, it will also return a list of dates where
    matches were found.
    """
    tol1 = par['t1']
    tol2 = par['t2']

    instrumentKey = {'512': 1, 'SPMG': 2, 'MDI': 3, 'HMI': 4}

    conn = z.load_database()
    cur = conn.cursor()
    cur.execute("SELECT a.filepath AS f1, b.filepath AS f2 \
            FROM file_time_diff main \
            JOIN file a ON main.file1 = a.id \
            JOIN file b ON main.file2 = b.id \
            WHERE a.instrument = %s AND b.instrument = %s \
            AND difference BETWEEN INTERVAL %s \
            AND INTERVAL %s;", (instrumentKey[i1.upper()],
            instrumentKey[i2.upper()], tol1, tol2))
    
    results = cur.fetchall()
    cur.close()
    conn.close()

    return results

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

def fix_longitude(f1, f2, raw_remap=False):
    """
    Will shift the longitude of second magnetogram to match the first.

    Uses differential rotation to update the second magnetogram's longitude.
    Then, afterwards it will shift the longitudes to match the first,
    considering they will only be off by some constant.
    """

    print(f1)
    print(f2)
    mgnt1 = CRD(f1)
    mgnt2 = CRD(f2)
    print(mgnt1.im_raw.date)
    print(mgnt2.im_raw.date)
    mgnt1.heliographic()
    mgnt2.heliographic()
    mgnt1.magnetic_flux()
    mgnt2.magnetic_flux()
    #Apply differential Rotation
    if mgnt2.im_raw.dimensions[0].value > mgnt1.im_raw.dimensions[0].value:
        rotation = z.diff_rot(mgnt2, mgnt1)
        mgnt1.lonhRot = mgnt1.lonh + rotation.value
        interpolate_remap(mgnt2, mgnt1, raw_remap)
        return mgnt2, mgnt1
    else:
        rotation = z.diff_rot(mgnt1, mgnt2)
        mgnt2.lonhRot = mgnt2.lonh + rotation.value
        interpolate_remap(mgnt1, mgnt2, raw_remap)
        return mgnt1, mgnt2
    
def interpolate_remap(m1, m2, raw=False):
    if raw:
        v2 = m2.im_raw.data.flatten()
    else:
        v2 = m2.im_corr.v.flatten()

    x2 = m2.lonhRot.v.flatten()
    y2 = m2.lath.v.flatten()
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
    new_m2[m2.rg > m2.par['rsun']*np.sin(75.0*np.pi/180)] = np.nan

    m2.remap = new_m2

def run_multiple_n(m):
    """Takes mgnt and returns list of different fragmented quadrangles."""
    nList = [i for i in range(10, 3100, 100)]

    n_dict_length = {}

    for n in nList:
        N = quad.fragment_single(m, n)
        n_dict_length[n] = len(N)
    return n_dict_length

def upload_quadrangles(conn, b, workingFiles):
    f1, f2 = get_file_id(conn, workingFiles)
    cur = conn.cursor()
    try:
        for quad in b:
            if np.isnan(quad.fluxDensity) or np.isnan(quad.fluxDensity2):
                continue
            cur.execute("INSERT INTO quadrangle\
                (referencemag, secondarymag, diskangle, area,\
                referencefluxdensity, secondaryfluxdensity, fragmentationvalue)\
                VALUES\
                (%s, %s, %s, %s, %s, %s, %s)",
                (f1, f2, np.float32(quad.diskAngle.v), quad.area.v,
                quad.fluxDensity, quad.fluxDensity2, quad.fragmentationValue))

        cur.execute("INSERT INTO uniquepairs\
                VALUES (%s, %s, %s)", (f1, f2, quad.fragmentationValue))
    except:
        print("Could not upload completely to database.")
        conn.rollback()
        cur.close()
        return
       
    conn.commit()
    cur.close()

def compare_day(i1, i2, par, filePair):
    """
    Fully autonomous magnetogram comparison function.

    Can input two instruments for a random sampling of their matches.
    If date is entered, it will compare magnetograms for that date.
    f1 and f2 are filenames that bypass all of this
    """
    if filePair is not None:
        files = filePair
    else:
        files = random.choice(process_instruments(i1, i2, par))
    try:
        m1, m2 = fix_longitude(files[0], files[1])
    except ValueError:
        raise
    blocks_n = quad.fragment_multiple(m1, m2, par['n'])

    return blocks_n

def get_instruments():
    global i1, i2
    i1 = input("Enter an instrument: ")
    i2 = input("Enter a second instrument: ")

def get_file_id(conn, files):
    cur = conn.cursor()
    cur.execute("SELECT id FROM file WHERE filepath = %s", (files[0],))
    fileID1 = cur.fetchone()[0]
    cur.execute("SELECT id FROM file WHERE filepath = %s", (files[1],))
    fileID2 = cur.fetchone()[0]
    cur.close()

    return fileID1, fileID2

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
    conn = z.load_database()

    while True:
        option = input("Choose a function: (r)andom [num], (s)elect date, switch (i)nstruments, (e)xit: ")
        if option=='i':
            get_instruments()
        elif 'r' in option:
            try:
                passes = int(option.split()[-1])
            except ValueError:
                passes = 1
            n = int(input("Enter segmentation level: "))
            tol1 = input("Enter minimum time: ")
            tol2 = input("Enter maximum time: ")
            params = {'n': n, 't1': tol1, 't2': tol2}
            fileMatches = process_instruments(i1, i2, params)
            for i in range(min(len(fileMatches),passes)):
                try:
                    choiceInt = int(random.random()*len(fileMatches))
                    workingFiles = fileMatches[choiceInt]
                    fileIDs = get_file_id(conn, workingFiles)
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM uniquepairs\
                            WHERE referencemag = %s \
                            AND secondarymag = %s\
                            AND fragmentationvalue = %s", (fileIDs[0], fileIDs[1], n))
                    if cur.fetchone() is not None:
                        cur.close()
                        continue
                    cur.close()
                    b = compare_day(i1, i2, params, workingFiles)
                    del fileMatches[choiceInt] #So we don't hit it again
                except ValueError:
                    continue
                upload_quadrangles(conn, b, workingFiles)
        elif 'rs' in option:
            """Special option for only selecting unique days, not pairs"""
            try:
                passes = int(option.split()[-1])
            except ValueError:
                passes = 1
            n = int(input("Enter segmentation level: "))
            tol1 = input("Enter minimum time: ")
            tol2 = input("Enter maximum time: ")
            params = {'n': n, 't1': tol1, 't2': tol2}
            fileMatches = process_instruments(i1, i2, params)
            dayMatches = []
            i = 0
            while i < passes:
                try:
                    choiceInt = int(random.random()*len(fileMatches))
                    workingFiles = fileMatches[choiceInt]
                    fileIDs = get_file_id(conn, workingFiles)
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM uniquepairs\
                            WHERE referencemag = %s \
                            AND secondarymag = %s\
                            AND fragmentationvalue = %s", (fileIDs[0], fileIDs[1], n))
                    if cur.fetchone() is not None:
                        cur.close()
                        continue
                    cur.execute("SELECT date FROM file WHERE id = %s OR id = %s", (fileIDs[0], fileIDs[1]))
                    if (cur.fetchone()[0].toordinal(), cur.fetchone()[0].toordinal()) in dayMatches:
                        continue
                    dayMatches.append((cur.fetchone()[0].toordinal(), cur.fetchone()[0].toordinal()))
                    cur.close()
                    b = compare_day(i1, i2, params, workingFiles)
                    del fileMatches[choiceInt] #So we don't hit it again
                except ValueError:
                    continue
                upload_quadrangles(conn, b, workingFiles)
                i += 1
        elif 'e' in option:
            break
    return

if __name__ == "__main__":
    main()
