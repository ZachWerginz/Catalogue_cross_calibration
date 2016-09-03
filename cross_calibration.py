# This is a script that is meant to look for magnetograms across instruments
# that are close in time and blocks them into latitude/longitude areas.

import zaw_util as z
from zaw_coord import CRD
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from uncertainty import Measurement as M
from astropy import units as u
from astropy.io import fits
import sunpy.time
import getopt
import sys
import copy


i1 = None       #Instrument 1
i2 = None       #Instrument 2
date = None     #Date to be examined

class block(Object):

    def __init(self, bounds, indices):
        """Accepts heliographic bounds and a list of indices for block """
        self.bounds = bounds
        self.indices = indices

    def min_latitude(self):
        return bounds


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
        timeDelta = date - z.get_header_date(f)
        if abs(timeDelta) < abs(bestTimeDelta):
            bestTimeDelta = timeDelta
            best = f

    return best

def process_date(i1, i2, date):
    print(date.isoformat())

    fList1 = []
    fList2 = []

    #Include loop to include 24 hour difference
    for i in range(-1, 2):
        try:
            fList1.extend(z.search_file(date + dt.timedelta(i), i1, auto=False))
            fList2.extend(z.search_file(date, i2, auto=False))
        except IOError:
            continue


    return list(set(itertools.product(fList1, fList2)))

def process_instruments(i1, i2):
    """Returns a list of valid file combinations.

    Searches for files within 48 hours of each other between instruments.
    """
    start_i1, end_i1 = z.date_defaults(i1)
    start_i2, end_i2 = z.date_defaults(i2)

    start = max(start_i1, start_i2)
    end = min(end_i1, end_i2)
    date = start
    files = []

    while date < end:
        try:
            files.extend(process_date(i1, i2, date))
        except IOError:
            continue
        finally:
            date += dt.timedelta(1)

    return list(set(files))

def coordinate_compare(i1, i2):
    """A function used to compare two instruments longitude stats."""
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
    m1.heliographic()
    m2.heliographic()
    #Apply differential Rotation
    rotation = z.diff_rot(m1, m2)
    m2.lonhRot = rotation.value + m2.lonh

    #Compare reference pixels
    x1 = int(np.around(m1.X0))
    y1 = int(np.around(m1.Y0))
    x2 = int(np.around(m2.X0))
    y2 = int(np.around(m2.Y0))

    lonDelta = m2.lonhRot[x2, y2] - m1.lonh[x1, y1]
    m2.lonhOld = m2.lonh
    m2.lonh = m2.lonhRot - lonDelta

    return m1, m2

def fragment(mgnt, n):
    """
    Takes in a magnetogram and n value and returns a dictionary of block indices.

    The magnetogram must have heliographic information stored (lonh and lath).

    """
    try:
        mgnt.lonhShift[mgnt.rg > mgnt.rsun*np.sin(85.0*np.pi/180)] = np.nan
    except AttributeError:
        mgnt.lonh[mgnt.rg > mgnt.rsun*np.sin(85.0*np.pi/180)] = np.nan
    #Split magnetogram up into bands bounded by latitude
    bandLats = split(n)
    bandInd = get_latBand_indices(mgnt, bandLats)
    bands = {'ind': bandInd, 'lat': bandLats}
    #Now split magnetogram up further into blocks bounded by
    #latitude and longitude
    blocks = split_latitude_bands(mgnt, bands['lat'], bands['ind'])
    blocks['lat'] = bandLats
    return blocks

def split(n, minimum=-90, maximum=90):
    """Returns an interval space of latitudes/longitudes based on n number of blocks."""
    return np.linspace(minimum, maximum, n + 1)

def get_latBand_indices(mgnt, bandLats):
    """"""
    ind_dict = {}
    for i in range(len(bandLats) - 1):
        ind_dict[i] = np.where((mgnt.lath > bandLats[i]) & (mgnt.lath < bandLats[i+1]))

    return ind_dict

def get_lonBand_indices(mgnt, minLat, maxLat, lonBand):
    ind_dict = {}
    for i in range(len(lonBand) - 1):
        ind_dict[i] = np.where((mgnt.lath > minLat) & (mgnt.lath < maxLat)& (mgnt.lonh > lonBand[i]) & (mgnt.lonh < lonBand[i+1]))

    return ind_dict

def split_latitude_bands(mgnt, bandLats, bandInd=None):
    blocks = {'bands': {}, 'lat': {} }
    blockBands = {}

    if bandInd is None:
        bandInd = get_latBand_indices(mgnt, bandLats)

    k = len(bandInd)
    print(k)
    for bandNum, indices in bandInd.items():
        blockBands[bandNum] = {}
        if np.size(indices) == 0:
            blockBands[bandNum] = np.array([])
            continue 
        minLat = M.nanmin(mgnt.lath[indices])
        maxLat = M.nanmax(mgnt.lath[indices])
        print((minLat + maxLat)/2)
        print(M.cos((minLat + maxLat)*np.pi/360))
        new_n = np.round(k*abs(M.cos((minLat + maxLat)*np.pi/360)))
        if new_n < 1: new_n = 1
        print(new_n)
        lonBand = split(new_n ,M.nanmin(mgnt.lonh[indices]), M.nanmax(mgnt.lonh[indices]))
        print(lonBand)
        blockBands[bandNum] = {'ind': get_lonBand_indices(mgnt, minLat, maxLat, lonBand), 'lonbands': lonBand}

    blocks['bands'] = blockBands

    return blocks

def calculate_param_from_block(mgnt, blocks):
    """Reads in a block dictionary, calculates new indices of mgnt from intervals."""
    bandLats = blocks['lat']
    blocks2 = copy.deepcopy(blocks)
    for bandnum, band in blocks2['bands'].items():
        blocks2['bands'][bandnum]['ind'] = get_lonBand_indices(mgnt, blocks2['lat'][bandnum], blocks2['lat'][bandnum + 1] , blocks2['bands'][bandnum]['lonbands'])

    return blocks2

def block_area(mgnt, blocks):
    """Based on a block dictionary, will calculate and print out areas."""
    for keyi in blocks['bands'].keys():
        for keyj, indices in blocks['bands'][keyi]['ind'].items():
            print(M.nansum(mgnt.area[indices]))


def block_plot(mgnt, blocks):
    """Based on a block dictionary, will plot a nice image of the blocks."""
    im = mgnt.lonhShift.v
    c = 0
    for keyi in blocks['bands'].keys():
        try:
            for keyj, indices in blocks['bands'][keyi]['ind'].items():
                try:
                    im[indices] = c
                except:
                    continue
                finally:
                    c += 1
        except:
            continue

    plt.imshow(im, cmap='prism', vmin=0, vmax=M.nanmax(im))
    plt.show()

def main():
    global i1, i2
    parse_args()
    print(i1)
    print(i2)
    print(date)
    if    i1 == None:    i1 = input('Enter first instrument: ')
    if    i2 == None:    i2 = input('Enter second instrument: ')
    # if date is not None:
    #     return process_date(i1, i2, date)
    # else:
    #     return process_instruments(i1, i2)
    instrumentInfo = coordinate_compare(i1, i2)
    print(instrumentInfo)
    df = pd.DataFrame(instrumentInfo, columns=instrumentInfo[0].keys())
    df.to_csv('test')


if __name__ == "__main__":
    main()