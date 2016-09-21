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
import random
import sunpy.time
import getopt
import sys
import copy

i1 = None       #Instrument 1
i2 = None       #Instrument 2
date = None     #Date to be examined

class Block:

    def __init__(self, mgnt, indices, ID, c=random.random()):
        """Accepts heliographic bounds and a list of indices for block """
        self.mgnt = mgnt
        self.indices = indices
        self.id = ID
        self.pltColor = c
        self.lat = (self.min_latitude(), self.max_latitude())
        self.lon = (self.min_longitude(), self.max_longitude())

    def min_latitude(self):
        return M.nanmin(self.mgnt.lath[self.indices])

    def max_latitude(self):
        return M.nanmax(self.mgnt.lath[self.indices])

    def min_longitude(self):
        return M.nanmin(self.mgnt.lonh[self.indices])

    def max_longitude(self):
        return M.nanmax(self.mgnt.lonh[self.indices])

    def mean_field(self):
        return M.nanmean(self.mgnt.im_raw_u[self.indices])

    def sum_flux(self):
        return M.nansum(self.mgnt.mflux_corr[self.indices])

    def sum_area(self):
        return M.nansum(self.mgnt.area[self.indices])


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
    m1.magnetic_flux()
    m2.magnetic_flux()
    m1.lonh += 360
    m2.lonh += 360
    #Apply differential Rotation
    rotation = z.diff_rot(m1, m2)
    m2.lonhRot = rotation.value + m2.lonh

    #Compare reference pixels
    # x1 = int(np.around(m1.X0))
    # y1 = int(np.around(m1.Y0))
    # x2 = int(np.around(m2.X0))
    # y2 = int(np.around(m2.Y0))

    # lonDelta = m2.lonhRot[x2, y2] - m1.lonh[x1, y1]
    # m2.lonhOld = m2.lonh
    # m2.lonh = m2.lonhRot - lonDelta

    return m1, m2

def fragment_single(mgnt, n):
    """
    Takes in a magnetogram and n value and returns a list of block objects.

    The magnetogram must have heliographic information stored (lonh and lath).
    """
    print("Processing {}".format(n))
    fragInfo = get_fragmentation_info(mgnt, n)
    return fragmentation_loop(fragInfo)

def fragment_multiple(m1, m2, n):
    """
    Takes in two magnetograms and n value and returns a list of block objects.

    The magnetograms must have heliographic information stored (lonh and lath).

    """
    if m1.X0 < m2.X0:
        refmgnt = m1
        secmgnt = m2
    else:
        refmgnt = m2
        secmgnt = m1
    print("Processing {}".format(n))
    refFragInfo = get_fragmentation_info(refmgnt, n)
    secFragInfo = get_fragmentation_info(secmgnt, n, refFragInfo['lonBands'])
    refBlocks, secBlocks = fragmentation_loop(refFragInfo, secFragInfo)
    
    return refBlocks, secBlocks

def get_fragmentation_info(m, n, lonBands=None):
    m.lonh[m.rg > m.rsun*np.sin(85.0*np.pi/180)] = np.nan
    m.lath[m.rg > m.rsun*np.sin(85.0*np.pi/180)] = np.nan
    flatten(m)

    # This supports reusing latitude and longitude bands for other mgnts.
    latBands = split(n)
    if lonBands is None:
        lonBands = split(n, M.nanmin(m.lonh), M.nanmax(m.lonh))
    lonMasks = []

    for lon in lonBands:
        lonMasks.append(m.lonh_1d > lon)

    fragInfo = {'mgnt': m, 'latBands': latBands, 'lonBands': lonBands, 
            'lonMasks': lonMasks}
    return fragInfo

def fragmentation_loop(refFragInfo, secFragInfo=None):
    """
    The main loop that finds valid indices for lat/lon conditions.

    Input a dictionary of information for a single magnetogram, or
    two if you want to compare two with the same heliographic bounds.
    This dictionary must have the following valid keys:

    mgnt: The magnetogram with flattened indices and heliographic information.
    latBands: The interval space of latitudes to be used.
    lonBands: The interval space of longitudes to be used.
    lonMasks: The list of boolean masks for valid indices over lonBand space.
    """
    blocks = []
    mgnt = refFragInfo['mgnt']
    latBands = refFragInfo['latBands']
    lonBands = refFragInfo['lonBands']
    lonMasks = refFragInfo['lonMasks']
    l = int(mgnt.im_raw.dimensions[0].value)
    n = len(lonBands) - 1

    if secFragInfo is not None:
        secBlocks = []
        secmgnt = secFragInfo['mgnt']
        secLatBands = secFragInfo['latBands']
        secLonBands = secFragInfo['lonBands']
        secLonMasks = secFragInfo['lonMasks']
        secCurrLat = (secmgnt.lath_1d > latBands[0])
        secL = int(secmgnt.im_raw.dimensions[0].value)

    currLat = (mgnt.lath_1d > latBands[0])
    for i in range(n):
        nextLat = (mgnt.lath_1d > latBands[i + 1])
        latitudeSetDiff = currLat*~nextLat
        if ~(latitudeSetDiff.any()):
            continue
        minLon = np.nanmin(mgnt.lonh_1d[latitudeSetDiff])
        maxLon = np.nanmax(mgnt.lonh_1d[latitudeSetDiff])

        latBounds = (latBands[i], latBands[i + 1])
        areaRatio = calc_area_ratio(latBands, lonBands, latBounds)
        s = max(np.searchsorted(lonBands, minLon) - 1, 0)
        e = np.searchsorted(lonBands, maxLon)
        skip = int(round(areaRatio))

        if secFragInfo is not None:
            secNextLat = (secmgnt.lath_1d > latBands[i + 1])
            secLatitudeSetDiff = secCurrLat*~secNextLat

        for j in range(s, e, skip):
            uuid = str(i) + str(j) + str(n)
            try:
                blockBool = lonMasks[j]*~lonMasks[j+skip]*latitudeSetDiff
            except IndexError:
                blockBool = lonMasks[j]*~lonMasks[-1]*latitudeSetDiff
            if ~(blockBool.any()):
                continue
            else:
                c = random.random()
                blockInd = transform_indices(mgnt.ind_1d[blockBool], l)
                x = Block(mgnt, blockInd, uuid, c)
                blocks.append(x)
            if secFragInfo is not None:
                try:
                    secBlockBool = secLonMasks[j]*~secLonMasks[j+skip]*secLatitudeSetDiff
                except IndexError:
                    secBlockBool = secLonMasks[j]*~secLonMasks[-1]*secLatitudeSetDiff
                if ~(secBlockBool.any()):
                    continue
                secBlockInd = transform_indices(secmgnt.ind_1d[secBlockBool], secL)
                y = Block(secmgnt, secBlockInd, uuid, c)
                secBlocks.append(y)
        currLat = nextLat
        if secFragInfo is not None:
            secCurrLat = secNextLat

    if secFragInfo is not None:
        return blocks, secBlocks
    else:
        return blocks

def split(n, minimum=-90, maximum=90):
    """Returns an interval space based on n number of blocks."""
    return np.linspace(minimum, maximum, n + 1)

def flatten(m):
    """
    Takes in a magnetogram and adds 1 dimensional flattened
    attributes for heliographic information.
    """

    m.lonh_1d = m.lonh.v.flatten()
    m.lath_1d = m.lath.v.flatten()
    m.ind_1d = np.arange(np.size(m.lath_1d))

    ind = np.where(M.isfinite(m.lonh_1d))

    m.lonh_1d = m.lonh_1d[ind]
    m.ind_1d = m.ind_1d[ind]
    m.lath_1d = m.lath_1d[ind]

def calc_area_ratio(latBands, lonBands, latBounds):
    """
    Calculates the area ratio of the given latitude band.

    The middle square solid angle is the centermost block of the sun.
    This is used for reference because arguably it is the biggest block given
    certain lat/lon bounds. This is used in skipping n values for a more
    comprehensive approach to enlarging the blocks near the poles for area
    similarities.
    """
    n = len(latBands) - 1
    lonBlockDiff = lonBands[n//2 + 1] - lonBands[n//2] 

    middleSquareSA = (np.sin(latBands[n//2 + 1]*np.pi/180) - np.sin(latBands[n//2]*np.pi/180))*lonBlockDiff
    sa = (np.sin(latBounds[1]*np.pi/180) - np.sin(latBounds[0]*np.pi/180))*lonBlockDiff
    areaRatio = middleSquareSA/sa

    return areaRatio

def transform_indices(ind, l):
    """Takes a 1D list of indices and calculates a 2D list of indices."""

    cols = ind % l
    rows = ind // l
    return (rows, cols)

def block_area(mgnt, blocks):
    """Given a list of blocks, will calculate and print out areas."""

    areas = []
    for block in blocks:
        try:
            areas.append(block.sum_area())
        except:
            areas.append(M(np.nan, np.nan))

    return areas

def block_field(mgnt, blocks, raw=False):
    """Given a list of blocks, will calculate and print out areas."""

    field = []
    for block in blocks:
        try:
            field.append(block.mean_field())
        except:
            field.append(M(np.nan, np.nan))
    return field

def block_flux(mgnt, blocks):
    """Given a list of blocks, will calculate and print out areas."""

    flux = []
    for block in blocks:
        try:
            flux.append(block.sum_flux())
        except:
            flux.append(M(np.nan, np.nan))

    return flux

def block_plot(*args):
    """Given a list of blocks, will plot a nice image differentiating them."""
    n = len(args)
    rows = int(round(np.sqrt(n)))
    cols = int(np.ceil(np.sqrt(n)))
    im = {}
    ax = {}
    for i in range(len(args)):
        if isinstance(args[i], type(list())):
            im[i] = args[i][0].mgnt.lonh.v.copy()
            for x in args[i]:
                im[i][x.indices] = x.pltColor
            ax[i] = plt.subplot2grid((rows, cols), (i%rows, i//rows))
            ax[i].imshow(im[i], vmin=0, vmax=1)
            ax[i].set_title(args[i][0].mgnt.im_raw.date)
        else:
            im[i] = args[i]
            ax[i] = plt.subplot2grid((rows, cols), (i%rows, i//rows))
            ax[i].imshow(im[i], cmap='binary')
    plt.show()

def run_multiple_n(m):
    nList = [i for i in range(10, 3100, 100)]

    n_dict_length = {}

    for n in nList:
        N = fragment(m, n)
        n_dict_length[n] = len(N)
    return 

def scatter_plot(dict1, dict2, separate=False):
    i = 1
    for n in sorted([x for x in dict1.keys() if x != 'instr']):
        if separate:
            plt.subplot(len(dict1), 1, i)
            plt.xlabel(dict1['instr'])
            plt.ylabel(dict2['instr'])
            plt.title('Number of blocks: {}'.format(n))
        plt.plot([np.nanmin(dict1[n]),np.nanmax(dict1[n])],[np.nanmin(dict1[n]),np.nanmax(dict1[n])], 'k-', lw=2)
        plt.plot(dict1[n], dict2[n], '.')
        i += 1
    plt.show()

def calc_n_lists(m, block):
    ar = {}
    flx = {}
    f = {}
    for key, value in block.items():
        ar[key], flx[key], f[key] = calc_list_parameters(m, value)
    ar['instr'] = m.im_raw.instrument
    flx['instr'] = m.im_raw.instrument
    f['instr'] = m.im_raw.instrument
    return ar, flx, f

def calc_list_parameters(m, blockList):
    ar = block_area(m, blockList)
    flx = block_flux(m, blockList)
    f = block_field(m, blockList)
    return ar, flx, f

def n_plot(n_dict):
    NList = []
    TList = []
    for key, value in n_dict.items():
        NList.append(key)
        TList.append(len(value))
    N = np.array(NList)
    T = np.array(TList)
    plt.plot(N, T, '.')

    plt.show()

def plot_instrument_comparisons():
    pass

def compare_day(i1, i2, n, date=None):
    if date is None:
        files = process_instruments(i1, i2)
    else:
        date = sunpy.time.parse_time(date)
        files = process_date(i1, i2, date)
        prompt = "Enter another date, CTRL-C to cancel: "
        while files == []:
            try:
                date = sunpy.time.parse_time(input(prompt))
            except KeyboardInterrupt:
                break
            files = process_date(i1, i2, date)
    m1, m2 = fix_longitude(files[0][0], files[0][1])
    i1_n, bands = fragment(m1, n)
    i2_n = fragment(m2, n, bands[0], bands[1])[0]
    ar_i1, flx_i1, f_i1 = calc_list_parameters(m1, i1_n)
    ar_i2, flx_i2, f_i2 = calc_list_parameters(m2, i2_n)


def main():
    global i1, i2
    parse_args()
    files = process_instruments('spmg', 'mdi')
    m1, m2 = fix_longitude(files[0][0], files[0][1])
    fragment_multiple(m1, m2, 9)
    

if __name__ == "__main__":
    main()