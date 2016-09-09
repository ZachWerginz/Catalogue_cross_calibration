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

class Block:

    def __init__(self, mgnt, latitude_bounds, longitude_bounds):
        """Accepts heliographic bounds and a list of indices for block """
        self.mgnt = mgnt
        self.lat = latitude_bounds
        self.lon = longitude_bounds
        self.indices = self.calculate_indices()

    def min_latitude(self):
        return min(self.lat)

    def max_latitude(self):
        return max(self.lat)

    def min_longitude(self):
        return min(self.lon)

    def max_longitude(self):
        return max(self.lon)

    def calculate_indices(self):
        m = self.mgnt
        return np.where((m.lath.v > self.min_latitude()) 
                & (m.lath.v < self.max_latitude())
                & (m.lonh.v > self.min_longitude()) 
                & (m.lonh.v < self.max_longitude()))
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
    m1.magnetic_flux()
    m2.magnetic_flux()
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
    mgnt.lonh[mgnt.rg > mgnt.rsun*np.sin(85.0*np.pi/180)] = np.nan

    #Split magnetogram up into bands bounded by latitude
    bands = split(n)
    print(bands)

    #Now split magnetogram up further into blocks bounded by
    #latitude and longitude
    blocks = calculate_blocks(mgnt, bands, n)
    return blocks

def split(n, minimum=-90, maximum=90):
    """Returns an interval space of latitudes/longitudes based on n number of blocks."""
    return np.linspace(minimum, maximum, n + 1)

def calculate_blocks(mgnt, bands, n):
    blocks = []
    for lat in range(len(bands) - 1):
        minLat = bands[lat]
        maxLat = bands[lat + 1]
        new_n = np.round(n*abs(M.cos((minLat + maxLat)*np.pi/360)))
        print(new_n)
        if new_n < 1: new_n = 1
        ind = np.where((mgnt.lath < maxLat) & (mgnt.lath > minLat))
        if np.size(ind) == 0: continue
        lonBand = split(new_n, M.nanmin(mgnt.lonh[ind]), M.nanmax(mgnt.lonh[ind]))
        print(lonBand)
        for lon in range(len(lonBand) - 1):
            minLon = lonBand[lon]
            maxLon = lonBand[lon + 1]
            blocks.append(Block(mgnt, (minLat, maxLat), (minLon, maxLon)))

    return blocks

def calculate_param_from_block(mgnt, blocks):
    """Reads in a list of blocks, returns list of new blocks for other mgnt."""
    blocks2 = []
    for block in blocks:
        blocks2.append(Block(mgnt, block.lat, block.lon))
    return blocks2

def block_area(mgnt, blocks):
    """Given a list of blocks, will calculate and print out areas."""
    areas = []
    for block in blocks:
        try:
            areas.append(M.nansum(mgnt.area[block.indices]))
        except:
            areas.append(M(np.nan, np.nan))

    return areas

def block_field(mgnt, blocks, raw=False):
    """Given a list of blocks, will calculate and print out areas."""
    field = []
    for block in blocks:
        try:
            field.append(np.nanmean(mgnt.im_raw.data[block.indices]))
        except:
            field.append(M(np.nan, np.nan))
    return field

def block_flux(mgnt, blocks):
    """Given a list of blocks, will calculate and print out areas."""
    flux = []
    for block in blocks:
        try:
            flux.append(M.nansum(mgnt.mflux_corr[block.indices]))
        except:
            flux.append(M(np.nan, np.nan))

    return flux

def block_plot(mgnt, blocks):
    """Given a list of blocks, will plot a nice image differentiating them."""
    im = mgnt.lonh.v.copy()
    c = 0
    for block in blocks:
        try:
            im[block.indices] = c
            c += 1
        except:
            continue
        finally:
            print(block.lat)
            print(block.lon)

    plt.imshow(im, cmap='prism', vmin=0, vmax=M.nanmax(im))
    plt.show()

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

def create_ar_fl(m, block):
    ar = {}
    flx = {}
    f = {}
    for key, value in block.items():
        ar[key] = [x.v for x in c.block_area(m, value)]
        flx[key] = [x.v for x in c.block_flux(m, value)]
        f[key] = [x.v for x in c.block_field(m, value)]
    ar['instr'] = m.im_raw.instrument
    flx['instr'] = m.im_raw.instrument
    f['instr'] = m.im_raw.instrument
    return ar, flx, f

def main():
    global i1, i2
    parse_args()
    files = process_instruments('512', 'spmg')
    m1, m2 = fix_longitude(files[0][0], files[0][1])
    block1_dict = {}
    block2_dict = {}
    for n in [10, 30, 100]:
        block1_dict[n] = fragment(m1, n)
        block2_dict[n] = calculate_param_from_block(m2, block1_dict[n])

    return m1, m2, block1_dict, block2_dict



if __name__ == "__main__":
    main()