from zaw_coord import CRD
from uncertainty import Measurement as M
import numpy as np
import random

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

info = True


class Quadrangle:
    """
    Quadrangle(mgnt, indices, ID)
    """

    def __init__(self, mgnt, i, ID, m2=None):
        """Accepts heliographic bounds and a list of indices for block """
        self.id = ID
        self.i1 = mgnt.im_raw.instrument
        self.date1 = mgnt.im_raw.date
        self.lat = (self.min_latitude(mgnt, i), self.max_latitude(mgnt, i))
        self.lon = (self.min_longitude(mgnt, i), self.max_longitude(mgnt, i))
        self.diskAngle = self.averageDA(mgnt, i)
        self.area = self.sum_area(mgnt, i)
        self.fluxDensity = self.mean_flux_density(mgnt.im_corr, i)
        if m2 is not None:
            self.fluxDensity2 = self.mean_flux_density(m2.remap, i)
            self.date2 = m2.im_raw.date
            self.i2 = m2.im_raw.instrument


    def min_latitude(self, m, ind):
        return M.nanmin(m.lath[ind])

    def max_latitude(self, m, ind):
        return M.nanmax(m.lath[ind])

    def min_longitude(self, m, ind):
        return M.nanmin(m.lonh[ind])

    def max_longitude(self, m, ind):
        return M.nanmax(m.lonh[ind])

    def averageDA(self, m, ind):
        """Returns the angular distance from disk center in degrees."""
        meanAngularRadius = M.nanmean(m.rg[ind])
        return M.arcsin(meanAngularRadius/m.rsun)*180/np.pi

    def mean_flux_density(self, arr, ind):
        """Returns the average uncorrected magnetic flux density."""
        return M.nanmean(arr[ind])

    def sum_area(self, m, ind):
        """Returns the total area covered by the quadrangle."""
        try:
            return M.nansum(m.area[ind])
        except AttributeError:
            m.eoa()
            return M.nansum(m.area[ind])

def fragment_single(mgnt, n):
    """
    Takes in a magnetogram and n value and returns a list of quadrangle objects.

    The magnetogram must have heliographic information stored (lonh and lath).
    """
    print("Processing {}".format(n))
    fragInfo = get_fragmentation_info(mgnt, n)
    return fragmentation_loop(fragInfo)

def fragment_multiple(m1, m2, n):
    """
    Takes in two magnetograms and n value and returns a list of quadrangle objects.

    The magnetograms must have heliographic information stored (lonh and lath).

    """
    refmgnt = m1
    secmgnt = m2

    printInfo("Processing {}".format(n))
    refFragInfo = get_fragmentation_info(refmgnt, n)
    blocks = fragmentation_loop(refFragInfo, secmgnt)
    
    return blocks

def get_fragmentation_info(m, n):
    """
    Inputs a mgnt and segmentation level (n) and outputs dict of parameters.

    This dictionary provides the fragmentation_loop with the requisite
    information such as latitude/longitude interval space.
    """

    m.lonh[m.rg > m.rsun*np.sin(85.0*np.pi/180)] = np.nan
    m.lath[m.rg > m.rsun*np.sin(85.0*np.pi/180)] = np.nan
    _flatten(m)

    # This supports reusing latitude and longitude bands for other mgnts.
    latBands = _split(n)
    lonBands = _split(n, M.nanmin(m.lonh), M.nanmax(m.lonh))
    lonMasks = []

    for lon in lonBands:
        lonMasks.append(m.lonh_1d > lon)

    fragInfo = {'mgnt': m, 'latBands': latBands, 'lonBands': lonBands, 
            'lonMasks': lonMasks}
    return fragInfo

def fragmentation_loop(refFragInfo, secmgnt=None):
    """
    The main loop that finds valid indices for lat/lon conditions.

    Input a dictionary of information for a single magnetogram, or
    two if you want to compare two with the same heliographic bounds.
    This dictionary must have the following valid keys:

    mgnt: The magnetogram with flattened indices and heliographic information.
    latBands: The interval space of latitudes to be used.
    lonBands: Them interval space of longitudes to be used.
    lonMasks: The list of boolean masks for valid indices over lonBand space.
    """

    blocks = []
    mgnt = refFragInfo['mgnt']
    latBands = refFragInfo['latBands']
    lonBands = refFragInfo['lonBands']
    lonMasks = refFragInfo['lonMasks']
    l = int(mgnt.im_raw.dimensions[0].value)
    n = len(lonBands) - 1

    currLat = (mgnt.lath_1d > latBands[0])
    for i in range(n):
        nextLat = (mgnt.lath_1d > latBands[i + 1])
        latitudeSetDiff = currLat*~nextLat
        if ~(latitudeSetDiff.any()):
            continue
        minLon = np.nanmin(mgnt.lonh_1d[latitudeSetDiff])
        maxLon = np.nanmax(mgnt.lonh_1d[latitudeSetDiff])

        latBounds = (latBands[i], latBands[i + 1])
        areaRatio = _calc_area_ratio(latBands, lonBands, latBounds)
        s = max(np.searchsorted(lonBands, minLon) - 1, 0)
        e = np.searchsorted(lonBands, maxLon)
        skip = int(round(areaRatio))

        for j in range(s, e, skip):
            uuid = str(i) + str(j) + str(n)
            try:
                blockBool = lonMasks[j]*~lonMasks[j+skip]*latitudeSetDiff
            except IndexError:
                blockBool = lonMasks[j]*~lonMasks[-1]*latitudeSetDiff
            if ~(blockBool.any()):
                continue
            else:
                blockInd = _transform_indices(mgnt.ind_1d[blockBool], l)
                x = Quadrangle(mgnt, blockInd, int(uuid), secmgnt)
                x.fragmentationValue = n
                blocks.append(x)
        currLat = nextLat

    return blocks

def _split(n, minimum=-90, maximum=90):
    """Returns an interval space based on n number of blocks."""
    return np.linspace(minimum, maximum, n + 1)

def _flatten(m):
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

def _calc_area_ratio(latBands, lonBands, latBounds):
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

def _transform_indices(ind, l):
    """Takes a 1D list of indices and calculates a 2D list of indices."""

    cols = ind % l
    rows = ind // l
    return (rows, cols)

def extract_valid_points(bl):
    """Extracts the valid points from the dictionary set."""

    flxD1 = bl['referenceFD']
    flxD2 = bl['secondaryFD']
    da = bl['diskangle']

    ind = np.where(np.logical_and( np.logical_or(np.abs(flxD1/flxD2) > 10, np.abs(flxD2/flxD1) > 10), (np.maximum(np.abs(flxD1),np.abs(flxD2)) > 15)))
    flxD1[ind] = np.nan
    flxD2[ind] = np.nan
    da[ind] = np.nan

    return flxD1, flxD2, da

def printInfo(str):
    if info:
        print(str)