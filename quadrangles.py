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

    def __init__(self, mgnt, indices, ID, c=random.random()):
        """Accepts heliographic bounds and a list of indices for block """
        self.mgnt = mgnt
        self.indices = indices
        self.id = ID
        self.pltColor = c
        self.lat = (self.min_latitude(), self.max_latitude())
        self.lon = (self.min_longitude(), self.max_longitude())
        self.diskAngle = self.averageDA()

    def min_latitude(self):
        return M.nanmin(self.mgnt.lath[self.indices])

    def max_latitude(self):
        return M.nanmax(self.mgnt.lath[self.indices])

    def min_longitude(self):
        return M.nanmin(self.mgnt.lonh[self.indices])

    def max_longitude(self):
        return M.nanmax(self.mgnt.lonh[self.indices])

    def averageDA(self):
        """Returns the angular distance from disk center in degrees."""
        mgnt = self.mgnt
        meanAngularRadius = M.nanmean(mgnt.rg[self.indices])
        return M.arcsin(meanAngularRadius/mgnt.rsun)*180/np.pi

    def mean_field(self):
        """Returns the average uncorrected magnetic field."""
        return M.nanmean(self.mgnt.im_raw_u[self.indices])

    def sum_flux(self):
        """Returns the total corrected magnetic flux."""
        return M.nansum(self.mgnt.mflux_corr[self.indices])

    def sum_area(self):
        """Returns the total area covered by the quadrangle."""
        return M.nansum(self.mgnt.area[self.indices])

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
    secFragInfo = get_fragmentation_info(secmgnt, n, refFragInfo['lonBands'])
    refBlocks, secBlocks = fragmentation_loop(refFragInfo, secFragInfo)
    
    return refBlocks, secBlocks

def get_fragmentation_info(m, n, lonBands=None):
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
    if lonBands is None:
        lonBands = _split(n, M.nanmin(m.lonh), M.nanmax(m.lonh))
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
        areaRatio = _calc_area_ratio(latBands, lonBands, latBounds)
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

                if secFragInfo is not None:
                    try:
                        secBlockBool = secLonMasks[j]*~secLonMasks[j+skip]*secLatitudeSetDiff
                    except IndexError:
                        secBlockBool = secLonMasks[j]*~secLonMasks[-1]*secLatitudeSetDiff
                    if ~(secBlockBool.any()):
                        continue
                    secBlockInd = _transform_indices(secmgnt.ind_1d[secBlockBool], secL)
                    y = Quadrangle(secmgnt, secBlockInd, uuid, c)
                    secBlocks.append(y)

                blockInd = _transform_indices(mgnt.ind_1d[blockBool], l)
                x = Quadrangle(mgnt, blockInd, uuid, c)
                blocks.append(x)
        currLat = nextLat
        if secFragInfo is not None:
            secCurrLat = secNextLat

    if secFragInfo is not None:
        return blocks, secBlocks
    else:
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

def block_area(mgnt, blocks):
    """Given a list of blocks, will calculate area totals."""

    areas = []
    for block in blocks:
        try:
            areas.append(block.sum_area())
        except AttributeError:
            mgnt.eoa()
            areas.append(block.sum_area())
        except:
            areas.append(M(np.nan, np.nan))

    return areas

def block_field(mgnt, blocks, raw=False):
    """Given a list of blocks, will calculate and print out mean flux density."""
    #TODO: add raw/corrected field support0
    field = []
    for block in blocks:
        try:
            field.append(block.mean_field())
        except:
            field.append(M(np.nan, np.nan))
    return field

def block_flux(mgnt, blocks):
    """Given a list of blocks, will calculate and print out total flux."""

    flux = []
    for block in blocks:
        try:
            flux.append(block.sum_flux())
        except AttributeError:
            mgnt.magnetic_flux()
            flux.append(block.sum_flux())
        except:
            flux.append(M(np.nan, np.nan))

    return flux

def calc_block_parameters(m, blockList, uncertainty=False):
    """Extracts values from block parameters and outputs ndarrays."""
    printInfo('Calculating block parameters...')
    ar = np.array([x.v for x in block_area(m, blockList)])
    f = np.array([x.v for x in block_field(m, blockList)])
    da = np.array([np.float32(x.diskAngle.v) for x in blockList])

    if uncertainty:
        ar_unc = np.array([x.u for x in block_area(m, blockList)])
        f_unc = np.array([x.u for x in block_field(m, blockList)])
        da_unc = np.array([np.float32(x.diskAngle.u) for x in blockList])
        return ar, f, da, ar_unc, f_unc, da_unc

    return ar, f, da

def printInfo(str):
    if info:
        print(str)