"""This module provides the framework for our fragmentation algorithm.

The class Quadrangle contains all the necessary information of each "pixel". This module can fragment a single
magnetogram or two at the same time - thus providing the comparison. The algorithm effectively increases the "pixel"
size for greater comparison accuracy between two magnetograms.
"""

import numpy as np

import uncertainty.measurement as mnp

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]


class Quadrangle:
    """This class contains information of the quadrangle regarding mean flux, total area, and distance from disk center.

    If two magnetograms were compared it will store both flux densities, otherwise just the one. The heliographic
    bounded information will be lost if it is uploaded into a database or condensed using the transform_blocks_to_dict
    function in cross_calibration, but it is here for informational purposes.
    """

    def __init__(self, mgnt, i, id_num, m2=None):
        """Accepts heliographic bounds and a list of indices for block """
        self.id = id_num
        self.i1 = mgnt.im_raw.instrument
        self.date1 = mgnt.im_raw.date
        self.lat = (self.min_latitude(mgnt, i), self.max_latitude(mgnt, i))
        self.lon = (self.min_longitude(mgnt, i), self.max_longitude(mgnt, i))
        self.diskAngle = self.averageDA(mgnt, i)
        self.area = self.sum_area(mgnt, i)
        self.fluxDensity = self.mean_flux_density(mgnt.im_corr.v, i)
        if m2 is not None:
            self.fluxDensity2 = self.mean_flux_density(m2.remap, i)
            self.date2 = m2.im_raw.date
            self.i2 = m2.im_raw.instrument

    def min_latitude(self, m, ind):
        """Calculate the minimum latitude."""
        return mnp.nanmin(m.lath[ind])

    def max_latitude(self, m, ind):
        """Calculate the maximum latitude."""
        return mnp.nanmax(m.lath[ind])

    def min_longitude(self, m, ind):
        """Calculate the minimum longitude."""
        return mnp.nanmin(m.lonh[ind])

    def max_longitude(self, m, ind):
        """Calculate the maximum longitude."""
        return mnp.nanmax(m.lonh[ind])

    def averageDA(self, m, ind):
        """Return the angular distance from disk center in degrees."""
        meanAngularRadius = mnp.nanmean(m.rg[ind])
        return mnp.arcsin(meanAngularRadius/m.par['rsun'])*180/np.pi

    def mean_flux_density(self, arr, ind):
        """Return the average corrected magnetic flux density."""
        return mnp.mean(arr[ind])

    def sum_area(self, m, ind):
        """Return the total area covered by the quadrangle."""
        if m.area is None:
            m.eoa()
        return mnp.nansum(m.area[ind])


def fragment_single(mgnt, n):
    """Take in a magnetogram and n value and returns a list of quadrangle objects.

    The magnetogram must have heliographic information stored (lonh and lath).

    Args:
        mgnt (obj): CRD object you want fragmented
        n (int): the fragmentation parameter

    Returns:
        list: a list of quadrangles
    """
    print("Processing {}".format(n))
    frag_info = get_fragmentation_info(mgnt, n)
    return fragmentation_loop(frag_info)


def fragment_multiple(m1, m2, n):
    """Take in two magnetograms and n value and returns a list of quadrangle objects.

    The magnetograms must have heliographic information stored (lonh and lath).

    Args:
        m1 (obj): reference CRD object
        m2 (obj): secondary CRD object
        n (int): the fragmentation parameter

    Returns:
        list: a list of quadrangles

    """
    refmgnt = m1
    secmgnt = m2

    print_info("Processing {}".format(n))
    ref_frag_info = get_fragmentation_info(refmgnt, n)
    blocks = fragmentation_loop(ref_frag_info, secmgnt)
    
    return blocks


def get_fragmentation_info(m, n):
    """Input a magnetogram and fragmentation parameter (n) and output dict of parameters.

    This dictionary provides the fragmentation_loop with the requisite information such as latitude/longitude interval
    space.

    Args:
        m (obj): CRD object
        n (int): fragmentation parameter

    Returns:
        dict: dictionary of values for the latitudes and longitudes used
    """

    m.lonh[m.rg > m.par['rsun']*np.sin(85.0*np.pi/180)] = np.nan
    m.lath[m.rg > m.par['rsun']*np.sin(85.0*np.pi/180)] = np.nan
    _flatten(m)

    # This supports reusing latitude and longitude bands for other mgnts.
    lat_bands = _split(n)
    lon_bands = _split(n, mnp.nanmin(m.lonh), mnp.nanmax(m.lonh))
    lon_masks = []

    for lon in lon_bands:
        lon_masks.append(m.lonh_1d > lon)

    frag_info = {'mgnt': m, 'lat_bands': lat_bands, 'lon_bands': lon_bands, 'lon_masks': lon_masks}
    return frag_info


def fragmentation_loop(ref_frag_info, secmgnt=None):
    """
    The main loop that finds valid indices for lat/lon conditions.

    Input a dictionary of information for a single magnetogram, or
    two if you want to compare two with the same heliographic bounds.
    This dictionary must have the following valid keys:

    mgnt: The magnetogram with flattened indices and heliographic information.
    lat_bands: The interval space of latitudes to be used.
    lon_bands: Them interval space of longitudes to be used.
    lon_masks: The list of boolean masks for valid indices over lonBand space.
    """

    blocks = []
    mgnt = ref_frag_info['mgnt']
    lat_bands = ref_frag_info['lat_bands']
    lon_bands = ref_frag_info['lon_bands']
    lon_masks = ref_frag_info['lon_masks']
    l, c = (int(x.value) for x in (mgnt.im_raw.dimensions[0], mgnt.im_raw.dimensions[1]))
    n = len(lon_bands) - 1

    curr_lat = (mgnt.lath_1d > lat_bands[0])
    for i in range(n):
        next_lat = (mgnt.lath_1d > lat_bands[i + 1])
        latitude_set_diff = curr_lat*~next_lat
        if ~(latitude_set_diff.any()):
            continue
        min_lon = np.nanmin(mgnt.lonh_1d[latitude_set_diff])
        max_lon = np.nanmax(mgnt.lonh_1d[latitude_set_diff])

        lat_bounds = (lat_bands[i], lat_bands[i + 1])
        area_ratio = _calc_area_ratio(lat_bands, lon_bands, lat_bounds)
        s = max(np.searchsorted(lon_bands, min_lon) - 1, 0)
        e = np.searchsorted(lon_bands, max_lon)
        skip = int(round(area_ratio))

        for j in range(s, e, skip):
            uuid = str(i) + str(j) + str(n)
            try:
                block_bool = lon_masks[j]*~lon_masks[j+skip]*latitude_set_diff
            except IndexError:
                block_bool = lon_masks[j]*~lon_masks[-1]*latitude_set_diff
            if ~(block_bool.any()):
                continue
            else:
                block_ind = _transform_indices(mgnt.ind_1d[block_bool], l, c)
                x = Quadrangle(mgnt, block_ind, int(uuid), secmgnt)
                x.fragmentationValue = n
                blocks.append(x)
        curr_lat = next_lat

    return blocks


def _split(n, minimum=-90, maximum=90):
    """Returns an interval space based on n number of blocks."""
    return np.linspace(minimum, maximum, n + 1)


def _flatten(m):
    """Take in a magnetogram and add 1 dimensional flattened attributes for heliographic information."""
    m.lonh_1d = m.lonh.v.flatten()
    m.lath_1d = m.lath.v.flatten()
    m.ind_1d = np.arange(np.size(m.lath_1d))

    ind = np.where(mnp.isfinite(m.lonh_1d))

    m.lonh_1d = m.lonh_1d[ind]
    m.ind_1d = m.ind_1d[ind]
    m.lath_1d = m.lath_1d[ind]


def _calc_area_ratio(lat_bands, lon_bands, lat_bounds):
    """Calculate the area ratio of the given latitude band.

    The middle square solid angle is the centermost block of the sun. This is used for reference because arguably it is
    the biggest block given certain lat/lon bounds. This is used in skipping n values for a more comprehensive approach
    to enlarging the blocks near the poles for area similarities.

    Args:
        lat_bands (array): array of numbers denoting where the latitudes are
        lon_bands(array): array of numbers denoting where the longitudes are
        lat_bounds (tuple): bounds of the latitude band you want to calculate the area ratio for

    Returns:
        float: a number specifying the ratio of areas between latitudes

    """
    n = len(lat_bands) - 1
    lon_block_diff = lon_bands[n // 2 + 1] - lon_bands[n // 2]

    middle_square_sa = (np.sin(lat_bands[n // 2 + 1] * np.pi / 180)
                        - np.sin(lat_bands[n // 2] * np.pi / 180)) * lon_block_diff
    sa = (np.sin(lat_bounds[1] * np.pi / 180) - np.sin(lat_bounds[0] * np.pi / 180)) * lon_block_diff
    area_ratio = middle_square_sa/sa

    return area_ratio


def _transform_indices(ind, column_count, row_count):
    """Takes a 1D list of indices and calculates a 2D list of indices."""
    cols = ind % column_count
    rows = ind // row_count
    return rows, cols


def extract_valid_points(bl):
    """Extracts the valid points from the dictionary set."""
    flx_d1 = bl['reference_fd']
    flx_d2 = bl['secondary_fd']
    da = bl['disk_angle']

    return flx_d1, flx_d2, da
