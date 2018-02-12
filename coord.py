"""Provides the machinery for reading in fits files and calculating attributes used in the analysis.

This module contains the CRD class used for working with fits files and the coordinate information we will be working
with for cross calibration in an easy way.

Todo:
    Get rid of the pixel thing and just make everything array-wide operations by now.

"""
from __future__ import division

import os.path
import kpvt

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.io import fits
from sunpy.sun import sun

import uncertainty.measurement as mnp

__authors__ = ["Zach Werginz", "Andrés Muñoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]


class CRD:
    """Read in a fits file and calculates various magnetogram coordinate information.

    First the class reads the fits file as a sunpy map object to provide us with some useful high-level functions for
    extracting information easiliy. A typical instatiation should go something like this: wrap the filename with CRD()
    and call the magnetic_flux() function to populate all the extra calculations. This will calculate heliographic
    longitude, heliographic latitude, a line-of-sight correction, pixel area, and pixel magnetic flux for the whole
    image. If the area and flux are not needed, it is sufficient to just call the heliographic() and los_corr() methods
    - they are much faster.

    Attributes:
        rsun_meters: radius of the sun in meters
        dsun_meters: distance of the sun from the earth in meters

    Examples:
        >>> mgnt = CRD('fd_M_96m_01d.4219.0000.fits')
        >>> mgnt.heliographic()
        >>> mgnt.los_corr()
        >>> mgnt.eoa()
        >>> mgnt.magnetic_flux()
        >>> mgnt.par
        {'B0': Measurement(4.918841303479, 0.04918841303479),
        'L0': Measurement(17.74726719194, 0.1774726719194),
        'SL0': Measurement(-0.24830962816345092, 0.1774726719194),
        'X0': 513.3475036737464,
        'Y0': 513.6332168712381,
        'dsun': Measurement(150578367044.1338, 0),
        'rsun': Measurement(952.7186785154763, 1),
        'xscale': Measurement(1.982, 0.003),
        'yscale': Measurement(1.982, 0.003)}
    """

    rsun_meters = mnp.Measurement(sun.constants.radius.si.value, 26000)
    dsun_meters = mnp.Measurement(sun.constants.au.si.value, 0)

    def __init__(self, filename, rotate=0):
        """Read a magnetogram as a sunpy.map object.

        Args:
            filename (str): filepath of magnetogram
            rotate (float, optional): amount to rotate the image, defaults to zero

        Raises:
            IOError: if the fits file does not fit one the file specifications or the file does not exist

        """
        self.im_raw = sunpy.map.Map(filename)
        self.fn = filename
        self.im_corr = None
        self.lath = None
        self.lonh = None
        self.mflux_corr = None
        self.mflux_raw = None
        self.area = None
        self.par = {}

        if self.im_raw.detector == '512':
            self.par['X0'] = self.im_raw.meta['CRPIX1A']
            self.par['Y0'] = self.im_raw.meta['CRPIX2A']
            self.par['B0'] = mnp.Measurement(self.im_raw.meta['B0'], np.abs(self.im_raw.meta['B0']) * .01)
            self.par['L0'] = mnp.Measurement(self.im_raw.meta['L0'], np.abs(self.im_raw.meta['L0']) * .01)
            self.par['SL0'] = mnp.Measurement(0, 0)  # Stonyhurst L0
            self.par['xscale'] = mnp.Measurement(self.im_raw.scale[0].value, 0.002)
            self.par['yscale'] = mnp.Measurement(self.im_raw.scale[1].value, 0.002)
            self.par['rsun'] = mnp.Measurement(self.im_raw.rsun_obs.value, 1)
            self.par['dsun'] = self.dsun_meters
            if rotate != 0:
                self.im_raw = self.im_raw.rotate(angle=rotate * u.deg)
        elif self.im_raw.detector == 'SPMG':
            self.par['X0'] = self.im_raw.meta['CRPIX1A']
            self.par['Y0'] = self.im_raw.meta['CRPIX2A']
            self.par['B0'] = mnp.Measurement(self.im_raw.meta['B0'], np.abs(self.im_raw.meta['B0']) * .01)
            self.par['L0'] = mnp.Measurement(self.im_raw.meta['L0'], np.abs(self.im_raw.meta['L0']) * .01)
            self.par['SL0'] = mnp.Measurement(0, 0)  # Stonyhurst L0
            self.par['xscale'] = mnp.Measurement(self.im_raw.scale[0].value, 0)
            self.par['yscale'] = mnp.Measurement(self.im_raw.scale[1].value, 0)
            self.par['rsun'] = mnp.Measurement(self.im_raw.rsun_obs.value, 1)
            self.par['dsun'] = self.dsun_meters
            if rotate != 0:
                self.im_raw.rotate(angle=rotate * u.deg)
        elif self.im_raw.detector == 'MDI':
            self.par['rsun'] = mnp.Measurement(self.im_raw.rsun_obs.value, 1)
            self.par['dsun'] = mnp.Measurement(self.im_raw.dsun.value, 0)
            try:
                self.P0 = self.im_raw.meta['p_angle'] - rotate
            except KeyError:
                self.P0 = self.im_raw.meta['solar_p'] - rotate
            if self.P0 != 0:
                self.im_raw = self.im_raw.rotate(angle=-self.P0 * u.deg)
            self.par['X0'], self.par['Y0'] = (x.value for x in self.im_raw.reference_pixel)
            try:
                self.par['B0'] = mnp.Measurement(self.im_raw.meta['B0'], np.abs(self.im_raw.meta['B0']) * .01)
                self.par['L0'] = mnp.Measurement(self.im_raw.meta['L0'], np.abs(self.im_raw.meta['L0']) * .01)
            except KeyError:
                self.par['B0'] = mnp.Measurement(self.im_raw.meta['OBS_B0'], np.abs(self.im_raw.meta['OBS_B0']) * .01)
                self.par['L0'] = mnp.Measurement(self.im_raw.meta['OBS_L0'], np.abs(self.im_raw.meta['OBS_L0']) * .01)
            self.par['SL0'] = self.par['L0'] - sun.heliographic_solar_center(self.im_raw.date)[0].value
            if self.par['SL0'] < -90:
                self.par['SL0'] += 360
            if self.par['SL0'] > 90:
                self.par['SL0'] -= 360
            self.par['xscale'] = mnp.Measurement(1.982, 0.003)
            self.par['yscale'] = mnp.Measurement(1.982, 0.003)
        elif self.im_raw.detector == 'HMI':
            self.par['rsun'] = mnp.Measurement(self.im_raw.rsun_obs.value, 1)
            self.par['dsun'] = mnp.Measurement(self.im_raw.dsun.value, 0)
            self.P0 = self.im_raw.meta['CROTA2'] - rotate
            if self.P0 != 0:
                self.im_raw = self.im_raw.rotate(angle=-self.P0 * u.deg)
            self.par['X0'], self.par['Y0'] = (x.value for x in self.im_raw.reference_pixel)
            self.par['B0'] = mnp.Measurement(self.im_raw.meta['CRLT_OBS'], np.abs(self.im_raw.meta['CRLT_OBS']) * .01)
            self.par['L0'] = mnp.Measurement(self.im_raw.meta['CRLN_OBS'], np.abs(self.im_raw.meta['CRLN_OBS']) * .01)
            self.par['SL0'] = self.par['L0'] - sun.heliographic_solar_center(self.im_raw.date)[0].value
            if self.par['SL0'] < 0:
                self.par['SL0'] += 360
            self.par['xscale'] = mnp.Measurement(self.im_raw.scale[0].value, 0.001)
            self.par['yscale'] = mnp.Measurement(self.im_raw.scale[1].value, 0.001)
        else:
            print("Not a valid instrument or missing header information regarding instrument.")
            raise IOError

        self.im_raw_u = mnp.Measurement(self.im_raw.data, np.abs(self.im_raw.data) * .10)
        # Fill in last bit of data if reading in cached file.
        if hasattr(self, 'lonh'):
            x, y = self._grid()

    def __repr__(self):
        print(self.im_raw.__repr__())

    def meta(self):
        """Prints the sunpy map header."""
        print(self.im_raw.meta)

    def heliographic(self, *args, array=True, corners=False):
        """Calculate heliographic coordinates from helioprojective cartesian coordinates and return it.

        Can accept either a coordinate pair (x, y) or the entire map. This pair cor5responds the the pixel you
        want information on. Use standard python indexing conventions for both the single coordinate and array
        calculations [row, column].


        Args:
            *args: pixel coordinates
            array (bool, optional): whether or not to calculate the whole image map, defaults to True
            corners (bool, optional): choose whether to shift by a half a pixel to get coordinates for original map
                pixel corners, defaults to False

        Returns:
            lonh (np.array): heliographic longitude
            lath (np.array): heliographic latitude

        """

        if array and self.lonh is not None and not corners:
            return

        # Check for single coordinate or ndarray object.
        if array:
            x, y = self._grid(corners)
        else:
            # Coordinate conventions go [row, col] with
            # row zero being at the bottom (fits file)
            x_scale = self.im_raw.scale[0].value
            y_scale = self.im_raw.scale[1].value
            x = (args[1] - self.par['X0']) * x_scale
            y = (args[0] - self.par['Y0']) * y_scale

        # Calculations taken from sunpy.wcs.
        # First convert to heliocentric cartesian coordinates.
        rx, ry, rz = self._hpc_hcc(x, y)

        # Now convert to heliographic coordinates.
        lonh, lath = self._hcc_hg(rx, ry, rz, self.par['B0'], self.par['SL0'])

        self.im_raw.data[self.rg > self.par['rsun']] = np.nan
        if not corners:
            self.lonh = lonh
            self.lath = lath
            return

        return lonh, lath

    def los_corr(self, *args, array=True):
        """Takes in coordinates and returns corrected magnetic field.

        Applies the dot product between the observers unit vector and the heliographic radial vector to get the true
        magnitude of the magnetic field vector. See geometric projection for calulations.

        Args:
            *args: pixel coordinates
            array (bool, optional): whether or not to calculate the whole image map, defaults to True

        Returns:
            calculation: the line-of-sight correction

        """

        if array and self.im_corr is not None:
            return
        elif array:
            print("Correcting line of sight magnetic field...")
            if self.lonh is None:
                self.heliographic()
                lonh, lath = mnp.deg2rad(self.lonh), mnp.deg2rad(self.lath)
            else:
                lonh, lath = mnp.deg2rad(self.lonh), mnp.deg2rad(self.lath)
        else:
            lonh, lath = mnp.deg2rad(self.heliographic(args[0], args[1]))

        b0 = mnp.deg2rad(self.par['B0'])
        sl0 = mnp.deg2rad(self.par['SL0'])

        x_obs = mnp.cos(b0) * mnp.cos(sl0)
        y_obs = mnp.cos(b0) * mnp.sin(sl0)
        z_obs = mnp.sin(b0)

        corr_factor = (mnp.cos(lath) * mnp.cos(lonh) * x_obs
                       + mnp.cos(lath) * mnp.sin(lonh) * y_obs
                       + mnp.sin(lath) * z_obs)

        if array:
            self.im_corr = self.im_raw_u / corr_factor
            bad_ind = np.where(self.rg > self.par['rsun'] * np.sin(75.0 * np.pi / 180))
            self.im_corr[bad_ind] = np.nan
            return
        else:
            return self.im_raw.data[args[0], args[1]] / corr_factor

    def eoa(self, *args, array=True):
        """Takes in coordinates and returns the area of pixels on sun.

        Each pixel is projected onto the sun, and therefore pixels close to the limbs have vastly greater areas. This
        function uses a closed form solution to a spherical area integral to calulate the area based on the heliographic
        coordinate unit vectors of each corner of the pixel. We use these to calculate a solid angle of a pyramid with
        its apex at the center of the sun. We will assume that the coordinate is in the center of the pixel as described
        in this article

        http://www.aanda.org/component/article?access=bibcode&bibcode=&bibcode=2002A%2526A...395.1061GFUL

        Args:
            *args: pixel coordinates
            array (bool, optional): whether or not to calculate the whole image map, defaults to True

        Returns:
            calculation: the area element for each pixel

        """

        if array and self.area is not None:
            return
        if array:
            print("Calculating element of area...")
            lon, lat = self.heliographic(corners=True)
            lon = lon * np.pi / 180
            lat = lat * np.pi / 180
            # Calculating unit vectors of pixel corners for solid angle.
            r1 = self._spherical_to_cartesian(lon, lat, 0, 0)
            r2 = self._spherical_to_cartesian(lon, lat, 1, 0)
            r3 = self._spherical_to_cartesian(lon, lat, 1, 1)
            r4 = self._spherical_to_cartesian(lon, lat, 0, 1)

        else:
            x = args[0]
            y = args[1]
            lon_ul, lat_ul = self.heliographic(x - .5, y - .5)
            lon_ll, lat_ll = self.heliographic(x + .5, y - .5)
            lon_lr, lat_lr = self.heliographic(x + .5, y + .5)
            lon_ur, lat_ur = self.heliographic(x - .5, y + .5)

            # Calculating unit vectors of pixel corners for solid angle.
            r1 = np.array([np.cos(np.deg2rad(lat_ul)) * np.cos(np.deg2rad(lon_ul)),
                           np.cos(np.deg2rad(lat_ul)) * np.sin(np.deg2rad(lon_ul)),
                           np.sin(np.deg2rad(lat_ul))])

            r2 = np.array([np.cos(np.deg2rad(lat_ll)) * np.cos(np.deg2rad(lon_ll)),
                           np.cos(np.deg2rad(lat_ll)) * np.sin(np.deg2rad(lon_ll)),
                           np.sin(np.deg2rad(lat_ll))])

            r3 = np.array([np.cos(np.deg2rad(lat_lr)) * np.cos(np.deg2rad(lon_lr)),
                           np.cos(np.deg2rad(lat_lr)) * np.sin(np.deg2rad(lon_lr)),
                           np.sin(np.deg2rad(lat_lr))])

            r4 = np.array([np.cos(np.deg2rad(lat_ur)) * np.cos(np.deg2rad(lon_ur)),
                           np.cos(np.deg2rad(lat_ur)) * np.sin(np.deg2rad(lon_ur)),
                           np.sin(np.deg2rad(lat_ur))])

        # Calculate solid angle of pixel based on a pyrimid shaped polygon.
        # See http://planetmath.org/solidangleofrectangularpyramid
        cross1 = mnp.cross(r1, r2, axis=0)
        cross2 = mnp.cross(r3, r4, axis=0)
        numerator1 = mnp.dot(cross1, r3)
        numerator2 = mnp.dot(cross2, r1)
        solid_angle1 = 2 * mnp.arctan2(numerator1, (mnp.dot(r1, r2) + mnp.dot(r2, r3) + mnp.dot(r3, r1) + 1))
        solid_angle2 = 2 * mnp.arctan2(numerator2, (mnp.dot(r3, r4) + mnp.dot(r4, r1) + mnp.dot(r3, r1) + 1))
        solid_angle = solid_angle1 + solid_angle2

        r = self.rsun_meters * 100  # Convert to centimeters
        if array:
            self.area = abs((r ** 2) * solid_angle)
            ind = np.where(self.Rg[1:len(self.Rg) - 1, 1:len(self.Rg) - 1] > self.par['rsun'])
            del self.Rg
            self.area[ind] = np.nan
            return
        else:
            if self.rg > self.par['rsun']:
                return np.nan
            else:
                return np.abs((r ** 2) * solid_angle)

    def magnetic_flux(self, *args, array=True, raw_field=False):
        """Takes in coordinates and returns magnetic flux of pixel.

        This calculation is just the area times the magnetic flux density (field strength).

        Args:
            *args: pixel coordinates
            array (bool, optional): whether or not to calculate the whole image map, defaults to True
            raw_field (bool, optional): choose between raw field (True) or line-of-sight correction (False), defaults
                to False

        Returns:
            object: the magnetic flux array

        """
        if not array:
            return self.eoa(*args) * self.los_corr(*args)
        else:
            if self.mflux_corr is not None:
                return
            if self.area is None:
                self.eoa()
                area = self.area
            else:
                area = self.area

            if raw_field:
                field = self.im_raw_u
                print("Calculating raw magnetic flux...")
                self.mflux_raw = area * field
                return
            else:
                if self.im_corr is None:
                    self.los_corr()
                    field = self.im_corr
                else:
                    field = self.im_corr
                print("Calculating corrected magnetic flux...")
                self.mflux_corr = area * field
                return

    def _grid(self, corners=False):
        """Create an xy grid of coordinates for heliographic array.

        Uses meshgrid. If corners is selected, this function will shift the array by half a pixel in both directions
        so that the corners of the normal array can be accessed easily.

        Args:
            corners (bool, optional): defaults to False, chooses whether to apply the corner calculation or not

        Returns:
            xg: 2D array containing the x-coordinates of each pixel
            yg: 2D array containing the y-coordinates of each pixel

        """
        # Retrieve integer dimensions and create arrays holding
        # x and y coordinates of each pixel
        x_dim = np.int(np.floor(self.im_raw.dimensions[0].value))
        y_dim = np.int(np.floor(self.im_raw.dimensions[1].value))

        if corners:
            x_row = (np.arange(0, x_dim + 1) - self.par['X0'] - 0.5) * self.par['xscale']
            y_row = (np.arange(0, y_dim + 1) - self.par['Y0'] - 0.5) * self.par['yscale']
            xg, yg = mnp.meshgrid(x_row, y_row)
            rg = mnp.sqrt(xg ** 2 + yg ** 2)
            self.Rg = rg
        else:
            x_row = (np.arange(0, x_dim) - self.par['X0']) * self.par['xscale']
            y_row = (np.arange(0, y_dim) - self.par['Y0']) * self.par['yscale']
            xg, yg = mnp.meshgrid(x_row, y_row)
            rg = mnp.sqrt(xg ** 2 + yg ** 2)
            self.xg = xg
            self.yg = yg
            self.rg = rg

        return xg, yg

    def _hpc_hcc(self, x, y):
        """Converts hpc coordinates to hcc coordinates.

        Calculations taken and shortened from sunpy.wcs. Transforms an array from helioprojective cartesian coordinates
        to heliocentric cartesian.

        Args:
            x: x coordinate in arcseconds
            y: y coordinate in arcseconds

        Returns:
            rx: x coordinate in meters
            ry: y coordinate in meters
            rz: z coordinate in meters

        """
        x *= np.deg2rad(1) / 3600.0
        y *= np.deg2rad(1) / 3600.0

        q = self.par['dsun'] * mnp.cos(y) * mnp.cos(x)
        distance = q ** 2 - self.par['dsun'] ** 2 + self.rsun_meters ** 2
        distance = q - mnp.sqrt(distance)

        rx = distance * mnp.cos(y) * mnp.sin(x)
        ry = distance * mnp.sin(y)
        rz = mnp.sqrt(self.rsun_meters ** 2 - rx ** 2 - ry ** 2)

        return rx, ry, rz

    def _hcc_hg(self, x, y, z, b0=0, l0=0):
        """Converts hcc coordinates to Stonyhurst heliographic.

        Calculations taken and shortened from sunpy.wcs. Transforms an array from heliocentric cartesian coordinates
        to heliographic.

        Args:
            x (array): x coordinate in meters
            y (array): y coordinate in meters
            z (array): z coordinate in meters

        Returns:
            hgln (array): heliographic longitude
            hglt (array): heliographic latitute

        """
        cosb = mnp.cos(mnp.deg2rad(b0))
        sinb = mnp.sin(mnp.deg2rad(b0))

        hecr = mnp.sqrt(x ** 2 + y ** 2 + z ** 2)
        hgln = mnp.arctan2(x, z * cosb - y * sinb) + mnp.deg2rad(l0)
        hglt = mnp.arcsin((y * cosb + z * sinb) / hecr)

        return hgln * 180 / np.pi, hglt * 180 / np.pi

    def _spherical_to_cartesian(self, lon, lat, i, j):
        """Takes latitude, longitude arrays and returns cartesian unit vector.

        Latitude and longitude must be in degrees and must have already been
        shifted .5 pixels as calculated from corners keyword of heliographic
        function. i and j represent the shift and thus corner.
        i, j
        0, 0: top-left
        1, 0: bottom-left
        1, 1: bottom-right
        0, 1: top-right

        Args:
            lon (array): the longitude array
            lat (array): the latitude array
            i (float): the shift in x you want, usually either .5, 0, or -.5
            j (float): the shift in y you want, usually either .5, 0, or -.5

        Returns:
            r (array): the resultant cartesian unit vector.

        """
        coslat = mnp.cos(lat)
        coslon = mnp.cos(lon)
        sinlat = mnp.sin(lat)
        sinlon = mnp.sin(lon)
        l = len(lat)
        x_ar = coslat[i:l - 1 + i, j:l - 1 + j] * coslon[i:l - 1 + i, j:l - 1 + j]
        y_ar = coslat[i:l - 1 + i, j:l - 1 + j] * sinlon[i:l - 1 + i, j:l - 1 + j]
        z_ar = sinlat[i:l - 1 + i, j:l - 1 + j]
        r = np.ndarray(shape=(3,), dtype=np.object)
        r[0] = x_ar
        r[1] = y_ar
        r[2] = z_ar
        return r
