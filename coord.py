from __future__ import division
import numpy as np
import sunpy.map
from sunpy.sun import sun
import astropy.units as u
from astropy.io import fits
import os.path
import kpvt

import uncertainty.measurement as mnp

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]


class CRD:
    """Calculates various magnetogram coordinate information.
    Can calculate heliographic coordinate information,
    line of sight (LOS) corrections for the magnetic field,
    area elements for each pixel, and magnetic flux. This can
    be done for one pixel, or the whole data map. If the whole
    data map is given as a parameter, it will save the information
    as an instance attribute for the object.
    """

    RSUN_METERS = mnp.Measurement(sun.constants.radius.si.value, 26000)
    DSUN_METERS = mnp.Measurement(sun.constants.au.si.value, 0)

    def __init__(self, filename, rotate=0):
        """Reads magnetogram as a sunpy.map object."""
        self.fn = filename
        self.cached = False
        self.im_corr = None
        self.lath = None
        self.lonh = None
        self.mflux_corr = None
        self.mflux_raw = None
        self.area = None
        self.par = {}
        try:
            self._load_cache()
        except IOError:
            self.im_raw = sunpy.map.Map(filename)

        if self.im_raw.detector == '512':
            self.par['X0'] = self.im_raw.meta['CRPIX1A']
            self.par['Y0'] = self.im_raw.meta['CRPIX2A']
            self.par['B0'] = mnp.Measurement(self.im_raw.meta['B0'], np.abs(self.im_raw.meta['B0']) * .01)
            self.par['L0'] = mnp.Measurement(self.im_raw.meta['L0'], np.abs(self.im_raw.meta['L0']) * .01)
            self.par['SL0'] = mnp.Measurement(0, 0)  # Stonyhurst L0
            self.par['xscale'] = mnp.Measurement(self.im_raw.scale[0].value, 0.002)
            self.par['yscale'] = mnp.Measurement(self.im_raw.scale[1].value, 0.002)
            self.par['rsun'] = mnp.Measurement(self.im_raw.rsun_obs.value, 1)
            self.par['dsun'] = self.DSUN_METERS
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
            self.par['dsun'] = self.DSUN_METERS
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

    def __del__(self):
        # if not self.cached and hasattr(self, 'area'):
        #    self._save_cache()
        pass

    def meta(self):
        """Prints the sunpy map header."""
        print(self.im_raw.meta)

    def heliographic(self, *args, array=True, corners=False):
        """Calculate heliographic coordinates from helioprojective cartesian coordinatesand returns it.

        Can accept either a coordinate pair (x, y) or the entire map.
        This pair corresponds the the pixel you want information on.

        Use standard python indexing conventions for both the single
        coordinate and array calculations [row, column].

        Examples:
        lath, lonh = kpvt.heliographic()
        aia.heliographic(320, 288, array=False)
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

        Applies the dot product between the observers unit vector and
        the heliographic radial vector to get the true magnitude of
        the magnetic field vector. See geometric projection for
        calulations.
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

        Each pixel is projected onto the sun, and therefore pixels close to
        the limbs have vastly greater areas. This function uses a closed form
        solution to a spherical area integral to calulate the area based on
        the heliographic coordinate unit vectors of each corner of the pixel.
        We use these to calculate a solid angle of a pyramid with its apex
        at the center of the sun.
        """

        if array and self.area is not None:
            return
        # Assume coordinate is in center of pixel.
        # Information on pixel standard is in this article.
        # http://www.aanda.org/component/article?access=bibcode&bibcode=&bibcode=2002A%2526A...395.1061GFUL
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

        r = self.RSUN_METERS * 100  # Convert to centimeters
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
        """Takes in coordinates and returns magnetic flux of pixel."""
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

        Uses meshgrid. If corners is selected, this function will shift
        the array by half a pixel in both directions so that the corners
        of the normal array can be accessed easily.
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

        x -- x coordinate in arcseconds
        y -- y coordinate in arcseconds
        Calculations taken and shortened from sunpy.wcs.
        """
        x *= np.deg2rad(1) / 3600.0
        y *= np.deg2rad(1) / 3600.0

        q = self.par['dsun'] * mnp.cos(y) * mnp.cos(x)
        distance = q ** 2 - self.par['dsun'] ** 2 + self.RSUN_METERS ** 2
        distance = q - mnp.sqrt(distance)

        rx = distance * mnp.cos(y) * mnp.sin(x)
        ry = distance * mnp.sin(y)
        rz = mnp.sqrt(self.RSUN_METERS ** 2 - rx ** 2 - ry ** 2)

        return rx, ry, rz

    def _hcc_hg(self, x, y, z, b0=0, l0=0):
        """Converts hcc coordinates to Stonyhurst heliographic.

        x - x coordinate in meters
        y - y coordinate in meters
        z - z coordinate in meters
        Calculations taken and shortened
        from sunpy.wcs.
        """
        cosb = mnp.cos(mnp.deg2rad(b0))
        sinb = mnp.sin(mnp.deg2rad(b0))

        hecr = mnp.sqrt(x ** 2 + y ** 2 + z ** 2)
        hgln = mnp.arctan2(x, z * cosb - y * sinb) + mnp.deg2rad(l0)
        hglt = mnp.arcsin((y * cosb + z * sinb) / hecr)

        return hgln * 180 / np.pi, hglt * 180 / np.pi

    def _dot(self, a, b):
        """Vectorized version of the dot product of two arrays.

        Wanted a dot product function that performed operations
        element-wise.
        """
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

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

    def _get_instrument(self, mfits):
        """Returns the fits primary data HDU index."""

        if len(mfits[0].header) < 10:
            return 1
        else:
            return 0

    def _save_cache(self):
        """Saves magnetogram with coordinate data in alternate fits file."""

        mfits = fits.open(self.fn)
        attr = [self.lonh, self.lath, self.area, self.im_corr]
        units = ['deg', 'deg', 'cm^2', 'G', 'Mx']
        x_name = ['LATITUDE', 'LONGITUDE', 'AREA', 'LOS CORRECTED FIELD']
        for dataArray in attr:
            mfits.append(fits.ImageHDU(np.array([dataArray.v.astype('float32'), dataArray.u.astype('float32')])))

        j = self._get_instrument(mfits)
        for i in range(1 + j, 5 + j):
            mfits[i].header['DATE-OBS'] = self.im_raw.date.isoformat()
            mfits[i].header['INSTRUME'] = self.im_raw.instrument
            mfits[i].header['BUNIT'] = units[i - 1 - j]
            mfits[i].header['EXTNAME'] = x_name[i - 1 - j]

        fn = self.fn.replace('.fits', '.CRD.fits')
        s = fn.split(':')
        s[0] = s[0] + ':'
        new_fn = os.path.join(s[0], 'CRD', s[-1])
        mfits.writeto(new_fn, output_verify='ignore')
        mfits.close()

    def _load_cache(self):
        paths = os.path.splitdrive(self.fn.replace('.fits', '.CRD.fits'))
        self.cached_fn = os.path.join(paths[0], 'CRD', paths[1])
        mfits = fits.open(self.cached_fn)
        i = self._get_instrument(mfits)
        self.im_raw = sunpy.map.Map(mfits[i].data, mfits[i].header)
        self.lonh = mnp.Measurement(mfits[i + 1].data[0], mfits[i + 1].data[1])
        self.lath = mnp.Measurement(mfits[i + 2].data[0], mfits[i + 2].data[1])
        self.area = mnp.Measurement(mfits[i + 3].data[0], mfits[i + 3].data[1])
        self.im_corr = mnp.Measurement(mfits[i + 4].data[0], mfits[i + 4].data[1])
        self.cached = True
        mfits.close()
