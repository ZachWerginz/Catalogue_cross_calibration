__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

import numpy as np
import sunpy.map
from sunpy.sun import constants
from sunpy.sun import sun
from uncertainty import Measurement as M
import astropy.units as u
import kpvt_class

class CRD:
    """Calculates various magnetogram coordinate information.
    Can calculate heliographic coordinate information,
    line of sight (LOS) corrections for the magnetic field,
    area elements for each pixel, and magnetic flux. This can
    be done for one pixel, or the whole data map. If the whole
    data map is given as a parameter, it will save the information
    as an instance attribute for the object.

    """

    RSUN_METERS = M(sun.constants.radius.si.value, 26000)
    DSUN_METERS = M(sun.constants.au.si.value, 0)

    
    def __init__(self, filename):
        """Reads magnetogram as a sunpy.map object."""
        self.im_raw = sunpy.map.Map(filename)

        if self.im_raw.detector == '512':

            # Define center of sun and location of detector.
            self.X0 = self.im_raw.meta['CRPIX1A']
            self.Y0 = self.im_raw.meta['CRPIX2A']
            self.B0 = M(self.im_raw.meta['B0'], np.abs(self.im_raw.meta['B0'])*.01)
            self.L0 = M(self.im_raw.meta['L0'], np.abs(self.im_raw.meta['L0'])*.01)
            self.xScale = M(self.im_raw.scale[0].value, 0.002)
            self.yScale = M(self.im_raw.scale[1].value, 0.002)
            self.rsun = M(self.im_raw.rsun_obs.value, 1)
            self.dsun = self.DSUN_METERS

        elif self.im_raw.detector == 'SPMG':

            # Define center of sun and location of detector.
            self.X0 = self.im_raw.meta['CRPIX1A']
            self.Y0 = self.im_raw.meta['CRPIX2A']
            self.B0 = M(self.im_raw.meta['B0'], np.abs(self.im_raw.meta['B0'])*.01)
            self.L0 = M(self.im_raw.meta['L0'], np.abs(self.im_raw.meta['L0'])*.01)
            self.xScale = M(self.im_raw.scale[0].value, 0)
            self.yScale = M(self.im_raw.scale[1].value, 0)
            self.rsun = M(self.im_raw.rsun_obs.value, 1)
            self.dsun = self.DSUN_METERS

        elif self.im_raw.detector == 'MDI':

            # Define center of sun and location of detector.
            self.X0 = self.im_raw.meta['X0']
            self.Y0 = self.im_raw.meta['Y0']
            try:
                self.B0 = M(self.im_raw.meta['B0'], np.abs(self.im_raw.meta['B0'])*.01)
                self.L0 = M(self.im_raw.meta['L0'], np.abs(self.im_raw.meta['L0'])*.01)
            except KeyError:
                self.B0 = M(self.im_raw.meta['OBS_B0'], np.abs(self.im_raw.meta['OBS_B0'])*.01)
                self.L0 = M(self.im_raw.meta['OBS_L0'], np.abs(self.im_raw.meta['OBS_L0'])*.01)
            self.xScale = M(1.982, 0.003)
            self.yScale = M(1.982, 0.003)
            self.rsun = M(self.im_raw.rsun_obs.value, 1)
            self.dsun = M(self.im_raw.dsun.value, 0)
            try:
                self.P0 = self.im_raw.meta['p_angle']
            except KeyError:
                self.P0 = self.im_raw.meta['solar_p']

            if self.P0 != 0:
                self.im_raw = self.im_raw.rotate(angle=self.P0*u.deg)

        elif self.im_raw.detector == 'HMI':
            self.X0 = self.im_raw.meta['CRPIX1']
            self.Y0 = self.im_raw.meta['CRPIX2']
            self.B0 = M(self.im_raw.meta['CRLT_OBS'], np.abs(self.im_raw.meta['CRLT_OBS'])*.01)
            self.L0 = M(self.im_raw.meta['CRLN_OBS'], np.abs(self.im_raw.meta['CRLN_OBS'])*.01)
            self.xScale = M(self.im_raw.scale[0].value, 0.001)
            self.yScale = M(self.im_raw.scale[1].value, 0.001)
            self.rsun = M(self.im_raw.rsun_obs.value, 1)
            self.dsun = M(self.im_raw.dsun.value, 0)
            #TODO: p_angle stuff for HMI as well
            if self.im_raw.meta['CROTA2'] != 0:
                self.im_raw.rotate(self.im_raw.meta['CROTA2'])

        else:
            print ("Not a valid instrument or missing header information regarding instrument.")
            raise IOError

        self.im_raw_u = M(self.im_raw.data, np.abs(self.im_raw.data)*.10)
            
    def __repr__(self):
        for key in ['X0', 'Y0', 'B0', 'L0', 'xScale', 'yScale', 'rsun', 'dsun']:
            print ("{0}: {1}".format(key, getattr(self, key)))
        #TODO: add attribute checks and fix Nonetype return error

    def meta(self):
        """Prints the fits header in a more readable fashion."""
        for key, value in self.im_raw.meta.items():
            print ("{}: {}".format(key, value))

    def heliographic(self, *args, array=True, corners=False):
        """Calculate hg coordinates from hpc and returns it.

        Can accept either a coordinate pair (x, y) or an entire 2D array.
        This pair corresponds the the pixel you want information on.

        The function will return a coordinate pair if given a coordinate
        pair or two arrays (latitude and longitude) if given a data array.
        Those arrays are indexed such that [0,0] is the top left pixel,
        although it refers to the bottom left portion of the picture.

        Use standard python indexing conventions for both the single
        coordinate and array calculations [row, column].

        Examples: 
        lath, lonh = kpvt.heliographic()
        aia.heliographic(320, 288, array=False)
        """

        # Check for single coordinate or ndarray object.
        if array:
            x, y = self._grid(corners)            
        else:
            # Coordinate conventions go [row, col] with
            # row zero being at the bottom (fits file)
            xScale = self.im_raw.scale[0].value
            yScale = self.im_raw.scale[1].value
            x = (args[1] - self.X0)*xScale
            y = (args[0] - self.Y0)*yScale

        # Calculations taken from sunpy.wcs.
        # First convert to heliocentric cartesian coordinates.
        rx, ry, rz = self._hpc_hcc(x, y)

        # Now convert to heliographic coordinates.
        lonh, lath = self._hcc_hg(rx, ry, rz)

        # Only add the instance attribute if it doesn't exist.
        if array and not hasattr(self, 'lonh') and not corners:
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

        print("Correcting line of sight magnetic field.")
        if array:
            try:
                lonh, lath = M.deg2rad(self.lonh), M.deg2rad(self.lath)
            except AttributeError:
                self.heliographic()
                lonh, lath = M.deg2rad(self.lonh), M.deg2rad(self.lath)
        else:
            lonh, lath = M.deg2rad(self.heliographic(args[0], args[1]))

        B0 = M.deg2rad(self.B0)
        L0 = M.deg2rad(self.L0)

        Xobs = M.cos(B0)*M.cos(L0)
        Yobs = M.cos(B0)*M.sin(L0)
        Zobs = M.sin(B0)

        corr_factor = (M.cos(lath)*M.cos(lonh)*Xobs
                + M.cos(lath)*M.sin(lonh)*Yobs
                + M.sin(lath)*Zobs)

        if array:
            self.im_corr = self.im_raw_u/corr_factor
            bad_ind = np.where(self.rg > self.rsun*np.sin(75.0*np.pi/180))
            self.im_corr[bad_ind] = np.nan
            return
        else:
            return self.im_raw.data[args[0], args[1]]/corr_factor

    def eoa(self, *args, array=True):
        """Takes in coordinates and returns the area of pixels on sun.

        Each pixel is projected onto the sun, and therefore pixels close to
        the limbs have vastly greater areas. This function uses a closed form
        solution to a spherical area integral to calulate the area based on
        the heliographic coordinate unit vectors of each corner of the pixel.
        We use these to calculate a solid angle of a pyramid with its apex
        at the center of the sun.
        """

        print ("Calculating element of area.")
        # Assume coordinate is in center of pixel.
        # Information on pixel standard is in this article.
        # http://www.aanda.org/component/article?access=bibcode&bibcode=&bibcode=2002A%2526A...395.1061GFUL
        if array:
            lon, lat = self.heliographic(corners=True)
            lon = lon*np.pi/180
            lat = lat*np.pi/180
            # Calculating unit vectors of pixel corners for solid angle.
            r1 = self._spherical_to_cartesian(lon, lat, 0, 0)
            r2 = self._spherical_to_cartesian(lon, lat, 1, 0)
            r3 = self._spherical_to_cartesian(lon, lat, 1, 1)
            r4 = self._spherical_to_cartesian(lon, lat, 0, 1)

        else:
            x = args[0]
            y = args[1]
            lonUL, latUL = self.heliographic(x - .5, y - .5)
            lonLL, latLL = self.heliographic(x + .5, y - .5)
            lonLR, latLR = self.heliographic(x + .5, y + .5)
            lonUR, latUR = self.heliographic(x - .5, y + .5)

            # Calculating unit vectors of pixel corners for solid angle.
            r1 = np.array([np.cos(np.deg2rad(latUL))*np.cos(np.deg2rad(lonUL)),
                            np.cos(np.deg2rad(latUL))*np.sin(np.deg2rad(lonUL)),
                            np.sin(np.deg2rad(latUL))])

            r2 = np.array([np.cos(np.deg2rad(latLL))*np.cos(np.deg2rad(lonLL)),
                            np.cos(np.deg2rad(latLL))*np.sin(np.deg2rad(lonLL)),
                            np.sin(np.deg2rad(latLL))])

            r3 = np.array([np.cos(np.deg2rad(latLR))*np.cos(np.deg2rad(lonLR)),
                            np.cos(np.deg2rad(latLR))*np.sin(np.deg2rad(lonLR)),
                            np.sin(np.deg2rad(latLR))])

            r4 = np.array([np.cos(np.deg2rad(latUR))*np.cos(np.deg2rad(lonUR)),
                            np.cos(np.deg2rad(latUR))*np.sin(np.deg2rad(lonUR)),
                            np.sin(np.deg2rad(latUR))])

        # Calculate solid angle of pixel based on a pyrimid shaped polygon.
        # See http://planetmath.org/solidangleofrectangularpyramid
        cross1 = M.cross(r1, r2, axis=0)
        cross2 = M.cross(r3, r4, axis=0)
        numerator1 = M.dot(cross1, r3)
        numerator2 = M.dot(cross2, r1)
        solid_angle1 = 2*M.arctan2(numerator1,
                (M.dot(r1, r2) + M.dot(r2, r3) + M.dot(r3, r1) + 1))
        solid_angle2 = 2*M.arctan2(numerator2, 
                (M.dot(r3, r4) + M.dot(r4, r1) + M.dot(r3, r1) + 1))
        solid_angle = solid_angle1 + solid_angle2

        r = self.RSUN_METERS*100 # Convert to centimeters
        if array:
            self.area = abs((r**2)*solid_angle)
            ind = np.where(self.Rg[1:len(self.Rg)-1, 1:len(self.Rg)-1] > self.rsun)
            del self.Rg
            self.area[ind] = np.nan
            return
        else:
            if self.rg > self.rsun:
                return np.nan
            else:    
                return np.abs((r**2)*solid_angle)

    def magnetic_flux(self, *args, array=True, raw_field=False):
        """Takes in coordinates and returns magnetic flux of pixel."""
        if array:
            try:
                area = self.area
            except AttributeError:
                self.eoa()
                area = self.area

            if raw_field:
                field = self.im_raw_u
                print ("Calculating raw magnetic flux.")
                self.mflux_raw = area*field
                return
            else:
                try:
                    field = self.im_corr
                except AttributeError:
                    self.los_corr()
                    field = self.im_corr
                finally:
                    print ("Calculating corrected magnetic flux.")
                    self.mflux_corr = area*field
                    return

        else:
            return self.eoa(*args)*self.los_corr(*args)
    
    def _grid(self, corners=False):
        """Create an xy grid of coordinates for heliographic array.

        Uses meshgrid. If corners is selected, this function will shift
        the array by half a pixel in both directions so that the corners
        of the normal array can be accessed easily.
        """
        # Retrieve integer dimensions and create arrays holding
        # x and y coordinates of each pixel
        xDim = np.int(np.floor(self.im_raw.dimensions[0].value))
        yDim = np.int(np.floor(self.im_raw.dimensions[1].value))
        
        if corners:
            xRow = (np.arange(0, xDim + 1) - self.X0 - 0.5)*self.xScale
            yRow = (np.arange(0, yDim + 1) - self.Y0 - 0.5)*self.yScale
            xg, yg = M.meshgrid(xRow, yRow)
            rg = M.sqrt(xg**2 + yg**2)
            self.Rg = rg
        else:
            xRow = (np.arange(0, xDim) - self.X0)*self.xScale
            yRow = (np.arange(0, yDim) - self.Y0)*self.yScale
            xg, yg = M.meshgrid(xRow, yRow)
            rg = M.sqrt(xg**2 + yg**2)
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
        x *= np.deg2rad(1)/3600.0
        y *= np.deg2rad(1)/3600.0

        q = self.dsun * M.cos(y) * M.cos(x)
        distance = q**2 - self.dsun**2 + self.RSUN_METERS**2
        distance = q - M.sqrt(distance)
        
        rx = distance * M.cos(y) * M.sin(x)
        ry = distance * M.sin(y)
        rz = M.sqrt(self.RSUN_METERS**2 - rx**2 - ry**2)

        return rx, ry, rz

    def _hcc_hg(self, x, y, z):
        """Converts hcc coordinates to Stonyhurst heliographic. 

        x - x coordinate in meters
        y - y coordinate in meters 
        z - z coordinate in meters
        Calculations taken and shortened
        from sunpy.wcs.
        """
        cosb = M.cos(M.deg2rad(self.B0))
        sinb = M.sin(M.deg2rad(self.B0))

        hecr = M.sqrt(x**2 + y**2 + z**2)
        hgln = M.arctan2(x, z*cosb - y*sinb) \
                + M.deg2rad(self.L0)
        hglt = M.arcsin((y * cosb + z * sinb)/hecr)

        return hgln*180/np.pi, hglt*180/np.pi

    def _dot(self, a, b):
        """Vectorized version of the dot product of two arrays.

        Wanted a dot product function that performed operations
        element-wise.
        """
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

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
        coslat = M.cos(lat)
        coslon = M.cos(lon)
        sinlat = M.sin(lat)
        sinlon = M.sin(lon)
        l = len(lat)
        Xar = coslat[i:l - 1 + i, j:l - 1 + j]*coslon[i:l - 1 + i, j:l - 1 + j]
        Yar = coslat[i:l - 1 + i, j:l - 1 + j]*sinlon[i:l - 1 + i, j:l - 1 + j]
        Zar = sinlat[i:l - 1 + i, j:l - 1 + j]
        r = np.ndarray(shape=(3,), dtype=np.object)
        r[0] = Xar
        r[1] = Yar
        r[2] = Zar
        return r