"""KPVT Map subclass definitions"""

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

import numpy as np
import sunpy.map
from matplotlib import colors

from astropy.units import Quantity

from sunpy.map import GenericMap
from sunpy.sun import constants
from sunpy.sun import sun
from sunpy.cm import cm

__all__ = ["512", "SPMG"]

class 512(sunpy.map.GenericMap):
    """KPVT 512 Channel Image Map.

    """

    def __init__(self, data, header, **kwargs):

        super(512, self).__init__(data, header, **kwargs)

        # Any 512 Instrument specific keyword manipulation
        self.meta['detector'] = "512"
        self._fix_dsun()
        self._nickname = str(self.detector) + "" + str(self.measurement)
        if self.meta['cunit1'] == 'ARC-SEC':
            self.meta['cunit1'] = 'arcsec'
        if self.meta['cunit2'] == 'ARC-SEC':
            self.meta['cunit2'] = 'arcsec'
        
        self.meta['pc2_1'] = 1
        self.meta['pc1_2'] = 1

        self.data = self.data[2,:,:]

     def _fix_dsun(self):
        """ Solar radius in arc-seconds at 1 au
            previous value radius_1au = 959.644
            radius = constants.average_angular_size
            There are differences in the keywords in the test FITS data and in
            the Helioviewer JPEG2000 files.  In both files, MDI stores the
            the radius of the Sun in image pixels, and a pixel scale size.
            The names of these keywords are different in the FITS versus the
            JP2 file.  The code below first looks for the keywords relevant to
            a FITS file, and then a JPEG2000 file.  For more information on
            MDI FITS header keywords please go to http://soi.stanford.edu/,
            http://soi.stanford.edu/data/ and
            http://soi.stanford.edu/magnetic/Lev1.8/ .
        """
        scale = self.meta.get('xscale', self.meta.get('cdelt1'))
        radius_in_pixels = self.meta.get('r_sun', self.meta.get('radius'))
        radius = scale * radius_in_pixels
        self.meta['radius'] = radius

        if not radius:
            # radius = sun.angular_size(self.date)
            self.meta['dsun_obs'] = constants.au
        else:
            self.meta['dsun_obs'] = _dsunAtSoho(self.date, radius)
    # Specify a classmethod that determines if the data-header pair matches
    # the new instrument
    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Determines if header corresponds to an 512 image"""
        return header.get('instrume') == '512-CH-MAG'
