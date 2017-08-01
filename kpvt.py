"""KPVT Map subclass definitions"""

from __future__ import absolute_import, print_function, division
from sunpy.map import GenericMap
import astropy.units as u
from sunpy.sun import sun
from collections import namedtuple

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

Pair = namedtuple('Pair', 'x y')

__all__ = ["Ch512Map", "SPMGMap"]


# noinspection PyUnresolvedReferences
class Ch512Map(GenericMap):
    """KPVT 512 Channel Image Map.

    """

    def __init__(self, data, header, **kwargs):

        GenericMap.__init__(self, data[2, :, :], header, **kwargs)

        # Any Ch512 Instrument specific keyword manipulation
        self.meta['detector'] = "512"
        self._nickname = str(self.detector) + "" + str(self.measurement)
        if self.meta['cunit1'] == 'ARC-SEC':
            self.meta['cunit1'] = 'arcsec'
        if self.meta['cunit2'] == 'ARC-SEC':
            self.meta['cunit2'] = 'arcsec'

        self.meta['pc2_1'] = 0
        self.meta['pc1_2'] = 0
        self.meta['B0'] = self.meta['EPH_B0']
        self.meta['L0'] = self.meta['EPH_L0']
        del self.meta['eph_b0']
        del self.meta['eph_l0']

    def __getitem__(self, key):
        raise NotImplementedError(
            "The ability to index Map by physical coordinate is not yet implemented.")

    @property
    def meta(self):
        return super(Ch512Map, self).meta()

    @property
    def scale(self):
        return Pair(self.meta['cdelt1'] * self.spatial_units.x / u.pixel * self.meta['CRR_SCLX'],
                    self.meta['cdelt2'] * self.spatial_units.y / u.pixel * self.meta['CRR_SCLY'])

    @property
    def rsun_obs(self):
        """KPVT Magnetograms use a different keyword for distance to sun."""
        return self.meta['EPH_R0'] * u.arcsec

    @property
    def dsun(self):
        """KPVT at earth."""
        dsun = sun.sunearth_distance(self.date).to(u.m)
        return u.Quantity(dsun, 'm')

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Determines if header corresponds to an 512 Channel image."""
        return header.get('instrume') == '512-CH-MAG'


class SPMGMap(GenericMap):
    """KPVT SPMG Channel Image Map.

    """

    def __init__(self, data, header, **kwargs):

        GenericMap.__init__(self, data[5, :, :], header, **kwargs)

        # Any Ch512 Instrument specific keyword manipulation
        if self.meta['cunit1'] == 'ARC-SEC':
            self.meta['cunit1'] = 'arcsec'
        if self.meta['cunit2'] == 'ARC-SEC':
            self.meta['cunit2'] = 'arcsec'
        self.meta['detector'] = "SPMG"

        self.meta['pc2_1'] = 0
        self.meta['pc1_2'] = 0
        self.meta['B0'] = self.meta['EPH_B0']
        self.meta['L0'] = self.meta['EPH_L0']
        del self.meta['eph_b0']
        del self.meta['eph_l0']

    def __getitem__(self, key):
        raise NotImplementedError(
            "The ability to index Map by physical coordinate is not yet implemented.")

    @property
    def meta(self):
        return super(SPMGMap, self).meta()

    @property
    def rsun_obs(self):
        """ KPVT Magnetograms use a different keyword for distance to sun"""
        return self.meta['EPH_R0'] * u.arcsec

    @property
    def dsun(self):
        """ KPVT at earth"""
        dsun = sun.sunearth_distance(self.date).to(u.m)
        return u.Quantity(dsun, 'm')

    @property
    def scale(self):
        return Pair(self.meta['cdelt1'] * self.spatial_units.x / u.pixel * self.meta['CRR_SCLX'],
                    self.meta['cdelt2'] * self.spatial_units.y / u.pixel * self.meta['CRR_SCLY'])

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Determines if header corresponds to an 512 Channel image"""
        return header.get('instrume') == 'SPECTROMAGNETOGRAPH'
