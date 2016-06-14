import sunpy.map
class KPVT(sunpy.map.GenericMap):

    def __init__(self, data, header, **kwargs):

        super(KPVT, self).__init__(data, header, **kwargs)

        # Any KPVT Instrument specific keyword manipulation
        self.meta['detector'] = "KPVT"
        self.meta['nickname'] = "KPVT"
        self._nickname = str(self.detector) + "" + str(self.measurement)
        if self.meta['cunit1'] == 'ARC-SEC':
            self.meta['cunit1'] = 'arcsec'
        if self.meta['cunit2'] == 'ARC-SEC':
            self.meta['cunit2'] = 'arcsec'
        
        self.meta['pc2_1'] = 1
        self.meta['pc1_2'] = 1

        self.data = self.data[2,:,:]

    # Specify a classmethod that determines if the data-header pair matches
    # the new instrument
    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Determines if header corresponds to an KPVT image"""
        return header.get('instrume') == '512-CH-MAG'
