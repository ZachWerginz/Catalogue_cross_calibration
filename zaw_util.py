import os.path
import glob
import datetime as dt
from astropy.io import fits
from zaw_coord import CRD

data_root = 'H:'
debug = False

def dateOffset(instr):
    if instr == 'spmg':
        year = 1990
    elif instr == 'mdi':
        year = 1993
    elif instr == 'hmi':
        year = 2009
    else:
        year = 1970
    
    return dt.date(year, 1, 1)

#Converts a standard date string into an instrument mission date.
def date2md(date, instr):
    return date.toordinal() - dateOffset(instr).toordinal()

#Converts an instrument mission date string into a standard date string.
def md2date(md, instr):
    return dt.datetime.fromordinal(md + dateOffset(instr).toordinal())

def CRD_read(date, instr):
    try:
        filename = search_file(date, instr)
    except IOError:
        return -1

    print(filename)
    
    try:
        mgnt = CRD(filename)
    except:
        return -1
    mgnt.heliographic()    
    mgnt.magnetic_flux()
    mgnt.magnetic_flux(raw_field=True)
    mgnt.date = mgnt.im_raw.date
    mgnt.md = date2md(date, instr)

    return mgnt

def search_file(date, instr, auto=True):
    # Set defaults
    subdir = ''
    fn0 = instr.upper()
    filename ='*%s*.fits' % date.strftime('%Y%m%d')

    # Set overrides
    if instr == '512':
        fn0 = 'KPVT'
        subdir = '%d%02d' % (date.year - 1900, date.month)
        filename = '*' + date.strftime('%Y%m%d') + '*.fits'

    elif instr == 'spmg':
        subdir = '%d%02d' % (date.year - 1900, date.month)

    elif instr == 'mdi':
        md = date2md(date, instr) + 1
        subdir = os.path.join(
                str(date.year)
                , 'fd_M_96m_01d.%06d' % md
        )
        filename ='fd_M_96m_01d.%d.0*.fits' % md
        
    elif instr == 'hmi':
        pass

    else:
        raise ValueError('Unrecognized instrument')

    # Execute
    searchspec = os.path.join(data_root, fn0, subdir, filename)
    pdebug('searchspec: ' + searchspec)

    files = glob.glob(searchspec)

    if not files:
        raise IOError('File not found')

    if instr == 'mdi' and auto:
        return mdi_file_choose(files)
    elif auto:
        return files[-1]
    else:
        return files

def mdi_file_choose(f):
    best = f[-1]    # default to last element
    ival = 0
    mv = 100000
    for x in f:     # but try to find a better match
        pdebug("mdi_file_choose - option: " + x)
        m = fits.open(x, mode='update')
        if 'INSTRUME' not in m[0].header.keys():
            m[0].header.set('instrume', 'MDI')
            m.flush()
        try:
            intv = m[0].header['INTERVAL']
            if intv == '':
                intv = 0
            else:
                intv = int(intv)
            if intv >= ival:
                if int(m[0].header['MISSVALS']) < mv:
                    best = x
                    ival = m[0].header['INTERVAL']
                    mv = m[0].header['MISSVALS']
        except KeyError:
            continue
        finally:
            m.close()

    pdebug("mdi_file_choose - selected: " + best)
    return best

def pdebug(str):
    if debug:
        print(str)