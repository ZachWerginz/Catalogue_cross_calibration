import os.path
import glob
import pickle
import datetime as dt
import numpy as np
from astropy.io import fits
import astropy.units as u
from zaw_coord import CRD
import sunpy.time
import sunpy.physics.differential_rotation as d
import psycopg2 as psy

psy.extensions.register_adapter(np.float32, psy._psycopg.AsIs)
DEC2FLOAT = psy.extensions.new_type(
    psy.extensions.DECIMAL.values,
    'DEC2FLOAT',
    lambda value, curs: float(value) if value is not None else None)
psy.extensions.register_type(DEC2FLOAT)

__authors__ = ["Zach Werginz", "Andres Munoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

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

def load_database():
    try:
        conn = psy.connect("dbname='cross_calibration' user='zwerginz' host='minerva' password='8raSp6tr'")
    except:
        print ("I am unable to connect to the database")
    return conn

def download_cc_data(i1, i2, n, tol1, tol2):
    conn = load_database()
    cur = conn.cursor()

    instrumentKey = {'512': 1, 'SPMG': 2, 'MDI': 3, 'HMI': 4}
    
    result = {}
    cur.execute("SELECT referencefluxdensity, secondaryfluxdensity, diskangle \
            FROM quadrangle q JOIN file a ON q.referencemag = a.id \
            JOIN file b ON q.secondarymag = b.id \
            WHERE fragmentationvalue = %s \
            AND a.instrument = %s AND b.instrument = %s \
            AND age(b.date, a.date) BETWEEN  INTERVAL %s AND  INTERVAL %s",
            (n, instrumentKey[i1.upper()], instrumentKey[i2.upper()], tol1, tol2))

    points = cur.fetchall()

    y, x, da = zip(*points)

    result['referenceFD'] = np.array(y)
    result['secondaryFD'] = np.array(x)
    result['diskangle'] = np.array(da)
    result['i1'] = i1
    result['i2'] = i2

    cur.execute("SELECT INTERVAL %s - INTERVAL %s", (tol1, tol2))
    result['timeDifference'] = cur.fetchone()[0]
    cur.close()

    return result

def date_defaults(instr):
    if instr == '512':
        return dt.datetime(1976, 1, 5), dt.datetime(1993, 4, 9)
    elif instr == 'spmg':
        return dt.datetime(1992, 4, 21), dt.datetime(1999, 12, 30)
    elif instr == 'mdi':
        return dt.datetime(1996, 4, 15), dt.datetime(2011, 4, 11)
    elif instr == 'hmi':
        return dt.datetime(2010, 4, 8), dt.datetime(2016, 7, 5)
    else:
        raise ValueError('Unrecognized instrument')

def get_header_date(f):
    hdulist = fits.open(f)

    for hdu in hdulist:
        try:
            time = sunpy.time.parse_time(hdu.header['DATE_OBS'])
        except KeyError:
            try:
                time = sunpy.time.parse_time(hdu.header['DATE-OBS'])
            except KeyError:
                try:
                    time = sunpy.time.parse_time(hdu.header['T_OBS'])
                except:
                    continue
    hdulist.close()

    return time

#Converts a standard date string into an instrument mission date.
def date2md(date, instr):
    return date.toordinal() - dateOffset(instr).toordinal()

#Converts an instrument mission date string into a standard date string.
def md2date(md, instr):
    return dt.datetime.fromordinal(md + dateOffset(instr).toordinal())

def CRD_read(date, instr):
    if not isinstance(date, dt.datetime):
        date = sunpy.time.parse_time(date)
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
    if not isinstance(date, dt.datetime):
        date = sunpy.time.parse_time(date)
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
        md = date2md(date, instr)
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
    files = glob.glob(searchspec)

    pdebug('searchspec: ' + searchspec)

    

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

def diff_rot(m1, m2):
    """Given two CRD objects, differentially rotate image 2 to match image 1

    Returns the rotation amount in degrees.
    """
    timeDiff = u.Quantity(
            (m1.im_raw.date - m2.im_raw.date).total_seconds(), 'second')
    rotation = d.diff_rot(timeDiff, m2.lath.v*u.deg, rot_type='snodgrass', frame_time='synodic')

    return rotation