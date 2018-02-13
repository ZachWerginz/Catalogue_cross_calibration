"""Provides utility functions for i/o.

This module provides utility functions designed for easy access to the Postgres database and dealing with magnetograms.
There are also functions to convert between data types like the integer mission day and a datetime object.

Attributes:
    data_root (str): This is the root level of where the fits files are being stored in.
    debug (bool): whether or not to show debugging messages

"""

import datetime as dt
import glob
import os.path
import pickle

import astropy.units as u
import numpy as np
import psycopg2 as psy
import sunpy.physics.differential_rotation as d
import sunpy.time
from astropy.io import fits

from coord import CRD

psy.extensions.register_adapter(np.float32, psy._psycopg.AsIs)
DEC2FLOAT = psy.extensions.new_type(
    psy.extensions.DECIMAL.values,
    'DEC2FLOAT',
    lambda value, curs: np.float32(value) if value is not None else None)
psy.extensions.register_type(DEC2FLOAT)

__authors__ = ["Zach Werginz", "Andrés Muñoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]

data_root = 'H:'
debug = False


def date_offset(instr):
    """Returns a datetime object of the instrument start year."""
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
    """Connects to and loads the Postgres database located on server."""
    try:
        conn = psy.connect("dbname=cross_calibration user=zwerginz host=192.168.86.137")
    except DatabaseError:
        print("Unable to connect to the database")
        return
    return conn


def download_cc_data(i1, i2, n, tol1, tol2):
    """Downloads cross-calibration data from database for a given time interval and set of instruments.

    Args:
        i1 (str): the reference instrument
        i2 (str): the secondary instrument
        n (int): fragmentation parameter
        tol1 (str): earliest time difference between two magnetograms
        tol2 (str): latest time difference between two magnetograms

    Returns:
        dict: dictionary of results in array form

    Example:
        >>> r_25 = u.download_cc_data('spmg', 'spmg', 25, '23 hours', '25 hours')
        >>> r_25
        {'diskangle': array([...]), 'i1': 'spmg', 'i2': 'spmg', 'n': 25, 'referenceFD': array([...]),
        'secondaryFD': array([...]), 'timeDifference': datetime.timedelta(1)}
    """
    conn = load_database()
    cur = conn.cursor("server_side")
    fetchlimit = 10000000

    instrument_key = {'512': 1, 'SPMG': 2, 'MDI': 3, 'HMI': 4, 'SIM': 5, 'SIM2': 6}
    
    result = {}
    points = [0]
    x = []
    y = []
    da = []
    cur.execute("SELECT referencefluxdensity, secondaryfluxdensity, diskangle \
                    FROM quadrangle q JOIN file a ON q.referencemag = a.id \
                    JOIN file b ON q.secondarymag = b.id \
                    WHERE fragmentationvalue = %s \
                    AND a.instrument = %s AND b.instrument = %s \
                    AND age(b.date, a.date) BETWEEN  INTERVAL %s AND  INTERVAL %s",
                (n, instrument_key[i1.upper()], instrument_key[i2.upper()], tol1, tol2))

    while points:
        points = cur.fetchmany(fetchlimit)
        x.extend([s[0] for s in points])
        y.extend([s[1] for s in points])
        da.extend([np.float32(s[2]) for s in points])
    cur.close()

    result['reference_fd'] = np.array(x)
    result['secondary_fd'] = np.array(y)
    result['disk_angle'] = np.array(da)
    result['i1'] = i1
    result['i2'] = i2
    result['n'] = n

    cur = conn.cursor()
    cur.execute("SELECT (INTERVAL %s)*SIGN(EXTRACT(epoch from INTERVAL %s)) + \
                (INTERVAL %s)*SIGN(EXTRACT(epoch from INTERVAL %s))",
                (tol2, tol2, tol1, tol1))
    result['timeDifference'] = cur.fetchone()[0]/2
    cur.close()

    return result


def date_defaults(instr):
    """Returns a tuple pair of dates denoting the start and end dates of the instrument files."""
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


def date2md(date, instr):
    """Converts a standard date string into an integer instrument mission date."""
    return date.toordinal() - date_offset(instr).toordinal()


def md2date(md, instr):
    """Converts an instrument mission date string into a standard date string."""
    return dt.datetime.fromordinal(md + date_offset(instr).toordinal())


def load_sim(fn):
    """Loads simulation maps and orients them properly in memory."""
    sim_map = np.fromfile(fn, dtype=np.float32).reshape(512, 1024)
    return np.flipud(sim_map)


def search_file(date, instr, auto=True):
    """Searches for a file with a given date and instrument and returns a list of filepaths.

    Args:
        date (str) (obj): date of desired file - datetime or str
        instr (str): instrument of desired file
        auto (bool): whether to autoselect a file from a group of similar files for one particular day

    Returns:
        list: list of files or singular file

    """
    if not isinstance(date, dt.datetime):
        date = sunpy.time.parse_time(date)
    # Set defaults
    subdir = ''
    fn0 = instr.upper()
    filename = '*%s*.fits' % date.strftime('%Y%m%d')

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
        filename = 'fd_M_96m_01d.%d.0*.fits' % md
        
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
    """Chooses the best file from a MDI mission day based on missing values and timing interval.

    Args:
        f (list): list of MDI files

    Returns:
        str: singular filepath for best magnetogram

    """
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


def pdebug(string):
    """Used for debugging - prints messages."""
    if debug:
        print(string)


def diff_rot(m1, m2):
    """Given two CRD objects, differentially rotate image 2 to match image 1

    Args:
        m1 (obj): first image to rotate to
        m2 (obj): second image to be rotated differentially

    Returns:
        rotation: rotation array containing values to add to longitude

    """
    time_diff = u.Quantity(
            (m1.im_raw.date - m2.im_raw.date).total_seconds(), 'second')
    rotation = d.diff_rot(time_diff, m2.lath.v*u.deg, rot_type='snodgrass', frame_time='synodic')

    if np.nanmean(rotation).value > 90:
        rotation -= 360*u.deg

    return rotation
