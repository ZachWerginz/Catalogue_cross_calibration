"""Provides the functions necessary for analyzing cross calibration between instruments.

Attributes:
    Pair (namedtuple): sort of a function definition to make sure MockCRD works with sunpy

"""

import datetime as dt
import itertools
import random
from collections import namedtuple

import astropy.units as units
import numpy as np
import psycopg2 as psy
from scipy.interpolate import griddata

import quadrangles as quad
import uncertainty.measurement as mnp
import util as u
from coord import CRD

Pair = namedtuple('Pair', 'x y')

psy.extensions.register_adapter(np.float32, psy._psycopg.AsIs)
DEC2FLOAT = psy.extensions.new_type(
    psy.extensions.DECIMAL.values,
    'DEC2FLOAT',
    lambda value, curs: float(value) if value is not None else None)
psy.extensions.register_type(DEC2FLOAT)

__authors__ = ["Zach Werginz", "Andrés Muñoz-Jaramillo"]
__email__ = ["zachary.werginz@snc.edu", "amunozj@gsu.edu"]


def get_file_list(i1, i2, tol1, tol2):
    """Returns a list of valid file combinations.

    Searches for files within tol1 of each other as a lower bound or tol2 as an upper bound.

    Args:
        i1 (str): reference instrument
        i2 (str): secondary instrument
        tol1 (str): minimum time difference between magnetograms
        tol2 (str): maximum time difference between magnetograms

    Returns:
        list: list of filename pairs that satisfy the condition in tuple form (file1, file2)

    """
    instrument_key = {'512': 1, 'SPMG': 2, 'MDI': 3, 'HMI': 4, 'SIM': 5, 'SIM2': 6}

    conn = u.load_database()
    cur = conn.cursor()
    cur.execute("SELECT a.filepath AS f1, b.filepath AS f2 \
                FROM file_time_diff main \
                JOIN file a ON main.file1 = a.id \
                JOIN file b ON main.file2 = b.id \
                WHERE a.instrument = %s AND b.instrument = %s \
                AND difference BETWEEN INTERVAL %s \
                AND INTERVAL %s;", (instrument_key[i1.upper()],
                                    instrument_key[i2.upper()], tol1, tol2))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results


def prepare_magnetograms(f1, f2, raw_remap=False, downscale=False):
    """Calculate heliographic information and apply differential rotation.

    The standard is to interpolate the smaller resolution magnetogram into the larger one unless downscale is chosen.

    Args:
        f1 (str): the first file
        f2 (str): the second file
        raw_remap (bool): defaults to False, uses raw flux density instead of corrected one
        downscale (bool): defaults to False, will downscale larger resolution to smaller one

    Returns:
        object: returns m1 and m2 as a tuple with rotations applied

    """
    print(f1)
    print(f2)
    mgnt1 = CRD(f1)
    mgnt2 = CRD(f2)
    print(mgnt1.im_raw.date)
    print(mgnt2.im_raw.date)
    mgnt1.heliographic()
    mgnt2.heliographic()
    mgnt1.magnetic_flux()
    mgnt2.magnetic_flux()
    # Apply differential Rotation
    if mgnt2.im_raw.dimensions[0].value > mgnt1.im_raw.dimensions[0].value and not downscale:
        rotation = u.diff_rot(mgnt2, mgnt1)
        mgnt1.lonhRot = mgnt1.lonh + rotation.value
        interpolate_remap(mgnt2, mgnt1, raw_remap)
        return mgnt2, mgnt1
    else:
        rotation = u.diff_rot(mgnt1, mgnt2)
        mgnt2.lonhRot = mgnt2.lonh + rotation.value
        interpolate_remap(mgnt1, mgnt2, raw_remap)
        return mgnt1, mgnt2


def interpolate_remap(m1, m2, raw=False):
    """Perform interpolation of m2 into coordinate system of m1.

    Args:
        m1 (obj): reference magnetogram
        m2 (obj): secondary magnetogram
        raw (bool): defaults to False, toggles use of corrected flux density

    """
    if raw:
        v2 = m2.im_raw.data.flatten()
    else:
        v2 = m2.im_corr.v.flatten()

    x2 = m2.lonhRot.v.flatten()
    y2 = m2.lath.v.flatten()
    x1 = m1.lonh.v.flatten()
    y1 = m1.lath.v.flatten()
    dim1 = m1.im_raw.dimensions

    latitude_mask2 = np.where(np.abs(y2) < 50)
    latitude_mask1 = np.where(np.abs(y1) < 50)
    minimum = max(np.nanmin(x2[latitude_mask2]), np.nanmin(x1[latitude_mask1]))
    maximum = min(np.nanmax(x2[latitude_mask2]), np.nanmax(x1[latitude_mask1]))

    ind2 = (np.isfinite(x2) * np.isfinite(y2) * np.isfinite(v2) * (x2 > minimum) * (x2 < maximum))
    ind1 = (np.isfinite(x1) * np.isfinite(y1) * (x1 > minimum) * (x1 < maximum))

    interp_data = griddata(np.array(x2[ind2], y2[ind2]), v2[ind2], (x1[ind1], y1[ind1]), method='cubic')
    new_m2 = np.full((int(dim1[0].value), int(dim1[1].value)), np.nan)

    new_m2.ravel()[ind1] = interp_data
    new_m2[m1.rg > m1.par['rsun'] * np.sin(75.0 * np.pi / 180)] = np.nan

    m2.remap = new_m2


def run_multiple_n(mgnt):
    """Takes mgnt and returns list of different fragmented quadrangles."""
    n_list = [i for i in range(10, 3100, 100)]

    n_dict_length = {}

    for n in n_list:
        num_blocks = quad.fragment_single(m, n)
        n_dict_length[n] = len(num_blocks)
    return n_dict_length


def upload_quadrangles(conn, b, working_files, sim=False):
    """Upload the fragmentation information to the postgres database.

    Args:
        conn (obj): psycopg2 connection object
        b (list): the list of qudrangles to upload
        working_files (tuple): the set of files the comparison was done on
        sim (bool): defaults to False, will use different keywords if True

    """
    f1, f2 = get_file_id(conn, working_files)
    cur = conn.cursor()
    if sim:
        try:
            for quadrangle in b:
                if np.isnan(quadrangle.fluxDensity) or np.isnan(quadrangle.fluxDensity2):
                    continue
                cur.execute("INSERT INTO quadrangle (referencemag, secondarymag, diskangle, area,\
                            referencefluxdensity, secondaryfluxdensity, fragmentationvalue)\
                            VALUES (%s, %s, %s, %s, %s, %s, %s)", (f1, f2, quadrangle.diskAngle, quadrangle.area,
                            quadrangle.fluxDensity, quadrangle.fluxDensity2, quadrangle.fragmentationValue))

            cur.execute("INSERT INTO uniquepairs VALUES (%s, %s, %s)", (f1, f2, b[0].fragmentationValue))
        except Exception as e:
            print("Could not upload completely to database.")
            conn.rollback()
            cur.close()
            return
    else:
        try:
            for quadrangle in b:
                if np.isnan(quadrangle.fluxDensity) or np.isnan(quadrangle.fluxDensity2):
                    continue
                cur.execute("INSERT INTO quadrangle (referencemag, secondarymag, diskangle, area,\
                            referencefluxdensity, secondaryfluxdensity, fragmentationvalue) VALUES "
                            "(%s, %s, %s, %s, %s, %s, %s)", (f1, f2, np.float32(quadrangle.diskAngle.v),
                                                             quadrangle.area.v, quadrangle.fluxDensity, quadrangle.
                                                             fluxDensity2, quadrangle.fragmentationValue))

            cur.execute("INSERT INTO uniquepairs VALUES (%s, %s, %s)", (f1, f2, b[0].fragmentationValue))
        except Exception as e:
            print("Could not upload completely to database.")
            conn.rollback()
            cur.close()
            return

    conn.commit()
    cur.close()


def get_file_id(conn, files):
    """Return file ids for filenames from database.

    Args:
        conn (obj): the psycopg2 connection object
        files (tuple): tuple containing the filepaths

    Returns:
        tuple: tuple of id numbers for file

    """
    cur = conn.cursor()
    cur.execute("SELECT id FROM file WHERE filepath = %s", (files[0],))
    file_id1 = cur.fetchone()[0]
    cur.execute("SELECT id FROM file WHERE filepath = %s", (files[1],))
    file_id2 = cur.fetchone()[0]
    cur.close()

    return file_id1, file_id2


def prepare_simulation(fn1, fn2):
    """Prepares two simulation maps for processing and return them.

    Args:
        fn1 (str): filename of reference magnetogram
        fn2 (str): filename of secondary magnetogram

    Returns:
        object: returns sim1 and sim2 prepared for fragmentation

    """
    class MockCRD:
        """Used as a class to replicate what CRD does temporarily."""

        def __init__(self, filename):
            class Im:
                """Quick class to replicate what CRD does temporarily."""

                pass

            self.im_raw = Im()
            self.im_raw.data = u.load_sim(filename)
            self.im_raw.dimensions = Pair(*self.im_raw.data.shape * units.pixel)
            self.im_raw.instrument = 'SIM'
            self.im_raw.date = dt.datetime.strptime((filename.split('.')[0].split('\\')[-1] + filename.split('.')[1]),
                                                    '%Y%m%d%H')
            self.im_corr = mnp.Measurement(self.im_raw.data, self.im_raw.data * 0)
            lat_space = np.linspace(90, -90, self.im_raw.data.shape[0] + 2)
            lon_space = np.linspace(-180, 180, self.im_raw.data.shape[1] + 2)
            self.lath, self.lonh = np.meshgrid(lat_space[1:-1], lon_space[1:-1], indexing='ij')
            self.lath = mnp.Measurement(self.lath, np.zeros(self.lath.shape))
            self.lonh = mnp.Measurement(self.lonh, np.zeros(self.lonh.shape))
            self.rg = np.array(self.im_raw.data, copy=True)
            self.rg[:] = np.nan
            self.area = np.array(self.im_raw.data, copy=True)
            self.area[:] = np.nan
            self.par = {'rsun': 960}

    sim1 = MockCRD(fn1)
    sim2 = MockCRD(fn2)
    rotation = u.diff_rot(sim2, sim1)
    sim2.lonhRot = sim1.lonh + rotation.value
    interpolate_remap(sim1, sim2, False)
    sim2.remap = sim2.im_raw.data

    return sim1, sim2


def transform_blocks_to_dict(blocks, fragmentation_parameter):
    """Transform a list of quadrangles into a dictionary with condensed information

    Args:
        blocks: the list of quadrangles containing flux and area information
        fragmentation_parameter: the fragmentation parameter used to calculate blocks

    Returns:
        dict: a dictionary containing condensed arrays of quadrangle information

    """
    # check for nested list for multiple comparable days and flatten it if so
    if any(isinstance(i, list) for i in blocks):
        blocks = list(itertools.chain.from_iterable(blocks))
    result = {'i1': blocks[0].i1, 'i2': blocks[0].i2, 'n': fragmentation_parameter}
    reference_fd = []
    secondary_fd = []
    disk_angle = []
    for quadrangle in blocks:
        reference_fd.append(quadrangle.fluxDensity)
        secondary_fd.append(quadrangle.fluxDensity2)
        disk_angle.append(quadrangle.diskAngle)
    result['reference_fd'] = np.array(reference_fd)
    result['secondary_fd'] = np.array(secondary_fd)
    result['disk_angle'] = np.array(disk_angle)

    return result


def compare_day(i1, i2, file1, file2, n):
    """Compare two days and return a list of quadrangles containing flux density information.

    Args:
        i1 (str): reference instrument
        i2 (str): secondary instrument
        file1 (str): filepath of reference magnetogram
        file2 (str): filepath of secondary magnetogram
        n (int): fragmentation parameter

    Returns:
        list: list of quadrangles for the fragmentation
    """
    if 'sim' in i1 or 'sim' in i2:
        m1, m2 = prepare_simulation(file1, file2)
    else:
        m1, m2 = prepare_magnetograms(file1, file2)

    blocks = quad.fragment_multiple(m1, m2, n)

    return blocks


def analyze_random_sample(i1, i2, tol1, tol2, n=25, passes=1, upload=False, unique_days=False):
    """Compare random samples of magnetograms between tolerance limits.

    This function will compare magnetograms from i1 and i2 that are between tol1 and tol2 time difference away from
    each other. It will do this the specified number of passes and aggregate data.

    Args:
        i1 (str): reference instrument
        i2 (str): secondary instrument
        tol1 (str): minimum time difference between magnetograms
        tol2 (str): maximum time difference between magnetograms
        n (int, optional): fragmentation parameter, defaults to 25
        passes (int, optional): number of pairs to compare, defaults to 1
        upload (bool): choose to upload to database, defaults to False
        unique_days (bool): defaults to False, sets limit to one pair a day

    Returns:
        dict (optional): condensed summary dictionary of block information if not uploaded to database

    """
    file_matches = get_file_list(i1, i2, tol1, tol2)
    conn = u.load_database()
    blocks_list = []
    day_matches = []
    i = 0
    while True:
        if i > passes:
            break
        try:
            choice_int = int(random.random() * len(file_matches))
            working_files = file_matches[choice_int]
            file_ids = get_file_id(conn, working_files)
            if upload:
                cur = conn.cursor()
                # search for files already in database
                cur.execute("SELECT * FROM uniquepairs\
                                    WHERE referencemag = %s \
                                    AND secondarymag = %s\
                                    AND fragmentationvalue = %s", (file_ids[0], file_ids[1], n))
                if cur.fetchone() is not None:
                    cur.close()
                    del file_matches[choice_int]
                    continue

                # search for date if unique
                if unique_days:
                    cur.execute("SELECT date FROM file WHERE id = %s OR id = %s", (file_ids[0], file_ids[1]))
                    day1, day2 = cur.fetchone()
                    if (day1.toordinal(), day2.toordinal()) in day_matches:
                        del file_matches[choice_int]
                        continue
                    else:
                        day_matches.append((day1.toordinal(), day2.toordinal()))

                cur.close()
                blocks = compare_day(i1, i2, working_files[0], working_files[1], n)
                upload_quadrangles(conn, blocks, working_files)
            else:
                blocks = compare_day(i1, i2, working_files[0], working_files[1], n)
                blocks_list.append(blocks)
            del file_matches[choice_int]
        except ValueError:
            continue
        i += 1
    if not upload:
        return transform_blocks_to_dict(blocks_list, n)
