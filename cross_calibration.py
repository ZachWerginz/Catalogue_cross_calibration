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


def fix_longitude(f1, f2, raw_remap=False, downscale=False):
    """
    Will shift the longitude of second magnetogram to match the first.

    Uses differential rotation to update the second magnetogram's longitude.
    Then, afterwards it will shift the longitudes to match the first,
    considering they will only be off by some constant.
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
    if raw:
        v2 = m2.im_raw.data.flatten()
    else:
        v2 = m2.im_corr.v.flatten()

    x2 = m2.lonhRot.v.flatten()
    y2 = m2.lath.v.flatten()
    x1 = m1.lonh.v.flatten()
    y1 = m1.lath.v.flatten()
    dim1 = m1.im_raw.dimensions

    latitudeMask2 = np.where(np.abs(y2) < 50)
    latitudeMask1 = np.where(np.abs(y1) < 50)
    minimum = max(np.nanmin(x2[latitudeMask2]), np.nanmin(x1[latitudeMask1]))
    maximum = min(np.nanmax(x2[latitudeMask2]), np.nanmax(x1[latitudeMask1]))

    ind2 = (np.isfinite(x2) * np.isfinite(y2) * np.isfinite(v2) * (x2 > minimum) * (x2 < maximum))
    ind1 = (np.isfinite(x1) * np.isfinite(y1) * (x1 > minimum) * (x1 < maximum))

    interp_data = griddata((x2[ind2], y2[ind2]), v2[ind2], (x1[ind1], y1[ind1]), method='cubic')
    new_m2 = np.full((int(dim1[0].value), int(dim1[1].value)), np.nan)

    new_m2.ravel()[ind1] = interp_data
    new_m2[m1.rg > m1.par['rsun']*np.sin(75.0*np.pi/180)] = np.nan

    m2.remap = new_m2


def run_multiple_n(m):
    """Takes mgnt and returns list of different fragmented quadrangles."""
    nList = [i for i in range(10, 3100, 100)]

    n_dict_length = {}

    for n in nList:
        N = quad.fragment_single(m, n)
        n_dict_length[n] = len(N)
    return n_dict_length


def upload_quadrangles(conn, b, workingFiles, sim=False):
    f1, f2 = get_file_id(conn, workingFiles)
    print(f1, f2)
    cur = conn.cursor()
    if sim:
        try:
            for quadrangle in b:
                if np.isnan(quadrangle.fluxDensity) or np.isnan(quadrangle.fluxDensity2):
                    continue
                cur.execute("INSERT INTO quadrangle\
                    (referencemag, secondarymag, diskangle, area,\
                    referencefluxdensity, secondaryfluxdensity, fragmentationvalue)\
                    VALUES\
                    (%s, %s, %s, %s, %s, %s, %s)",
                    (f1, f2, quadrangle.diskAngle, quadrangle.area,
                    quadrangle.fluxDensity, quadrangle.fluxDensity2, quadrangle.fragmentationValue))

            cur.execute("INSERT INTO uniquepairs\
                    VALUES (%s, %s, %s)", (f1, f2, quadrangle.fragmentationValue))
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
                cur.execute("INSERT INTO quadrangle\
                    (referencemag, secondarymag, diskangle, area,\
                    referencefluxdensity, secondaryfluxdensity, fragmentationvalue)\
                    VALUES\
                    (%s, %s, %s, %s, %s, %s, %s)",
                    (f1, f2, np.float32(quadrangle.diskAngle.v), quadrangle.area.v,
                    quadrangle.fluxDensity, quadrangle.fluxDensity2, quadrangle.fragmentationValue))

            cur.execute("INSERT INTO uniquepairs\
                    VALUES (%s, %s, %s)", (f1, f2, quadrangle.fragmentationValue))
        except Exception as e:
            print("Could not upload completely to database.")
            conn.rollback()
            cur.close()
            return
       
    conn.commit()
    cur.close()


def compare_day(i1, i2, n, files):
    """Compare two magnetograms by fragmentation.

    Args:
        i1 (str): reference instrument
        i2 (str): secondary instrument
        n (int): fragmentation parameter - the level of fragmentation
        files: a tuple of two filepaths to analyze

    Returns:
        list: list of quadrangle objects containing flux density information

    """

    try:
        m1, m2 = fix_longitude(files[0], files[1])
    except ValueError:
        raise
    blocks_n = quad.fragment_multiple(m1, m2, n)

    return blocks_n


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


def sim_compare(fn1, fn2, rotate=0):
    class MockCRD:
        def __init__(self, filename):
            class Im:
                pass
            self.im_raw = Im()
            self.im_raw.data = u.load_sim(filename)
            self.im_raw.dimensions = Pair(*self.im_raw.data.shape*units.pixel)
            self.im_raw.instrument = 'SIM'
            self.im_raw.date = dt.datetime.strptime((filename.split('.')[0].split('\\')[-1] + filename.split('.')[1]), '%Y%m%d%H')
            self.im_corr = mnp.Measurement(self.im_raw.data, self.im_raw.data*0)
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


def analyze_random_sample(i1, i2, tol1, tol2, n=25, passes=1, upload=False):
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

    Returns:
        dict (optional): condensed summary dictionary of block information if not uploaded to database

    """

    file_matches = get_file_list(i1, i2, tol1, tol2)
    conn = u.load_database()
    blocks_list = []
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
                cur.execute("SELECT * FROM uniquepairs\
                                    WHERE referencemag = %s \
                                    AND secondarymag = %s\
                                    AND fragmentationvalue = %s", (file_ids[0], file_ids[1], n))
                if cur.fetchone() is not None:
                    cur.close()
                    continue
                cur.close()
                blocks = compare_day(i1, i2, n, working_files)
                upload_quadrangles(conn, blocks, working_files)
            else:
                blocks_list.append(compare_day(i1, i2, n, working_files))
            del file_matches[choice_int]  # So we don't hit it again
        except ValueError:
            continue
        i += 1
    if not upload:
        return transform_blocks_to_dict(blocks_list, n)


def main():
    """
    Main loop guiding the user though different cross calibration processing options.
    For each magnetogram pair chosen, the list of quadrangles and their parameters will be uploaded to the SQL database.

    --Options--
    r [num]:    will choose a random magnetogram num times out of the instrument overlap
    rsim [num]: will choose a random simulated magnetogram (num) amount of times out of the instrument overlap
    rs [num]:   chooses unique days instead of pairs for random processing
    i:          allows user to switch instruments
    e:          exit the loop
    """
    if i1 is None or i2 is None:
        get_instruments()
    conn = u.load_database()

    while True:
        option = input("Choose a function: (r)andom [num], (s)elect date, switch (i)nstruments, (e)xit: ")
        if option == 'i':
            get_instruments()
        elif 'rsim' in option:
            try:
                passes = int(option.split()[-1])
            except ValueError:
                passes = 1
            n = int(input("Enter segmentation level: "))
            tol1 = input("Enter minimum time: ")
            tol2 = input("Enter maximum time: ")
            params = {'n': n, 't1': tol1, 't2': tol2}
            fileMatches = get_file_list(i1, i2, params)
            for i in range(min(len(fileMatches), passes)):
                try:
                    choiceInt = int(random.random() * len(fileMatches))
                    workingFiles = fileMatches[choiceInt]
                    fileIDs = get_file_id(conn, workingFiles)
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM uniquepairs\
                             WHERE referencemag = %s \
                             AND secondarymag = %s\
                             AND fragmentationvalue = %s", (fileIDs[0], fileIDs[1], n))
                    if cur.fetchone() is not None:
                        cur.close()
                        continue
                    cur.close()
                    sim1, sim2 = sim_compare(workingFiles[0], workingFiles[1])
                    b = quad.fragment_multiple(sim1, sim2, params['n'])
                    del fileMatches[choiceInt]  # So we don't hit it again
                except ValueError:
                    continue
                upload_quadrangles(conn, b, workingFiles, sim=True)

        elif 'rs' in option:
            """Special option for only selecting unique days, not pairs"""
            try:
                passes = int(option.split()[-1])
            except ValueError:
                passes = 1
            n = int(input("Enter segmentation level: "))
            tol1 = input("Enter minimum time: ")
            tol2 = input("Enter maximum time: ")
            params = {'n': n, 't1': tol1, 't2': tol2}
            fileMatches = get_file_list(i1, i2, params)
            dayMatches = []
            i = 0
            while i < passes:
                try:
                    choiceInt = int(random.random()*len(fileMatches))
                    workingFiles = fileMatches[choiceInt]
                    fileIDs = get_file_id(conn, workingFiles)
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM uniquepairs\
                            WHERE referencemag = %s \
                            AND secondarymag = %s\
                            AND fragmentationvalue = %s", (fileIDs[0], fileIDs[1], n))
                    if cur.fetchone() is not None:
                        cur.close()
                        continue
                    cur.execute("SELECT date FROM file WHERE id = %s OR id = %s", (fileIDs[0], fileIDs[1]))
                    if (cur.fetchone()[0].toordinal(), cur.fetchone()[0].toordinal()) in dayMatches:
                        continue
                    dayMatches.append((cur.fetchone()[0].toordinal(), cur.fetchone()[0].toordinal()))
                    cur.close()
                    b = compare_day(i1, i2, params, workingFiles)
                    del fileMatches[choiceInt] #So we don't hit it again
                except ValueError:
                    continue
                upload_quadrangles(conn, b, workingFiles)
                i += 1

        elif 'r' in option:
            print('Random fetch')
            try:
                passes = int(option.split()[-1])
            except ValueError:
                passes = 1
            n = int(input("Enter segmentation level: "))
            tol1 = input("Enter minimum time: ")
            tol2 = input("Enter maximum time: ")
            params = {'n': n, 't1': tol1, 't2': tol2}
            fileMatches = get_file_list(i1, i2, params)
            i = 0
            while True:
                if i > passes:
                    break
                try:
                    choiceInt = int(random.random()*len(fileMatches))
                    workingFiles = fileMatches[choiceInt]
                    fileIDs = get_file_id(conn, workingFiles)
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM uniquepairs\
                            WHERE referencemag = %s \
                            AND secondarymag = %s\
                            AND fragmentationvalue = %s", (fileIDs[0], fileIDs[1], n))
                    if cur.fetchone() is not None:
                        cur.close()
                        continue
                    cur.close()
                    b = compare_day(i1, i2, params, workingFiles)
                    del fileMatches[choiceInt] #So we don't hit it again
                except ValueError:
                    continue
                upload_quadrangles(conn, b, workingFiles)
                i += 1

        elif 'e' in option:
            break
    return


# if __name__ == "__main__":
#     main()
