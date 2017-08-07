import cross_calibration as c
from coord import CRD
import numpy as np
import psycopg2 as psy
import util as z

def main():
    conn = z.load_database()
    cur = conn.cursor()
    cur.execute("SELECT filepath FROM file WHERE instrument=3")
    files = cur.fetchall()
    cur.close()
    badFiles = []

    for f in files:
        try:
            m = CRD(f[0])
            latitudeMask = np.where(m.lath.v < 50)
        except AttributeError:
            continue
        except KeyError:
            continue
        except ValueError as e:
            print(e)
            continue

        if np.nanmax(np.abs(m.lonh.v[latitudeMask])) > 90:
            badFiles.append(f[0])
    return badFiles