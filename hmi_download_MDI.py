#!/usr/bin/env python
 
import sys
sys.dont_write_bytecode=True

import datetime
import pandas as pd
from sunpy.net import jsoc
import numpy as np

MDI = pd.read_csv('mdi_for_andres.csv', header = 0).values

for i in np.arange(0, MDI.shape[0], 10):

    print(i)

    date_in = MDI[i, 2]
    date = datetime.datetime.strptime(date_in.replace(':',''), '%Y-%m-%d %H%M%S.%f%z').replace(tzinfo=None, microsecond=0)

    date_s = date - datetime.timedelta(seconds=720/2)
    date_e = date + datetime.timedelta(seconds=720/2)

    print('From {0} to {1}'.format(date_s, date_e))

    client = jsoc.JSOCClient()
    response = client.search(jsoc.attrs.Time(date_s.isoformat(), date_e.isoformat()), jsoc.attrs.Series('hmi.M_720s'),
                             jsoc.attrs.Notify('zachary.werginz@snc.edu'))
    print(response)
    try:
        if response.table.columns['INSTRUME'][0] != 'HMI_SIDE1':
            continue
    except KeyError:
        if len(response.table) == 0:
            continue
    try:
        res = client.fetch(response, path='H:')
        res.wait(progress=True)
    except BaseException:
        continue