#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2016 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c14526.ad.smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Read Synop data and plot the locations on a map
"""

import os
from glob import glob
from synop_dwd import get_data
import pandas as pd
from datetime import datetime

SYNOP_DATADIR = "./DataFromDwd/data"


# -------------------------------------------------------------------------
if __name__ == "__main__":

    starttime = datetime(2009, 11, 4, 9, 0)
    #starttime = datetime(2009, 11, 4, 0, 0)
    endtime = datetime(2009, 11, 4, 23, 59)

    synopfiles = glob(os.path.join(SYNOP_DATADIR, "sy_*20091104.qc"))
    print synopfiles

    if len(synopfiles) == 0:
        raise IOError("No synop files found! " +
                      "Synop data dir = %s" % SYNOP_DATADIR)
    print("Synop files considered: " +
          str([os.path.basename(s) for s in synopfiles]))

    items = []
    for filename in synopfiles:
        items.append(get_data(filename))

    synops = pd.concat(items, ignore_index=True)

    lonpos = []
    latpos = []

    # Go through all synops
    for synop in synops.iterrows():
        tup = (synop[1]['lon'], synop[1]['lat'],
               synop[1]['station'], synop[1]['date'].to_datetime(),
               synop[1]['total_cloud_cover'])
        lon, lat, station_name, dtobj, total_cloud_cover = tup

        if dtobj > endtime or dtobj < starttime:
            continue

        if total_cloud_cover >= 9:
            #print("Cloud cover invalid in Synop...")
            continue

        lonpos.append(lon)
        latpos.append(lat)

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    m = Basemap(llcrnrlon=-180, llcrnrlat=-80,
                urcrnrlon=180, urcrnrlat=80, projection='mill')
    m.drawcoastlines(linewidth=0.5)
    x, y = m(lonpos, latpos)
    m.scatter(x, y, 0.1, marker='.', color=(0.1, 0.1, 0.1))

    plt.title('All Synops in database: Time interval = [%s, %s]' % (
        str(starttime), str(endtime)))
    # plt.show()
    plt.savefig('./synops_%s_%s_onmap.png' % (starttime.strftime('%Y%m%d%H%M'),
                                              endtime.strftime('%Y%m%d%H%M')))
