#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c20671.ad.smhi.se>

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

"""Plot positions of Øysteins radiation validation stations on a map
"""

from pps_matchup import get_radvaldata

# -------------------------------------------------------------------------
if __name__ == "__main__":

    datapoints = get_radvaldata("radval-stlist.txt")

    lonpos = datapoints['lon']
    latpos = datapoints['lat']

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # m = Basemap(llcrnrlon=-180, llcrnrlat=-80,
    #             urcrnrlon=180, urcrnrlat=80, projection='mill')
    m = Basemap(projection='stere', lon_0=0, lat_0=90., lat_ts=60,
                llcrnrlat=40, urcrnrlat=60,
                llcrnrlon=-30, urcrnrlon=120,
                rsphere=6371200., resolution='l', area_thresh=10000)

    m.drawcoastlines(linewidth=0.5)
    x, y = m(lonpos, latpos)
    m.scatter(x, y, 10, marker='o', color=(0.8, 0.2, 0.2))

    plt.title('Positions of radiation validation stations')
    # plt.show()
    plt.savefig('./radval_onmap.png')
