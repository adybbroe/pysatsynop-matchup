#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015, 2016 Adam.Dybbroe

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

"""Read and analyse the validation results
"""

from glob import glob
import pandas as pd
import numpy as np
from datetime import datetime

NAMES = ['lon', 'lat', 'station',
         'yyyymmddhhmm',
         'delta_minutes',
         'obs',
         'sat']

DTYPE_SYNOP = [('lon', 'f8'),
               ('lat', 'f8'),
               ('station', '|S5'),
               ('date', object),
               ('delta_t', 'f8'),
               ('sat', 'f8'),
               ('pps_ctype', 'f8'),
               ('obs', 'f8'),
               ('nh', 'i4'),
               ('cl', 'i4'),
               ('cm', 'i4'),
               ('ch', 'i4'),
               ('vvvv', 'i4'),
               ('wv', 'i4'),
               ('sunz', 'f8'),
               ('satz', 'f8'),
               ('ssazd', 'f8'),
               ('ciwv', 'f8'),
               ('tsur', 'f8'),
               ('r06', 'f8'),
               ('r08', 'f8'),
               ('r13', 'f8'),
               ('r16', 'f8'),
               ('r22', 'f8'),
               ('t37', 'f8'),
               ('t85', 'f8'),
               ('t11', 'f8'),
               ('t12', 'f8'),
               ]

DTYPE_POINT = [('lon', 'f8'),
               ('lat', 'f8'),
               ('station', '|S5'),
               ('date', object),
               ('delta_t', 'f8'),
               ('sat', 'f8'),
               ('pps_ctype', 'f8'),
               ('sunz', 'f8'),
               ('satz', 'f8'),
               ('ssazd', 'f8'),
               ('ciwv', 'f8'),
               ('tsur', 'f8'),
               ('r06', 'f8'),
               ('r08', 'f8'),
               ('r13', 'f8'),
               ('r16', 'f8'),
               ('r22', 'f8'),
               ('t37', 'f8'),
               ('t85', 'f8'),
               ('t11', 'f8'),
               ('t12', 'f8'),
               ]


class MatchupAnalysis(object):

    """Reading and analysing the satellite-surface point measurement matchup
    data"""

    def __init__(self, **kwargs):

        if not 'mtype' in kwargs or kwargs['mtype'] == 'synop':
            self.synop = True
            self.datetime_colnum = 3
            self.dtype = DTYPE_SYNOP
            self.point = False
        elif kwargs['mtype'] == 'point':
            self.synop = False
            self.datetime_colnum = 3
            self.point = True
            self.dtype = DTYPE_POINT
        else:
            raise IOError("matchup type has to be either 'synop' or 'point'")

        self.data = None

    def get_results(self, filenames):
        """Read in all the matchup results"""

        items = []
        for dummy, filename in enumerate(filenames):
            try:
                data = np.genfromtxt(filename, skip_header=3,
                                     converters={self.datetime_colnum: lambda x:
                                                 datetime.strptime(x, "%Y%m%d%H%M")},
                                     unpack=True,
                                     dtype=self.dtype)
            except IndexError:
                print "File probably empty..."
                continue

            try:
                pdf = pd.DataFrame(data)
                pdf = pdf.drop_duplicates()
                pdf = pdf.dropna()
            except ValueError:
                print(
                    "Failed on file: %s - Perhaps only one row of data?" % filename)
                names = [names[0] for names in DTYPE_POINT]
                dd = {}
                for var in names:
                    dd[var] = data[var].item()
                pdf = pd.DataFrame(columns=names)
                pdf.loc[len(names)] = dd

            items.append(pdf)

        self.data = pd.concat(items, ignore_index=True)

    def add_sunzenith(self):
        """Derive the sun zenith angles and add to data series"""

        from pyorbital import astronomy

        sunz = []
        for idx in self.data.index:

            sunz.append(astronomy.sun_zenith_angle(self.data.loc[idx, 'date'],
                                                   self.data.loc[idx, 'lon'],
                                                   self.data.loc[idx, 'lat']))

        self.data['sunz'] = pd.Series(sunz, index=self.data.index)


def geo_filter(pdf, outside=True, areaid='euron1'):
    """
    Filter data according to position. All data inside area are ignored
    """
    from pyresample import spherical_geometry, utils

    area = utils.load_area(
        '/local_disk/src/mpop-devel/mpop-smhi/etc/areas.def', areaid)

    # lons = np.array([ pdf['lon'][i] for i in pdf.index ])
    # lats = np.array([ pdf['lat'][i] for i in pdf.index ])

    idx_selected = []
    for idx in pdf.index:
        try:
            loc = spherical_geometry.Coordinate(
                pdf['lon'][idx], pdf['lat'][idx])
        except ValueError:
            import pdb
            pdb.set_trace()

        is_inside = spherical_geometry.point_inside(loc, area.corners)
        if ((outside and not is_inside) or
                (not outside and is_inside)):
            idx_selected.append(idx)

    return pdf.loc[idx_selected, :]


def sunz_filter(pdf, sunz_range):
    """Filter the data according to the sun zenith angle"""

    if not 'sunz' in pdf.keys():
        from pyorbital import astronomy
        idx_selected = []
        for idx in pdf.index:
            sunz = astronomy.sun_zenith_angle(pdf.loc[idx, 'date'],
                                              pdf.loc[idx, 'lon'],
                                              pdf.loc[idx, 'lat'])
            if sunz > sunz_range[0] and sunz < sunz_range[1]:
                idx_selected.append(idx)
        return pdf.loc[idx_selected, :]
    else:
        pdf = pdf[pdf['sunz'] > sunz_range[0]]
        pdf = pdf[pdf['sunz'] < sunz_range[01]]
        return pdf


def station_filter(pdf, station_list):
    """Filter the data according to station number"""

    idx_selected = []
    for idx in pdf.index:

        if pdf.station[idx] in station_list:
            idx_selected.append(idx)

    return pdf.loc[idx_selected, :]


def synop_validation(matchups, filename):
    """Perform the Synop validation of the cloud mask"""

    nodata = matchups['sat'] < 0
    # Filter the data so no data contains observations with partly cloudy
    # (3,4,5 octas)
    clmask_obs = np.logical_and(
        matchups['obs'] > 2. / 8., matchups['obs'] < 6. / 8.)
    clmask_sat = np.logical_and(
        matchups['sat'] > 5. / 16., matchups['sat'] < 10. / 16.)

    mask = np.logical_or(clmask_obs, clmask_sat)
    mask = np.logical_or(mask, nodata)

    obs_f = np.ma.masked_array(matchups['obs'], mask=mask)
    sat_f = np.ma.masked_array(matchups['sat'], mask=mask)

    obs_f_bin = obs_f.compressed() > 5. / 8.
    sat_f_bin = sat_f.compressed() > 5. / 8.

    obs_cloudy_sat_cloudy = (obs_f_bin & sat_f_bin).sum()
    obs_clear_sat_clear = np.logical_and(
        obs_f_bin == False, sat_f_bin == False).sum()
    obs_clear_sat_cloudy = np.logical_and(obs_f_bin == False, sat_f_bin).sum()
    obs_cloudy_sat_clear = np.logical_and(obs_f_bin, sat_f_bin == False).sum()

    print(obs_cloudy_sat_cloudy, obs_clear_sat_clear,
          obs_clear_sat_cloudy, obs_cloudy_sat_clear)
    ntot = (obs_cloudy_sat_cloudy +
            obs_clear_sat_clear +
            obs_clear_sat_cloudy +
            obs_cloudy_sat_clear)

    hr_ = (obs_cloudy_sat_cloudy + obs_clear_sat_clear) / float(ntot)
    pod_cloudy = float(obs_cloudy_sat_cloudy) / \
        float(obs_cloudy_sat_cloudy + obs_cloudy_sat_clear)
    far_cloudy = float(obs_clear_sat_cloudy) / \
        float(obs_clear_sat_cloudy + obs_cloudy_sat_cloudy)
    pod_clear = float(obs_clear_sat_clear) / \
        float(obs_clear_sat_clear + obs_clear_sat_cloudy)
    far_clear = float(obs_cloudy_sat_clear) / \
        float(obs_cloudy_sat_clear + obs_clear_sat_clear)

    bias = (matchups['sat'] - matchups['obs']).sum() / \
        float(matchups['obs'].shape[0])
    rms = np.sqrt(
        ((matchups['sat'] - matchups['obs']) ** 2).sum() / float(matchups['obs'].shape[0]))
    # Mean absolute error:
    ma_ = np.abs(matchups['sat'] - matchups['obs']).sum() / \
        float(matchups['obs'].shape[0])

    print "Hit Rate: ", hr_
    print "POD and FA cloudy: ", pod_cloudy, far_cloudy
    print "POD and FA clear: ", pod_clear, far_clear
    print "MA, RMS and BIAS: " + str(ma_) + " " + str(rms) + " " + str(bias)
    print "N-total = " + str(ntot)

    fd_ = open(filename, 'w')
    lines = []
    lines.append("Hit Rate: " + str(hr_) + '\n')
    lines.append("POD and FA cloudy: " + str(pod_cloudy) +
                 ' ' + str(far_cloudy) + '\n')
    lines.append(
        "POD and FA clear: " + str(pod_clear) + ' ' + str(far_clear) + '\n')
    lines.append(
        "MA, RMS and BIAS: " + str(ma_) + " " + str(rms) + " " + str(bias) + '\n')
    lines.append("N-total = " + str(ntot) + '\n')
    fd_.writelines(lines)
    fd_.close()

    return


def hist1d_plot(data, varname, dataset_name, **kwargs):

    import matplotlib.pyplot as plt

    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = 'green'

    data = data[data['sat'] >= 0]
    plt.hist(data[varname].values, bins=9, color=color)
    if 'ymax' in kwargs:
        #plt.ylim((0, 25000))
        plt.ylim((0, kwargs['ymax']))

    plt.title('Histogram of cloud cover - %s' % dataset_name)
    plt.xlabel("Cloud cover fraction - %s" % dataset_name)
    plt.ylabel('Number of data points')
    plt.savefig('./histogram_%s.png' % dataset_name.replace(' ', '').lower())
    del plt

    return

# --------------------------------------------------------
if __name__ == "__main__":

    #names = glob('./data/matchup_*txt')
    fnames = glob('./data/results_n*txt')

    #this = MatchupAnalysis(mtype='point')
    this = MatchupAnalysis(mtype='synop')
    res = this.get_results(fnames)

    res = res[np.isfinite(res['obs'])]
    #res = geo_filter(res, outside=False)
    #res = geo_filter(res)

    # res = sunz_filter(res, [0, 80])  # Daytime
    res = sunz_filter(res, [90, 180])  # Night
    # res = sunz_filter(res, [80, 90])  # Twilight

    #hist1d_plot(res, 'obs', 'Observations')
    #hist1d_plot(res, 'sat', 'PPS cloud cover', color='blue')

    synop_validation(res, './ppsval_nightime.txt')

    import matplotlib.pyplot as plt

    # Skip the data points where PPS does not provide any cloud cover:
    res = res[res['sat'] >= 0]
    plt.hist2d(res['obs'].values, res['sat'].values, bins=8, cmap=plt.cm.OrRd)
    plt.title('Comparing Synop and PPS Cloud mask')
    plt.xlabel("Cloud cover fraction - Obs")
    plt.ylabel("Cloud cover fraction - PPS")
    cb = plt.colorbar()
    cb.set_label('Num of pixels')
    # plt.savefig('./obs_sat_scatter.png')
    plt.savefig('./obs_sat_scatter_night.png')

    # Make a map plot of the number of occurences and station locations
    station_names = res['station']
    land_stations = np.unique([s for s in station_names if s != 'SHIP'])
    land_stations = land_stations[land_stations != 'nan']
    numobs = np.array([len(res[res['station'] == s]) for s in land_stations])
    lons = np.array([res['lon'][res[res['station'] == s].index[0]]
                     for s in land_stations])
    lats = np.array([res['lat'][res[res['station'] == s].index[0]]
                     for s in land_stations])

    # idx_ships = [ i for i in res.index if res['station'][i] == 'SHIP']
    # lons = np.array([ res['lon'][i] for i in idx_ships])
    # lats = np.array([ res['lat'][i] for i in idx_ships])
    # numobs = np.ones(lons.shape)

    # import pyresample as pr
    # from matplotlib import cm
    # area_id = 'europa'
    # name = 'Europa'
    # proj_id = 'tmp'
    # proj4_args = 'proj=stere, ellps=bessel, lat_0=90, lon_0=14, lat_ts=60'
    # x_size = 700
    # y_size = 700
    # area_extent = (-3700000, -7000000, 3300000, -500000)
    # proj_dict = {'ellps': 'bessel', 'units': 'm', 'lon_0': '14',
    #              'proj': 'stere', 'lat_0': '90'}
    # area_def = pr.geometry.AreaDefinition(area_id, name, proj_id, proj_dict, x_size,
    #                                       y_size, area_extent)
    # swath_def = pr.geometry.SwathDefinition(lons, lats)

    # result = pr.kd_tree.resample_nearest(swath_def, numobs, area_def,
    #                                      radius_of_influence=10000,
    #                                      fill_value=None)
    # pr.plot.save_quicklook('/tmp/map_synops_jan2014.png',
    # area_def, result, label='Number of matchups')

    # bmap = pr.plot.area_def2basemap(area_def)
    # bmng = bmap.drawlsmask(land_color=(0.2,0.6,0.2),
    #                        ocean_color=(0.4,0.4,1.0),
    #                        lakes=True)
    # legend = cm.gist_heat
    # legend = cm.OrRd
    # col = bmap.imshow(result, origin='upper', cmap=legend)
    # cbar = bmap.colorbar()
    # cbar.set_label('Number of matchups')
    # plt.savefig('/tmp/bmap_synops_2013.png.png', bbox_inches='tight')

    # from mpl_toolkits.basemap import Basemap, addcyclic
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(11, 9))
    # ax = plt.subplot(111)
    # m = Basemap(width=10000000, height=7000000,
    #             resolution='l', projection='laea',
    #             lat_ts=60, lat_0=60, lon_0=10.)
    # m.drawcoastlines(linewidth=0.25)
    # xx_ = []
    # yy_ = []
    # intervals = [0, 1, 3, 6, 10, 900]
    # intervals = [0, 1, 10, 100, 200, 1000]
    # colors = [(0.0, 0.0, 0.0),
    #           (0.8, 0.2, 0.8),
    #           (0.0, 0.0, 1.0),
    #           (0.2, 1.0, 0.2),
    #           (1.0, 0.0, 0.0), ]
    # labels = ['1', '2-3', '4-6', '7-10', '>10']
    # labels = ['1', '2-10', '11-99', '100-199', '>200']
    # for i in range(len(intervals) - 1):
    #     mask = np.logical_or(numobs > intervals[i + 1], numobs <= intervals[i])
    #     xlo = np.ma.masked_array(lons, mask=mask).compressed()
    #     yla = np.ma.masked_array(lats, mask=mask).compressed()
    #     x, y = m(xlo, yla)
    #     m.scatter(x, y, 10, marker='o', color=colors[i],
    #               label=labels[i])

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           ncol=3, scatterpoints=1, markerscale=2.0, fancybox=True)
    # plt.title('Locations of Co-located Synops')
    # plt.title('Locations of colocated Surface radiation stations')
    # plt.show()
    # plt.savefig('./worldmap_matchups.png', bbox_inches='tight')
    # plt.savefig('./europamap_matchups.png', bbox_inches='tight')
    # plt.savefig('./worldmap_matchups.png')
    # plt.savefig('./worldmap_matchups_ships.png', bbox_inches='tight')
