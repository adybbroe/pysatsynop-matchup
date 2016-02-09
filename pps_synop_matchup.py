#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015 Adam.Dybbroe

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

"""Get the PPS cloud mask data for some scenes and do the satellite-synop
co-location. Write data to files. One file for each satellite swath.
"""

import os
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from pypps_reader import NwcSafPpsData
from pyresample import spherical_geometry
from pykdtree.kdtree import KDTree
from synop_dwd import get_data

# The window size over which the satellite cloud cover is derived:
ROW_SZ, COL_SZ = (20, 20)
#ROW_SZ, COL_SZ = (6, 6)
# The time it takes for one sensor scan line:
# AVHRR, scans the earth at a rate of 6 scans per second
# GAC AVHRR, scan rate is 2 GAC scans per second
LINE_TIME = {}
LINE_TIME['avhrr_lac'] = timedelta(seconds=1. / 6.)
LINE_TIME['avhrr_gac'] = timedelta(seconds=1. / 2.)
LINE_TIME['viirs'] = timedelta(seconds=86. / (48. * 16))

time_thr = timedelta(seconds=1800)
#time_thr = timedelta(seconds=3600*6)
time_thr_swath = timedelta(seconds=(1800 + 100 * 60))

#ROOTDIR = "/data/proj/satval/data"
#ROOTDIR = "/nobackup/smhid10/sm_kgkar/ppsdata_testv2014_1g"
#ROOTDIR = "/nobackup/smhid10/sm_shorn/ppsdata_testv2014_1i"
ROOTDIR = "/data/arkiv/proj/safworks/satval_archieve/data/pps_v2014_val"

#SYNOP_DATADIR = "/local_disk/laptop/satellite_synop_matchup/DataFromDwd"
#SYNOP_DATADIR = "./DataFromDwd"
SYNOP_DATADIR = "/data/proj6/saf/adybbroe/satellite_synop_matchup/DataFromDwd"
OUTPUT_DIR = "/local_disk/laptop/satellite_synop_matchup/data"

INSTRUMENT = {'npp': 'viirs', 'noaa18': 'avhrr', 'noaa19': 'avhrr'}
ANGLE_NAMES = ['SUNZ', 'SATZ', 'SSAZD']
SAT_FIELDS = {}
SAT_FIELDS['viirs'] = ['sunz', 'satz', 'ssazd', 'm05',
                       'm07', 'm10', 'm12', 'm14', 'm15', 'm16']
SAT_FIELDS['avhrr'] = ['sunz', 'satz', 'ssazd', '1',
                       '2', '3a', '3b', 'dummy', '4', '5']


def get_satellite_data(imagerdata, sunsat_angles, point):
    """Get the instrument data and viewing angles"""

    retv = {}
    for idx in range(1, 6):
        if hasattr(sunsat_angles, 'image%.1d' % idx):
            obj = getattr(sunsat_angles, 'image%.1d' % idx)
            info = getattr(obj, 'info')
            for angle_name in ANGLE_NAMES:
                if info['product'] == angle_name:
                    dname = angle_name.lower()
                    value = getattr(obj, 'data')[point[0], point[1]]
                    if value != info['nodata'] or value != info['missingdata']:
                        retv[dname] = value * info['gain'] + info['offset']
                    else:
                        retv[dname] = -999

    for idx in range(1, 8):
        if hasattr(imagerdata, 'image%.1d' % idx):
            obj = getattr(imagerdata, 'image%.1d' % idx)
            info = getattr(obj, 'info')
            if info['product'] == 'SATCH':
                dname = info['channel'].lower()
                value = getattr(obj, 'data')[point[0], point[1]]
                if value != info['nodata'] or value != info['missingdata']:
                    retv[dname] = value * info['gain'] + info['offset']
                else:
                    retv[dname] = -999

    return retv


def get_cloud_cover(pps_ctype, point, box):
    """Get the cloud cover around the pixel *point* using a box with dimensions
    *box* """

    try:
        ctypedata = pps_ctype.cloudtype.data
    except AttributeError:
        ctypedata = pps_ctype.ct.data

    shape = ctypedata.shape

    center_val = ctypedata[point[0], point[1]]

    y_left, y_right = point[0] - box[0] / 2, point[0] + box[0] / 2
    x_left, x_right = point[1] - box[1] / 2, point[1] + box[1] / 2
    if (x_left < 0 or y_left < 0 or
            x_right >= shape[1] or y_right >= shape[0]):
        return None, center_val

    ctyarr = ctypedata[y_left: y_right,
                       x_left: x_right]
    ncl = np.sum(np.logical_and(np.greater(ctyarr, 4),
                                np.less(ctyarr, 20)))
    ntot = np.sum(np.logical_and(ctyarr != 0, ctyarr != 20))
    if ntot == 0:
        return None, center_val

    return float(ncl) / float(ntot), center_val


def mask_bounds(boundary, nodata=None):
    """Mask out no-data in the boundary lon,lat coordinates"""

    for idx in range(2):
        mask = np.logical_and(boundary[idx].side1 > -0.999,
                              boundary[idx].side1 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side1 - nodata) < 0.01)
        boundary[idx].side1 = np.ma.masked_array(
            boundary[idx].side1, mask=mask)

        mask = np.logical_and(boundary[idx].side2 > -0.999,
                              boundary[idx].side2 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side2 - nodata) < 0.01)
        boundary[idx].side2 = np.ma.masked_array(
            boundary[idx].side2, mask=mask)

        mask = np.logical_and(boundary[idx].side3 > -0.999,
                              boundary[idx].side3 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side3 - nodata) < 0.01)
        boundary[idx].side3 = np.ma.masked_array(
            boundary[idx].side3, mask=mask)

        mask = np.logical_and(boundary[idx].side4 > -0.999,
                              boundary[idx].side4 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side4 - nodata) < 0.01)
        boundary[idx].side4 = np.ma.masked_array(
            boundary[idx].side4, mask=mask)

    return boundary


def matchup(ctype, avhrr, angles):
    """Do the avhrr/viirs - synop matchup and write data to file.
    """

    if avhrr._how['instrument'] == 'viirs':
        data_type = 'viirs'
    elif avhrr._how['instrument'] == 'avhrr':
        if avhrr._how['orbit_number'] == 99999:
            data_type = 'avhrr_gac'
        else:
            data_type = 'avhrr_lac'
    else:
        raise IOError(
            "Satellite instrument type not supported: " + str(avhrr._how['instrument']))

    obstime = ctype.info['time']
    print("Observation time = " + str(obstime))

    platform = avhrr._how['platform']
    instrument = avhrr._how['instrument']

    resultfile = os.path.join(
        OUTPUT_DIR, './results_%s_%s.txt' % (platform, obstime.strftime('%Y%m%d%H%M')))
    if os.path.exists(resultfile):
        print("File already there! %s ...continue" % resultfile)
        return

    geodata = np.vstack((avhrr.area.lons.ravel(),
                         avhrr.area.lats.ravel())).T
    kd_tree = KDTree(geodata)
    satdata_shape = avhrr.area.lons.shape

    synopfiles = glob(os.path.join(SYNOP_DATADIR, obstime.strftime('%Y%m')) +
                      "/sy_*_%s.qc" % obstime.strftime('%Y%m%d'))
    if obstime.hour >= 23:
        tslot = obstime + timedelta(days=1)
        synopfiles = (synopfiles +
                      glob(os.path.join(SYNOP_DATADIR, tslot.strftime('%Y%m')) +
                           "/sy_*_%s.qc" % tslot.strftime('%Y%m%d')))
    if obstime.hour <= 1:
        tslot = obstime - timedelta(days=1)
        synopfiles = (synopfiles +
                      glob(os.path.join(SYNOP_DATADIR, tslot.strftime('%Y%m')) +
                           "/sy_*_%s.qc" % tslot.strftime('%Y%m%d')))
    # synopfiles = glob(os.path.join(SYNOP_DATADIR, "sy_*_%s.qc" % obstime.strftime('%Y%m%d')))
    # if obstime.hour >= 23:
    #     tslot = obstime + timedelta(days=1)
    #     synopfiles = (synopfiles +
    #                   glob(os.path.join(SYNOP_DATADIR, "sy_*_%s.qc" % tslot.strftime('%Y%m%d'))))
    # if obstime.hour <= 1:
    #     tslot = obstime - timedelta(days=1)
    #     synopfiles = (synopfiles +
    # glob(os.path.join(SYNOP_DATADIR, "sy_*_%s.qc" %
    # tslot.strftime('%Y%m%d'))))

    if len(synopfiles) == 0:
        raise IOError("No synop files found! " +
                      "Synop data dir = %s" % SYNOP_DATADIR)
    print("Synop files considered: " +
          str([os.path.basename(s) for s in synopfiles]))

    items = []
    for filename in synopfiles:
        items.append(get_data(filename))

    synops = pd.concat(items, ignore_index=True)

    try:
        corners = avhrr.area.corners
        clons = np.array([s.lon for s in corners])
        clats = np.array([s.lat for s in corners])
    except ValueError:
        print "Failed getting the corners..."
        corners = None

    if not corners or (0. in np.diff(clons) and 0. in np.diff(clats)):
        print("Something suspecious with the geolocation corners!")
        # Lets set the bounding corners ignoring the first and last few lines
        # of swath!
        bounds = avhrr.area.get_boundary_lonlats()
        # Should mask out nodata values in lon,lat:
        nodata = avhrr.lon.info['nodata'] * avhrr.lon.info['gain']
        # FIXME!

        bounds = mask_bounds(bounds, nodata)
        corners = [spherical_geometry.Coordinate(bounds[0].side4.compressed()[-1],
                                                 bounds[1].side4.compressed()[-1]),
                   spherical_geometry.Coordinate(bounds[0].side2.compressed()[0],
                                                 bounds[1].side2.compressed()[0]),
                   spherical_geometry.Coordinate(bounds[0].side2.compressed()[-1],
                                                 bounds[1].side2.compressed()[-1]),
                   spherical_geometry.Coordinate(bounds[0].side4.compressed()[0],
                                                 bounds[1].side4.compressed()[0])]

        clons = np.array([s.lon for s in corners])
        clats = np.array([s.lat for s in corners])
        if 0. in np.diff(clons) and 0. in np.diff(clats):
            raise ValueError(
                "After adjusting corners, still something fishy...")

    # Select only those observations that are on the northern hemisphere:
    # FIXME!
    # Should be faster then...

    # Select only those observations that fit within time window:
    t1_ = obstime - time_thr_swath
    t2_ = obstime + time_thr_swath
    newsynops = synops[synops['date'] < t2_]
    synops = newsynops[newsynops['date'] > t1_]

    fd_ = open(resultfile, 'w')
    fd_.write('# Platform: ' + str(platform) + '\n')
    fd_.write('# Instrument: ' + str(instrument) + '\n')
    headerline = ('# lon, lat, station id, time, sat-synop time diff, ' +
                  'synop cloud cover, satellite cloud cover, pps cloudtype')
    headerline = headerline + ', nh, cl, cm, ch, vvvv, ww'
    for field in SAT_FIELDS[instrument]:
        headerline = headerline + ', ' + field
    fd_.write(headerline + '\n')

    # Go through all synops (selected above):
    for synop in synops.iterrows():
        tup = (synop[1]['lon'], synop[1]['lat'],
               synop[1]['station'], synop[1]['date'].to_datetime(),
               synop[1]['total_cloud_cover'],
               synop[1]['nh'],
               synop[1]['cl'],
               synop[1]['cm'],
               synop[1]['ch'],
               synop[1]['vvvv'],
               synop[1]['ww'],
               )
        lon, lat, station_name, dtobj, total_cloud_cover, nh_, cl_, cm_, ch_, vvvv, ww_ = tup

        obs_loc = spherical_geometry.Coordinate(lon, lat)

        is_inside = True
        try:
            is_inside = spherical_geometry.point_inside(obs_loc, corners)
        except ZeroDivisionError:
            print("date, station, lon,lat: %r %s (%f,%f)" %
                  (dtobj.strftime('%Y-%m-%d %H:%M'),
                   station_name, lon, lat))

            fd_.close()
            os.remove(resultfile)
            raise

        if not is_inside:
            # print("Outside...")
            continue

        if total_cloud_cover >= 9:
            print("Cloud cover invalid in Synop...")
            continue

        # Find the index of the closest pixel in the satellite data:
        req_point = np.array([[np.rad2deg(obs_loc.lon),
                               np.rad2deg(obs_loc.lat)]])
        dist, kidx = kd_tree.query(req_point, k=1)
        if dist > 0.1:
            print("Point too far away from swath...")
            continue

        row, col = np.unravel_index(kidx[0], satdata_shape)
        # Now that we have the pixel position in swath, we can calculate the
        # actual observation time for that pixel, assuming the observation time
        # at hand applies to the first line:
        pixel_time = obstime + row * LINE_TIME[data_type]
        t1_ = pixel_time - time_thr
        t2_ = pixel_time + time_thr
        if dtobj < t1_ or dtobj > t2_:
            print("Pixel time outside window: " +
                  str(pixel_time) + " " + str(obstime))
            continue

        satdata = get_satellite_data(avhrr, angles, (row, col))

        cloudcover, ctype_center = get_cloud_cover(
            ctype, (row, col), (ROW_SZ, COL_SZ))
        if not cloudcover:
            print "Cloud cover not available from PPS..."
            # continue
            cloudcover = -9

        tdelta = pixel_time - dtobj
        line = (lon, lat, station_name,
                dtobj.strftime('%Y%m%d%H%M'),
                (tdelta.seconds + tdelta.days * 24 * 3600) / 60,
                float(total_cloud_cover) / 8.0, cloudcover,
                ctype_center)
        fd_.write("%6.2f %6.2f %s %s %5.1f %4.2f %5.2f %3d  " % line)
        fd_.write("%4d %4d %4d %4d %6d %5d " %
                  (nh_, cl_, cm_, ch_, vvvv, ww_))

        nosat = True
        for elem in SAT_FIELDS[instrument]:
            if satdata.has_key(elem) and satdata[elem] != -999:
                fd_.write(' %6.2f' % satdata[elem])
                if elem not in [s.lower() for s in ANGLE_NAMES]:
                    nosat = False
            else:
                fd_.write(' -999')
        fd_.write('\n')

        if nosat and cloudcover > -9:
            print("Perhaps something fishy: PPS cloud cover available," +
                  " but no satellite data...")
            print(
                "Coordinates: Row,Col = (" + str(row) + ', ' + str(col) + ')')
            continue
        if not nosat and cloudcover == -9:
            print("Perhaps something fishy: PPS cloud cover not available," +
                  " but satellite data are...")
            print(
                "Coordinates: Row,Col = (" + str(row) + ', ' + str(col) + ')')
            continue

    fd_.close()

    return


def get_scenes(tstart, tend, satellite='npp'):
    """Get all scene files within specified time interval"""

    instr = INSTRUMENT.get(satellite, 'avhrr')
    scene_files = []
    oneday = timedelta(days=1)
    tslot = tstart
    while tslot < tend:
        # Find the cloudtype:
        # matchstr = os.path.join(ROOTDIR,
        #                         "npp_pps_out/%s/%s/npp_%s_????_?????_satproj_*_cloudtype.h5" % (tslot.strftime('%Y%m'), tslot.strftime('%d'), tslot.strftime('%Y%m%d')))
        matchstr = os.path.join(ROOTDIR,
                                "export/S_NWC_CT_%s_?????_%sT*_*Z.h5" % (satellite, tslot.strftime('%Y%m%d')))
        cty_files_aday = glob(matchstr)
        for ctyfile in cty_files_aday:
            fname = os.path.basename(ctyfile).replace('_CT_',
                                                      '_%s_' % instr)
            # dirname = os.path.dirname(ctyfile).replace('npp_pps_out',
            #                                           'npp_pps_int')
            dirname = os.path.dirname(ctyfile).replace('export',
                                                       'import/PPS_data/remapped')
            viirsfile = os.path.join(dirname, fname)

            fname = os.path.basename(ctyfile).replace('_CT_',
                                                      '_sunsatangles_')
            dirname = os.path.dirname(ctyfile).replace('export',
                                                       'import/ANC_data/remapped')
            angles_file = os.path.join(dirname, fname)

            if os.path.exists(viirsfile):
                scene_files.append((ctyfile, viirsfile, angles_file))

        tslot = tslot + oneday

    return scene_files

# -------------------------------------------------------------------------
if __name__ == "__main__":

    starttime = datetime(2012, 1, 1, 0, 0)
    #starttime = datetime(2013, 1, 1, 0, 0)
    endtime = datetime(2013, 1, 1, 0, 0)
    #endtime = datetime(2014, 4, 1, 0, 0)
    scenes = []
    scenes = get_scenes(starttime, endtime, 'npp')
    scenes = scenes + get_scenes(starttime, endtime, 'noaa18')
    scenes = scenes + get_scenes(starttime, endtime, 'noaa19')

    for scene in scenes:
        ctype_obj = NwcSafPpsData(scene[0])
        avhrr_obj = NwcSafPpsData(scene[1])
        angles_obj = NwcSafPpsData(scene[2])
        matchup(ctype_obj, avhrr_obj, angles_obj)
