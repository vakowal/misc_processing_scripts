"""Geoprocessing workflow: Intersection of ES with KBA.

Calculate the spatial intersection of ecosystem service (ES) provision from
IP-BES results with areas protected as Key Biodiversity Areas (KBA).
"""
import os
import tempfile
import re

from osgeo import gdal
import numpy
import pandas
from datetime import datetime
# import urllib2
# import arcpy

import pygeoprocessing
# from arcpy import sa

# arcpy.CheckOutExtension("Spatial")

_TARGET_NODATA = -9999


def base_data_dict(resolution):
    """Base data input locations on disk."""
    if resolution == '10km':
        data_dict = {
            'kba_raster': 'C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/Global_KBA_10km.tif',  # 'C:/Users/ginge/Dropbox/KBA_ES/Global_KBA_10km.tif',
            'countries_mask': 'C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/onekmgaul_country__world.asc'  # 'C:/Users/ginge/Dropbox/KBA_ES/Mark_Mulligan_data/onekmgaul_country__world/onekmgaul_country__world.asc',
        }
    elif resolution == '300m':
        data_dict = {
            'kba_raster': 'C:/Users/ginge/Dropbox/KBA_ES/Global_KBA_10s.tif',
            'countries_mask': 'C:/Users/ginge/Dropbox/KBA_ES/onekmgaul_country_world_10s.asc',
        }
    elif resolution == '10kmMark':
        data_dict = {
            'kba_raster': 'C:/Users/ginge/Dropbox/KBA_ES/Mark_Mulligan_data/kbas1k/kbas1k.asc',
            'countries_mask': 'C:/Users/ginge/Dropbox/KBA_ES/onekmgaul_country_world_10s.asc',
        }
    return data_dict


def zonal_stats(service_raster_path, zonal_raster_path):
    """Calculate zonal statistics from service raster.

    Calculate sum and average values from the service raster within zones
    specified by integer values in the zonal raster. The two rasters must be
    spatially aligned.

    Parameters:
        service_raster_path (string): path to ecosystem services raster
            giving service provision per pixel
        zonal_raster_path (string): path to raster containing zones within
            which the service raster will be summarized, where each zone is
            identified by a positive integer

    Returns:
        dictionary containing key, value pairs where each key is a unique
            values inside the zonal raster, and each value is a nested
            dictionary containing the average service value within the zone,
            and the sum of service values within each zone
    """
    service_nodata = pygeoprocessing.get_raster_info(
        service_raster_path)['nodata'][0]

    service_raster = gdal.OpenEx(service_raster_path)
    service_band = service_raster.GetRasterBand(1)

    zone_raster = gdal.OpenEx(zonal_raster_path)
    zone_band = zone_raster.GetRasterBand(1)

    try:
        zonal_stat_dict = {}
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (zonal_raster_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                service_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(service_band))
                zone_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(zone_band))
                last_blocksize = blocksize

            service_data = block_offset.copy()
            service_data['buf_obj'] = service_array
            service_band.ReadAsArray(**service_data)

            zone_data = block_offset.copy()
            zone_data['buf_obj'] = zone_array
            zone_band.ReadAsArray(**zone_data)

            zone_values = numpy.unique(zone_array[zone_array > 0])

            for zone in zone_values:
                valid_mask = (
                    (service_array != service_nodata) &
                    (zone_array == zone))
                # the below was necessary for 10km CV raster
                # valid_mask = (
                #     (service_array > 0) &
                #     (zone_array == zone))
                valid_block = service_array[valid_mask]
                zone_sum = numpy.sum(valid_block)
                if zone_sum > 0:
                    zone_avg = zone_sum / valid_block.size
                else:
                    zone_avg = 0
                if zone in zonal_stat_dict:
                    zonal_stat_dict[zone]['sum'] += zone_sum
                    zonal_stat_dict[zone]['average'] += zone_avg
                else:
                    zonal_stat_dict[zone] = {
                        'sum': zone_sum,
                        'average': zone_avg,
                    }
    finally:
        service_band = None
        zone_band = None
        gdal.Dataset.__swig_destroy__(service_raster)
        gdal.Dataset.__swig_destroy__(zone_raster)

    return zonal_stat_dict


def summarize_nested_zonal_stats(
        zonal_stat_dict, nested_zonal_stat_dict, save_as):
    """Calculate proportion of service in each country that is in KBAs."""
    country_dict = {
        'country_id': [key for key in zonal_stat_dict.iterkeys()],
        'total_service_sum': [
            zonal_stat_dict[key]['sum'] for key in
            zonal_stat_dict.iterkeys()],
        'service_sum_in_KBAs': [
            nested_zonal_stat_dict[key]['sum'] for key in
            zonal_stat_dict.iterkeys()],
        'total_area_km2': [
            nested_zonal_stat_dict[key]['total_area_km2'] for key in
            zonal_stat_dict.iterkeys()],
        'kba_area_km2': [
            nested_zonal_stat_dict[key]['kba_area_km2'] for key in
            zonal_stat_dict.iterkeys()],
    }
    summary_df = pandas.DataFrame(data=country_dict)
    summary_df['proportion_service_in_KBAs'] = (
        summary_df.service_sum_in_KBAs / summary_df.total_service_sum)
    summary_df['percent_area_in_KBAs'] = (
        summary_df.kba_area_km2 / summary_df.total_area_km2)
    summary_df['service_relative_to_area'] = (
        summary_df.proportion_service_in_KBAs /
        summary_df.percent_area_in_KBAs)
    summary_df.to_csv(save_as, index=False)


def zonal_stats_to_csv(zonal_stat_dict, zone_identifier, save_as):
    """Save zonal stats as a csv table.

    Convert the zonal stat dictionary to a table with three columns:
    zone_identifier (e.g., "country_id" or "KBA_id"), sum (the sum of
    service values within that zone), and average (the average of service
    values within that zone).

    Parameters:
        zonal_stat_dict (dict): dictionary of key, value pairs where each key
            is a unique value inside the zonal raster, and each value is a
            nested dictionary containing the average service value within the
            zone, and the sum of service values within each zone
        zone_identifier (string): column label for zone identifier in newly
            created csv table
        save_as (string): path to location on disk to save the new csv

    Returns:
        None
    """
    service_by_zone_dict = {
        zone_identifier: [key for key in zonal_stat_dict.iterkeys()],
        'sum': [
            zonal_stat_dict[key]['sum'] for key in
            zonal_stat_dict.iterkeys()],
        'average': [
            zonal_stat_dict[key]['average'] for key in
            zonal_stat_dict.iterkeys()],
    }
    zonal_df = pandas.DataFrame(data=service_by_zone_dict)
    zonal_df.to_csv(save_as, index=False)


def zonal_stat_to_raster(zonal_stat_csv, zone_raster, sum_or_avg, save_as):
    """Display zones in a raster by their sum or average value.

    Reclassify the zone_raster by sum or average service value from the
    zonal_stat_csv.

    Parameters:
        zonal_stat_csv (string): path to csv file containing zonal statistics
            where each row contains a zone id (e.g., country_id or kba_id),
            average service value inside that zone, and the sum of service
            values within that zone
        zone_raster (string): path to raster containing zones as summarized
            in the zonal_stat_csv.
        sum_or_avg (string): if `sum`, display sum of service values per zone;
            if `average`, display average service value per zone
        save_as (string): path to save reclassified raster

    Returns:
        None
    """
    zonal_stat_df = pandas.read_csv(zonal_stat_csv)
    zonal_stat_df.fillna(value=_TARGET_NODATA, inplace=True)
    id_field = [
        c for c in zonal_stat_df.columns.values.tolist() if
        c.endswith('_id')][0]
    zonal_stat_df.set_index(id_field, inplace=True)
    zonal_stat_dict = zonal_stat_df.to_dict(orient='index')
    reclass_dict = {
        key: zonal_stat_dict[key][sum_or_avg]
        for key in zonal_stat_dict.iterkeys()
    }

    target_datatype = gdal.GDT_Float32
    if 0 not in reclass_dict:
        reclass_dict[0] = _TARGET_NODATA
    pygeoprocessing.reclassify_raster(
        (zone_raster, 1), reclass_dict, save_as, target_datatype,
        _TARGET_NODATA)


def nested_zonal_stats(
        service_raster_path, country_raster_path, kba_raster_path,
        area_raster_path):
    """Calculate nested zonal statistics from a service raster.

    Calculate the sum and average service values from the service raster
    falling within KBAs for each country. I.e., the sum and average of
    service values inside KBAs, summarized by country.

    Parameters:
        service_raster_path (string): path to ecosystem services raster
            giving service provision per pixel
        country_raster_path (string): path to countries raster where each
            country is identified by a positive integer
        kba_raster_path (string): path to KBA raster where each KBA is
            identified by a positive integer
        area_raster_path (string): path to raster where the value of each pixel
            is its area in square km

    Returns:
        dictionary containing key, value pairs where each key is a unique
            value inside the countries raster, and each value is a nested
            dictionary containing the average service value within KBAs
            within the country, and the sum of service values within KBAs
            within the country
    """
    service_nodata = pygeoprocessing.get_raster_info(
        service_raster_path)['nodata'][0]
    country_nodata = pygeoprocessing.get_raster_info(
        country_raster_path)['nodata'][0]

    service_raster = gdal.OpenEx(service_raster_path)
    service_band = service_raster.GetRasterBand(1)

    country_raster = gdal.OpenEx(country_raster_path)
    country_band = country_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(kba_raster_path)
    kba_band = kba_raster.GetRasterBand(1)

    area_raster = gdal.OpenEx(area_raster_path)
    area_band = area_raster.GetRasterBand(1)

    try:
        zonal_stat_dict = {}
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (country_raster_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                service_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(service_band))
                country_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(country_band))
                kba_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))
                area_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(area_band))
                last_blocksize = blocksize

            service_data = block_offset.copy()
            service_data['buf_obj'] = service_array
            service_band.ReadAsArray(**service_data)

            country_data = block_offset.copy()
            country_data['buf_obj'] = country_array
            country_band.ReadAsArray(**country_data)

            kba_data = block_offset.copy()
            kba_data['buf_obj'] = kba_array
            kba_band.ReadAsArray(**kba_data)

            area_data = block_offset.copy()
            area_data['buf_obj'] = area_array
            area_band.ReadAsArray(**area_data)

            country_values = numpy.unique(
                country_array[country_array != country_nodata])
            for country_id in country_values:
                # sum of service inside KBAs, inside country
                service_in_kba_country_mask = (
                    (service_array != service_nodata) &
                    (kba_array > 0) &
                    (country_array == country_id))
                service_in_kba_in_country = numpy.sum(
                    service_array[service_in_kba_country_mask])
                # area of pixels inside the country
                country_mask = (country_array == country_id)
                kba_mask = (
                    (kba_array > 0) &
                    (country_array == country_id))
                total_area = numpy.sum(area_array[country_mask])
                kba_area = numpy.sum(area_array[kba_mask])
                if country_id in zonal_stat_dict:
                    zonal_stat_dict[country_id]['sum'] += (
                        service_in_kba_in_country)
                    zonal_stat_dict[country_id]['total_area_km2'] += (
                        total_area)
                    zonal_stat_dict[country_id]['kba_area_km2'] += (
                        kba_area)
                else:
                    zonal_stat_dict[country_id] = {
                        'sum': service_in_kba_in_country,
                        'total_area_km2': total_area,
                        'kba_area_km2': kba_area,
                    }
    finally:
        service_band = None
        country_band = None
        kba_band = None
        gdal.Dataset.__swig_destroy__(service_raster)
        gdal.Dataset.__swig_destroy__(country_raster)
        gdal.Dataset.__swig_destroy__(kba_raster)

    return zonal_stat_dict


def carbon_kba_summary(
        biomass_tile_path, biomass_loss_tile_path, kba_raster_path):
    """Calculate sum of global carbon, and global carbon in KBAs.

    Use a biomass loss tile to mask out areas of forest loss occurring before
    2015.  Calculate the sum of biomass globally and the sum inside KBAs.

    - KBAs are identified as having value == 1
    - valid carbon is aboveground biomass > 0

    Returns:
        dictionary with sum of service inside tile and sum of service inside
            KBAs inside tile
    """
    # check that all inputs are all the same dimensions
    raster_info_list = [
        pygeoprocessing.get_raster_info(path_band)
        for path_band in [
            biomass_tile_path, biomass_loss_tile_path, kba_raster_path]]
    geospatial_info_set = set()
    for raster_info in raster_info_list:
        geospatial_info_set.add(raster_info['raster_size'])
    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(
                geospatial_info_set))

    service_raster = gdal.OpenEx(biomass_tile_path)
    service_band = service_raster.GetRasterBand(1)

    loss_raster = gdal.OpenEx(biomass_loss_tile_path)
    loss_band = loss_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(kba_raster_path)
    kba_band = kba_raster.GetRasterBand(1)

    try:
        summary_dict = {
            'global_service_sum': 0,
            'global_service_sum_in_KBA': 0,
            'n_pixels': 0,
            'n_KBA_pixels': 0,
        }
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (biomass_tile_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                service_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(service_band))
                kba_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))
                loss_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(loss_band))
                last_blocksize = blocksize

            service_data = block_offset.copy()
            service_data['buf_obj'] = service_array
            service_band.ReadAsArray(**service_data)

            kba_data = block_offset.copy()
            kba_data['buf_obj'] = kba_array
            kba_band.ReadAsArray(**kba_data)

            loss_data = block_offset.copy()
            loss_data['buf_obj'] = loss_array
            loss_band.ReadAsArray(**loss_data)

            valid_mask = (
                (service_array > 0) &
                ((loss_array == 0) |
                    (loss_array > 2014)))
            service_divided = service_array / 10000.
            summary_dict['global_service_sum'] += (
                numpy.sum(service_divided[valid_mask]))
            summary_dict['n_pixels'] += len(service_divided[valid_mask])

            kba_mask = (
                valid_mask &
                (kba_array == 1))
            summary_dict['global_service_sum_in_KBA'] += (
                numpy.sum(service_divided[kba_mask]))
            summary_dict['n_KBA_pixels'] += (
                len(service_divided[kba_mask]))

        summary_dict['global_service_percent_in_KBA'] = (
            float(summary_dict['global_service_sum_in_KBA']) /
            summary_dict['global_service_sum'] * 100)

    finally:
        service_band = None
        kba_band = None
        loss_band = None
        gdal.Dataset.__swig_destroy__(service_raster)
        gdal.Dataset.__swig_destroy__(kba_raster)
        gdal.Dataset.__swig_destroy__(loss_raster)

    return summary_dict


def carbon_kba_summary_no_loss(
        biomass_tile_path, kba_raster_path):
    """Calculate sum of global carbon, and global carbon in KBAs.

    Calculate the sum of biomass globally and the sum inside KBAs.

    - KBAs are identified as having value > 0
    - valid carbon is aboveground biomass > 0

    Returns:
        dictionary with sum of service inside tile and sum of service inside
            KBAs inside tile
    """
    # check that all inputs are all the same dimensions
    raster_info_list = [
        pygeoprocessing.get_raster_info(path_band)
        for path_band in [biomass_tile_path, kba_raster_path]]
    geospatial_info_set = set()
    for raster_info in raster_info_list:
        geospatial_info_set.add(raster_info['raster_size'])
    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(
                geospatial_info_set))

    service_raster = gdal.OpenEx(biomass_tile_path)
    service_band = service_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(kba_raster_path)
    kba_band = kba_raster.GetRasterBand(1)

    try:
        summary_dict = {
            'global_service_sum': 0,
            'global_service_sum_in_KBA': 0,
            'n_pixels': 0,
            'n_KBA_pixels': 0,
        }
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (biomass_tile_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                service_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(service_band))
                kba_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))
                last_blocksize = blocksize

            service_data = block_offset.copy()
            service_data['buf_obj'] = service_array
            service_band.ReadAsArray(**service_data)

            kba_data = block_offset.copy()
            kba_data['buf_obj'] = kba_array
            kba_band.ReadAsArray(**kba_data)

            valid_mask = (service_array > 0)
            service_divided = service_array / 10000.
            summary_dict['global_service_sum'] += (
                numpy.sum(service_divided[valid_mask]))
            summary_dict['n_pixels'] += len(service_divided[valid_mask])

            kba_mask = (
                valid_mask &
                (kba_array > 0))
            summary_dict['global_service_sum_in_KBA'] += (
                numpy.sum(service_divided[kba_mask]))
            summary_dict['n_KBA_pixels'] += (
                len(service_divided[kba_mask]))

        summary_dict['global_service_percent_in_KBA'] = (
            float(summary_dict['global_service_sum_in_KBA']) /
            summary_dict['global_service_sum'] * 100)

    finally:
        service_band = None
        kba_band = None
        gdal.Dataset.__swig_destroy__(service_raster)
        gdal.Dataset.__swig_destroy__(kba_raster)

    return summary_dict


def global_kba_summary_country_mask(
        service_raster_path, kba_raster_path, countries_raster_path):
    """Calculate global masked service sum and sum of service in KBAs.

    Calculate global % service inside KBAs from one service raster, counting
    only areas that fall within the countries mask. All input rasters must
    align.

    Parameters:
        service_raster_path (string): path to raster containing service
            values
        kba_raster_path (string): path to raster where Key Biodiversity
            Areas are identified by values > 0
        countries_raster_path (string): path to raster where valid land area
            is identified by values > 0

    Returns:
        dictionary with global sum, global sum in KBA, and global percent
            in KBA
    """
    service_nodata = pygeoprocessing.get_raster_info(
        service_raster_path)['nodata'][0]
    kba_nodata = pygeoprocessing.get_raster_info(
        kba_raster_path)['nodata'][0]
    countries_nodata = pygeoprocessing.get_raster_info(
        countries_raster_path)['nodata'][0]

    service_raster = gdal.OpenEx(service_raster_path)
    service_band = service_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(kba_raster_path)
    kba_band = kba_raster.GetRasterBand(1)

    countries_raster = gdal.OpenEx(countries_raster_path)
    countries_band = countries_raster.GetRasterBand(1)

    try:
        summary_dict = {
            'global_service_sum': 0,
            'global_service_sum_in_KBA': 0,
            'global_num_pixels': 0,
        }
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (service_raster_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                service_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(service_band))
                kba_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))
                countries_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(countries_band))
                last_blocksize = blocksize

            service_data = block_offset.copy()
            service_data['buf_obj'] = service_array
            service_band.ReadAsArray(**service_data)

            kba_data = block_offset.copy()
            kba_data['buf_obj'] = kba_array
            kba_band.ReadAsArray(**kba_data)

            countries_data = block_offset.copy()
            countries_data['buf_obj'] = countries_array
            countries_band.ReadAsArray(**countries_data)

            valid_mask = (
                (service_array != service_nodata) &
                (countries_array != countries_nodata) &
                (countries_array > 0))
            summary_dict['global_service_sum'] += (
                numpy.sum(service_array[valid_mask]))
            summary_dict['global_num_pixels'] += (
                service_array[valid_mask].size)

            kba_mask = (
                valid_mask &
                (kba_array != kba_nodata) &
                (kba_array > 0))
            summary_dict['global_service_sum_in_KBA'] += (
                numpy.sum(service_array[kba_mask]))

        summary_dict['global_service_percent_in_KBA'] = (
            float(summary_dict['global_service_sum_in_KBA']) /
            summary_dict['global_service_sum'] * 100)

    finally:
        service_band = None
        kba_band = None
        gdal.Dataset.__swig_destroy__(service_raster)
        gdal.Dataset.__swig_destroy__(kba_raster)

    return summary_dict


def global_kba_summary(service_raster_path, kba_raster_path):
    """Calculate global service sum and sum of service in KBAs.

    Calculate global % service inside KBAs from one service raster. Service
    raster and KBA raster must align.

    Parameters:
        service_raster_path (string): path to raster containing service
            values
        kba_raster_path (string): path to raster where Key Biodiversity
            Areas are identified by values > 0

    Returns:
        dictionary with global sum, global sum in KBA, and global percent
            in KBA
    """
    service_nodata = pygeoprocessing.get_raster_info(
        service_raster_path)['nodata'][0]
    kba_nodata = pygeoprocessing.get_raster_info(
        kba_raster_path)['nodata'][0]

    service_raster = gdal.OpenEx(service_raster_path)
    service_band = service_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(kba_raster_path)
    kba_band = kba_raster.GetRasterBand(1)

    try:
        summary_dict = {
            'global_service_sum': 0,
            'global_service_sum_in_KBA': 0,
        }
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (service_raster_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                service_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(service_band))
                kba_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))
                last_blocksize = blocksize

            service_data = block_offset.copy()
            service_data['buf_obj'] = service_array
            service_band.ReadAsArray(**service_data)

            kba_data = block_offset.copy()
            kba_data['buf_obj'] = kba_array
            kba_band.ReadAsArray(**kba_data)

            valid_mask = (service_array != service_nodata)
            # the below was necessary for 10km CV raster
            # valid_mask = (service_array > 0)
            summary_dict['global_service_sum'] += (
                numpy.sum(service_array[valid_mask]))

            kba_mask = (
                valid_mask &
                (kba_array != kba_nodata) &
                (kba_array > 0))
            summary_dict['global_service_sum_in_KBA'] += (
                numpy.sum(service_array[kba_mask]))

        summary_dict['global_service_percent_in_KBA'] = (
            float(summary_dict['global_service_sum_in_KBA']) /
            summary_dict['global_service_sum'] * 100)

    finally:
        service_band = None
        kba_band = None
        gdal.Dataset.__swig_destroy__(service_raster)
        gdal.Dataset.__swig_destroy__(kba_raster)

    return summary_dict


def cv_habitat_attribution_workflow(workspace_dir):
    """Use focal statistics to attribute cv service value to marine habitat.

    For each marine habitat type, use a moving window analysis to calculate
    average service value of points falling within a search radius of habitat
    pixels. The search radius for each habitat type corresponds to the effect
    distance for that habitat type used by Rich for the IPBES analysis.

    Parameters:
        workspace_dir (string): path to directory where intermediate rasters
            and results will be stored

    Returns:
        None
    """
    def convert_m_to_deg_equator(distance_meters):
        """Convert meters to degrees at the equator."""
        distance_dict = {
            500: 0.004476516196036,
            1000: 0.008953032392071,
            2000: 0.017906064784142,
        }
        return distance_dict[distance_meters]

    def extract_by_mask(value_raster, mask_raster):
        """Extract values from value_raster under mask_raster.

        Convert value_raster from float to int data type by multiplying the
        values by 1000, to save memory.

        Parameters:
            value_raster: raster containing service values
            mask_raster: raster containing mask by which to select values

        Returns:
            Values from value_raster, multiplied by 1000 and intersecting areas
                of mask_raster
        """
        valid_mask = (
            (value_raster != value_nodata) &
            (mask_raster != mask_nodata))
        result = numpy.empty(value_raster.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = value_raster[valid_mask] * 1000
        integer_result = result.astype(dtype=numpy.int16)
        return integer_result

    def multiply_by_1000(value_raster):
        """Convert float raster to int by multiplying by 1000."""
        valid_mask = (value_raster != _TARGET_NODATA)
        result = numpy.empty(value_raster.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = value_raster[valid_mask] * 1000
        integer_result = result.astype(dtype=numpy.int16)
        return integer_result

    def mosaic_max_value(*raster_list):
        """Mosaic maximum pixels in raster_list.

        Where the rasters in raster_list overlap, keep the maximum value.
        The rasters in raster_list are assumed to contain values >= 0
        only.

        Returns:
            mosaic of rasters including the maximum value in overlapping
                pixels
        """
        raster_copy_list = []
        for r in raster_list:
            r_copy = numpy.copy(r)
            numpy.place(r_copy, numpy.isclose(r_copy, _TARGET_NODATA), [0])
            raster_copy_list.append(r_copy)
        max_mosaic = numpy.amax(raster_copy_list, axis=0)
        invalid_mask = numpy.all(
            numpy.isclose(numpy.array(raster_list), _TARGET_NODATA), axis=0)
        max_mosaic[invalid_mask] = _TARGET_NODATA
        return max_mosaic

    habitat_dict = {
        'mangrove': {
            'risk': 1,
            'effect_distance': 1000,
            'habitat_shp': "F:/cv_workspace/mangrove_proj.shp",
        },
        'saltmarsh': {
            'risk': 2,
            'effect_distance': 1000,
            'habitat_shp': "F:/cv_workspace/saltmarsh_proj.shp",
        },
        'coralreef': {
            'risk': 1,
            'effect_distance': 2000,
            'habitat_shp': "F:/cv_workspace/coral_reef_proj.shp",
        },
        'seagrass': {
            'risk': 4,
            'effect_distance': 500,
            'habitat_shp': "F:/cv_workspace/seagrass_proj.shp",
        },
    }
    # cv_results_shp = r"C:/Users/ginge/Documents/ArcGIS/Default.gdb/main_Project"
    # moving window analysis: mean of service values within effect distance
    # of each pixel
    # did this in ArcMap with the file "kba_es_arcmap_snippets.py"
    # for scenario in ['cur', 'ssp1', 'ssp3', 'ssp5']:
    #     for hab_distance in [2000, 1000, 500]:  # unique effect distance vals
    #         service_field = 'Service_{}'.format(scenario)
    #         effect_distance = convert_m_to_deg_equator(hab_distance)
    #         target_pixel_size = effect_distance / 2
    #         point_stat_raster_path = os.path.join(
    #             workspace_dir,
    #             'pt_st_{}_{}.tif'.format(scenario, hab_distance))
    #         if not os.path.exists(point_stat_raster_path):
    #             print "generating point statistics:"
    #             print point_stat_raster_path
    #             neighborhood = arcpy.sa.NbrCircle(effect_distance, "MAP")
    #             point_statistics_raster = arcpy.sa.PointStatistics(
    #                 cv_results_shp, field=service_field,
    #                 cell_size=target_pixel_size, neighborhood=neighborhood,
    #                 statistics_type="MEAN")
    #             point_statistics_raster.save(point_stat_raster_path)

    for habitat_type in habitat_dict.iterkeys():
        hab_distance = habitat_dict[habitat_type]['effect_distance']
        effect_distance = convert_m_to_deg_equator(hab_distance)
        target_pixel_size = effect_distance / 2
        point_stat_raster_path = os.path.join(
            workspace_dir, 'pt_st_cur_{}.tif'.format(hab_distance))
        # habitat polygons to raster, aligned with point statistics raster
        habitat_raster_path = os.path.join(
            workspace_dir, 'habitat_{}.tif'.format(habitat_type[:3]))
        if not os.path.exists(habitat_raster_path):
            if "grid_code" in arcpy.ListFields(
                    habitat_dict[habitat_type]['habitat_shp']):
                raster_field = "grid_code"
            else:
                raster_field = "FID"
            arcpy.env.cellSize = point_stat_raster_path
            arcpy.env.snapRaster = point_stat_raster_path
            print("polygon to raster ...")
            print(habitat_raster_path)
            arcpy.PolygonToRaster_conversion(
                habitat_dict[habitat_type]['habitat_shp'], raster_field,
                habitat_raster_path, cell_assignment="MAXIMUM_AREA")

    for scenario in ['cur', 'ssp1', 'ssp3', 'ssp5']:
        for habitat_type in habitat_dict.iterkeys():
            hab_distance = habitat_dict[habitat_type]['effect_distance']
            point_stat_raster_path = os.path.join(
                workspace_dir,
                'pt_st_{}_{}.tif'.format(scenario, hab_distance))
            habitat_raster_path = os.path.join(
                workspace_dir, 'habitat_{}.tif'.format(habitat_type[:3]))

            # align habitat raster with point stat raster exactly
            rasters_to_align = [point_stat_raster_path, habitat_raster_path]
            aligned_rasters = [
                os.path.join(workspace_dir, '{}_{}_aligned.tif'.format(
                    os.path.basename(r)[:-4],
                    scenario)) for r in rasters_to_align]
            if not all([os.path.exists(p) for p in aligned_rasters]):
                print("aligning habitat and point statistics ...")
                bounding_box = pygeoprocessing.get_raster_info(
                    point_stat_raster_path)['bounding_box']
                pygeoprocessing.align_and_resize_raster_stack(
                    rasters_to_align, aligned_rasters,
                    ['near'] * len(rasters_to_align),
                    (target_pixel_size, target_pixel_size),
                    bounding_box_mode=bounding_box, raster_align_index=0)

            # mask out habitat pixels from point statistics raster,
            # save memory by multiplying by 1000 and saving as int
            point_stat_habitat_path = os.path.join(
                workspace_dir,
                'pt_st_hab_{}_{}_x1000.tif'.format(
                    scenario, habitat_type[:3]))
            if not os.path.exists(point_stat_habitat_path):
                value_raster = os.path.join(
                    workspace_dir,
                    'pt_st_{}_{}_{}_aligned.tif'.format(
                        scenario, hab_distance, scenario))
                mask_raster = os.path.join(
                    workspace_dir, 'habitat_{}_{}_aligned.tif'.format(
                        habitat_type[:3], scenario))
                value_nodata = pygeoprocessing.get_raster_info(
                    value_raster)['nodata'][0]
                mask_nodata = pygeoprocessing.get_raster_info(
                    mask_raster)['nodata'][0]
                print("extracting point statistics by habitat mask:")
                print(point_stat_habitat_path)
                pygeoprocessing.raster_calculator(
                    [(path, 1) for path in [
                        value_raster, mask_raster]],
                    extract_by_mask, point_stat_habitat_path, gdal.GDT_Int16,
                    _TARGET_NODATA)

    # mosaic attributed habitat rasters to new raster, keeping maximum value
    # in overlapping pixels
    arcpy.env.workspace = workspace_dir
    arcpy.env.cellSize = "MINOF"
    for scenario in ['cur', 'ssp1', 'ssp3', 'ssp5']:
        input_raster_list = [
            'pt_st_hab_{}_{}_x1000.tif'.format(scenario, habitat_type[:3])
            for habitat_type in habitat_dict.keys()]
        input_rasters = ';'.join(input_raster_list)
        output_raster = 'service_mosaic_{}.tif'.format(scenario)
        if not os.path.isfile(os.path.join(workspace_dir, output_raster)):
            # do it in Arc
            print("Mosaic to new raster:")
            print(output_raster)
            arcpy.MosaicToNewRaster_management(
                input_rasters=input_rasters,
                output_location=workspace_dir,
                raster_dataset_name_with_extension=output_raster,
                coordinate_system_for_the_raster="""
                    GEOGCS['GCS_WGS_1984',
                    DATUM['D_WGS_1984', SPHEROID['WGS_1984',
                    6378137.0, 298.257223563]], PRIMEM['Greenwich',0.0],
                    UNIT['Degree', 0.0174532925199433]]""",
                pixel_type="16_BIT_SIGNED", cellsize="",
                number_of_bands="1", mosaic_method="MAXIMUM",
                mosaic_colormap_mode="FIRST")


def service_field_cv_shp():
    """Create service field from text field "Service_cu"."""
    # cv_results_shp = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/CV_outputs_11.28.18/main.CV_outputs"
    cv_results_shp = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/CV_outputs_11.28.18/cv_11_28_18.gdb/main_CV_outputs"
    # arcpy.AddField_management(cv_results_shp, "Service_GK", field_type="FLOAT")
    service_avg = 0
    editing_workspace = os.path.dirname(cv_results_shp)
    with arcpy.da.Editor(editing_workspace) as edit:
        with arcpy.da.UpdateCursor(
                cv_results_shp, ['Service_cur', 'Service_GK']) as cursor:
            for row in cursor:
                service_fl = float(row[0])
                row[1] = service_fl
                cursor.updateRow(row)


def cv_field_check():
    """Check that the service values in the cv raster are correct."""
    cv_results_shp = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/CV_outputs_11.28.18/cv_11_28_18.gdb/main_CV_outputs"
    field_dict = {'Rt_cur': [], 'Rtnohab_cur': [], 'Service_cur': []}
    with arcpy.da.SearchCursor(
            cv_results_shp,
            ['Rt_cur', 'Rtnohab_cur', 'Service_cur']) as cursor:
        for row in cursor:
            field_dict['Rt_cur'].append(row[0])
            field_dict['Rtnohab_cur'].append(row[1])
            field_dict['Service_cur'].append(row[2])
    field_df = pandas.DataFrame(data=field_dict)
    field_df.to_csv("C:/Users/ginge/Desktop/field_df.csv", index=False)


def process_pollination_service_rasters(
        workspace_dir, service_raster_list, resolution='10km'):
    """Process service rasters for summarizing via KBA and country zonal stats.

    Resample service rasters to match KBA and country rasters.

    Parameters:
        workspace_dir (string): path to directory where intermediate and output
            datasets will be written
        service_raster_list (list): list of paths to raw service rasters that
            should be pre-processed prior to summarizing via country and KBA
            zonal stats
        resolution (string): scale or resolution of the analysis

    Returns:
        dictionary of aligned inputs
    """
    data_dict = base_data_dict(resolution)
    aligned_raster_dir = os.path.join(workspace_dir, 'aligned_inputs')
    if not os.path.exists(aligned_raster_dir):
        os.makedirs(aligned_raster_dir)
    aligned_inputs_dict = {
        'kba_raster': os.path.join(
            aligned_raster_dir, os.path.basename(data_dict['kba_raster'])),
        'countries_mask': os.path.join(
            aligned_raster_dir, os.path.basename(data_dict['countries_mask'])),
        'service_raster_list': [
            os.path.join(aligned_raster_dir, os.path.basename(r)) for r in
            service_raster_list],
    }
    existing_processed_inputs = [
        os.path.join(workspace_dir, 'aligned_inputs', os.path.basename(r))
        for r in service_raster_list]
    if all([os.path.exists(p) for p in existing_processed_inputs]):
        return aligned_inputs_dict

    # use block statistics to aggregate service to approximate KBA resolution
    # I did this in R because ArcMap ran out of memory
    blockstat_dir = os.path.join(workspace_dir, 'block_statistic_aggregate')
    kba_pixel_size = pygeoprocessing.get_raster_info(
        data_dict['kba_raster'])['pixel_size']
    service_pixel_size = pygeoprocessing.get_raster_info(
        service_raster_list[0])['pixel_size']

    # align service rasters with KBA raster via near neighbor resampling
    if resolution == "10km" or resolution == "10kmMark":
        input_path_list = (
            [data_dict['kba_raster'], data_dict['countries_mask']] +
            [os.path.join(blockstat_dir, os.path.basename(r)) for r in
                service_raster_list])
    else:
        input_path_list = (
            [data_dict['kba_raster'], data_dict['countries_mask']] +
            service_raster_list)
    aligned_path_list = [
        os.path.join(aligned_raster_dir, os.path.basename(r)) for r in
        input_path_list]
    pygeoprocessing.align_and_resize_raster_stack(
        input_path_list, aligned_path_list,
        ['near'] * len(aligned_path_list), kba_pixel_size,
        bounding_box_mode="union", raster_align_index=0)
    return aligned_inputs_dict


def pollination_workflow(outer_workspace_dir):
    """Frame out workflow to summarize one service raster.

    Parameters:
        outer_workspace_dir (string): path to folder where outputs will be
            created

    Returns:
        none
    """
    raw_data_dir = "F:/IPBES_data_layers_3.24.19"
    service_raster_list = [
        os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if
        f.startswith("prod_poll_dep_realized")]
    global_summary_dict = {
        'service': [],
        'scenario': [],
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'global_service_percent_in_KBA': [],
        }
    # for resolution in ['10kmMark', '10km', '300m']:
    workspace_dir = os.path.join(outer_workspace_dir, 'pollination_summary')
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    aligned_inputs = process_pollination_service_rasters(
        workspace_dir, service_raster_list)
    # rescale to 0 - 1?
    # combine the 3 nutrients?
    result_dir = os.path.join(workspace_dir, 'summary_tables_and_maps')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # spatial summaries
    for service_raster in aligned_inputs['service_raster_list']:
        service = os.path.basename(service_raster)[:25]
        scenario = re.search('10s_(.+?)_md5', service_raster).group(1)
        # global summary of service sum and sum in KBAs
        summary_dict = global_kba_summary_country_mask(
            service_raster, aligned_inputs['kba_raster'],
            aligned_inputs['countries_mask'])
        global_summary_dict['service'].append(service)
        global_summary_dict['scenario'].append(scenario)
        global_summary_dict['global_service_sum'].append(
            summary_dict['global_service_sum'])
        global_summary_dict['global_service_sum_in_KBA'].append(
            summary_dict['global_service_sum_in_KBA'])
        global_summary_dict['global_service_percent_in_KBA'].append(
            summary_dict['global_service_percent_in_KBA'])

        # sum and average of service values within each country
        service_by_country = zonal_stats(
            service_raster, aligned_inputs['countries_mask'])
        service_in_KBA_by_country = nested_zonal_stats(
            service_raster, aligned_inputs['countries_mask'],
            aligned_inputs['kba_raster'],
            "F:/pollination_summary/aligned_inputs/pixel_area_km2.tif")
        country_zonal_stat_csv = os.path.join(
            result_dir, "zonal_stats_by_country_{}_{}.csv".format(
                service, scenario))
        summarize_nested_zonal_stats(
            service_by_country, service_in_KBA_by_country,
            country_zonal_stat_csv)
        # proportion_in_kba_by_country_raster = os.path.join(
        #     result_dir,
        #     "proportion_sum_service_in_kba_by_country_{}_{}.tif".format(
        #         service, scenario))
        # zonal_stat_to_raster(
        #     country_zonal_stat_csv, aligned_inputs['countries_mask'],
        #     'proportion_service_in_KBAs',
        #     proportion_in_kba_by_country_raster)
        service_relative_to_area_by_country_raster = os.path.join(
            result_dir,
            "service_relative_to_area_by_country_{}_{}.tif".format(
                service, scenario))
        zonal_stat_to_raster(
            country_zonal_stat_csv, aligned_inputs['countries_mask'],
            'service_relative_to_area',
            service_relative_to_area_by_country_raster)
    global_summary_df = pandas.DataFrame(data=global_summary_dict)
    global_summary_df.to_csv(os.path.join(
        result_dir, 'global_service_summary.csv'), index=False)


def coastal_vulnerability_workflow(workspace_dir):
    """Summarize coastal vulnerability service in and outside KBAs.

    Parameters:
        workspace_dir (string): path to folder where outputs
            will be created

    Returns:
        None
    """
    service_raster_pattern = "F:/cv_workspace/block_statistic_aggregate/service_mosaic_<scenario>.tif"
    global_summary_dict = {
        'scenario': [],
        'service': [],
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'global_service_percent_in_KBA': [],
        }
    result_dir = os.path.join(workspace_dir, 'summary_tables_and_maps')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for scenario in ['cur', 'ssp1', 'ssp3', 'ssp5']:
        service_raster = service_raster_pattern.replace('<scenario>', scenario)
        aligned_inputs = process_cv_rasters(workspace_dir, service_raster)
        # global summary of service sum and sum in KBAs
        summary_dict = global_kba_summary(
            aligned_inputs['service_raster'], aligned_inputs['kba_raster'])
        global_summary_dict['scenario'].append(scenario)
        global_summary_dict['service'].append('coastal_vulnerability')
        global_summary_dict['global_service_sum'].append(
            summary_dict['global_service_sum'])
        global_summary_dict['global_service_sum_in_KBA'].append(
            summary_dict['global_service_sum_in_KBA'])
        global_summary_dict['global_service_percent_in_KBA'].append(
            summary_dict['global_service_percent_in_KBA'])

        # sum and average of service values within each KBA
        # service_by_KBA = zonal_stats(
        #     aligned_inputs['service_raster'], aligned_inputs['kba_raster'])
        # KBA_zonal_stat_csv = os.path.join(
        #     result_dir, "zonal_stats_by_KBA_{}_{}.csv".format(
        #         os.path.basename(service_raster), resolution))
        # zonal_stats_to_csv(service_by_KBA, 'KBA_id', KBA_zonal_stat_csv)
        # avg_by_KBA_raster = os.path.join(
        #     result_dir, "average_service_by_KBA_{}.tif".format(
        #         os.path.basename(service_raster)))
        # zonal_stat_to_raster(
        #     KBA_zonal_stat_csv, aligned_inputs['kba_raster'], 'average',
        #     avg_by_KBA_raster)
        # sum_by_KBA_raster = os.path.join(
        #     result_dir, "sum_service_by_KBA_{}.tif".format(
        #         os.path.basename(service_raster)))
        # zonal_stat_to_raster(
        #     KBA_zonal_stat_csv, aligned_inputs['kba_raster'], 'sum',
        #     sum_by_KBA_raster)
    global_summary_df = pandas.DataFrame(data=global_summary_dict)
    global_summary_df.to_csv(os.path.join(
        result_dir, 'cv_global_service_summary.csv'), index=False)


def process_cv_rasters(workspace_dir, service_raster, resolution='10km'):
    """Align and resample service raster with KBA raster prior to zonal stats.

    Resample coastal vulnerability service raster to match KBA raster.

    Parameters:
        workspace_dir (string): path to directory where intermediate and output
            datasets will be written
        service_raster (string): path to raw service raster that should be
            pre-processed prior to summarizing via KBA zonal stats
        resolution (string): scale or resolution of the analysis

    Returns:
        dictionary of aligned inputs
    """
    data_dict = base_data_dict(resolution)
    aligned_raster_dir = os.path.join(workspace_dir, 'aligned_inputs')
    if not os.path.exists(aligned_raster_dir):
        os.makedirs(aligned_raster_dir)
    aligned_inputs_dict = {
        'kba_raster': os.path.join(
            aligned_raster_dir, os.path.basename(data_dict['kba_raster'])),
        'service_raster': os.path.join(
            aligned_raster_dir, os.path.basename(service_raster))
    }
    if all([os.path.exists(r) for r in aligned_inputs_dict.iterkeys()]):
        return aligned_inputs_dict

    # use block statistics to aggregate service to approximate KBA resolution
    # I did this in R because ArcMap ran out of memory
    blockstat_dir = os.path.join(workspace_dir, 'block_statistic_aggregate')
    kba_pixel_size = pygeoprocessing.get_raster_info(
        data_dict['kba_raster'])['pixel_size']
    service_pixel_size = pygeoprocessing.get_raster_info(
        service_raster)['pixel_size']

    # align service rasters with KBA raster via near neighbor resampling
    if resolution == "10km" or resolution == "10kmMark":
        input_path_list = [
            data_dict['kba_raster'], os.path.join(
                blockstat_dir, os.path.basename(service_raster))]
    else:
        input_path_list = [
            data_dict['kba_raster'], service_raster]
    aligned_path_list = [
        os.path.join(aligned_raster_dir, os.path.basename(r)) for r in
        input_path_list]
    pygeoprocessing.align_and_resize_raster_stack(
        input_path_list, aligned_path_list,
        ['near'] * len(aligned_path_list), kba_pixel_size,
        bounding_box_mode="union", raster_align_index=0)
    return aligned_inputs_dict


def cv_habitat_in_kbas(outer_workspace_dir):
    """Calculate the total area of marine habitat falling inside KBAs.

    Summarize the area of habitat within KBAs (analogous to total land area
    for pollination service). First merge all habitat polygons together, then
    convert to raster using the same method that was used to generate 10km
    KBA raster; then count total habitat pixels and # habitat pixels in KBAs.

    Returns:
        none
    """
    result_dir = os.path.join(outer_workspace_dir, 'summary_tables_and_maps')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    raw_kba_raster_path = "F:/cv_workspace/aligned_inputs/Global_KBA_10km.tif"
    raw_habitat_raster_path = "F:/cv_workspace/habitat_merge.tif"

    # make sure habitat and kba raster align exactly
    aligned_kba_path = 'F:/cv_workspace/aligned_inputs/Global_KBA_10km_habitat_merge.tif'
    aligned_habitat_path = 'F:/cv_workspace/aligned_inputs/habitat_merge.tif'
    rasters_to_align = [raw_kba_raster_path, raw_habitat_raster_path]
    aligned_rasters = [aligned_kba_path, aligned_habitat_path]
    if not all([os.path.exists(p) for p in aligned_rasters]):
        bounding_box = pygeoprocessing.get_raster_info(
            raw_kba_raster_path)['bounding_box']
        target_pixel_size = pygeoprocessing.get_raster_info(
            raw_kba_raster_path)['pixel_size']
        pygeoprocessing.align_and_resize_raster_stack(
            rasters_to_align, aligned_rasters,
            ['near'] * len(rasters_to_align),
            target_pixel_size,
            bounding_box_mode=bounding_box, raster_align_index=0)

    area_dict = {
        'total_habitat_pixels': [],
        'habitat_in_KBA_pixels': [],
        'KBA_percent_of_habitat_pixels': [],
    }
    habitat_raster = gdal.OpenEx(aligned_habitat_path)
    habitat_band = habitat_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(aligned_kba_path)
    kba_band = kba_raster.GetRasterBand(1)

    try:
        total_habitat_pixels = 0
        total_kba_pixels = 0
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (aligned_habitat_path, 1), offset_only=True):
            blocksize = (
                block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                habitat_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(
                        habitat_band))
                kba_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))

            habitat_data = block_offset.copy()
            habitat_data['buf_obj'] = habitat_array
            habitat_band.ReadAsArray(**habitat_data)

            kba_data = block_offset.copy()
            kba_data['buf_obj'] = kba_array
            kba_band.ReadAsArray(**kba_data)
            habitat_mask = (habitat_array > 0)
            kba_mask = (
                (habitat_array > 0) &
                (kba_array > 0))
            block_habitat_pixels = habitat_array[habitat_mask].size
            block_kba_pixels = habitat_array[kba_mask].size
            total_habitat_pixels += block_habitat_pixels
            total_kba_pixels += block_kba_pixels
        area_dict['total_habitat_pixels'].append(total_habitat_pixels)
        area_dict['habitat_in_KBA_pixels'].append(total_kba_pixels)
        area_dict['KBA_percent_of_habitat_pixels'].append(
            (float(total_kba_pixels) / total_habitat_pixels) * 100)
    finally:
        habitat_band = None
        kba_band = None
        gdal.Dataset.__swig_destroy__(habitat_raster)
        gdal.Dataset.__swig_destroy__(kba_raster)

    area_summary_df = pandas.DataFrame(data=area_dict)
    area_summary_df.to_csv(os.path.join(
        result_dir, 'KBA_area_summary.csv'), index=False)


def area_of_kbas(outer_workspace_dir):
    """Calculate the area of KBAs relative to land mask.

    For three spatial resolutions, calculate the percent of pixels covered by
    KBAs within the countries mask used to aggregate terrestrial results.

    Parameters:
        outer_workspace_dir (string): directory where summary csv will be
            stored

    Returns:
        None
    """
    area_dict = {
        'spatial_resolution': [],
        'total_land_area_km2': [],
        'KBA_area_km2': [],
        'KBA_percent_of_area': [],
    }
    service_raster_list = []
    # "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_en_10s_ESACCI_LC_L4_LCSS.tif"]
    for resolution in ['10km']:  # ['10kmMark', '10km', '300m']:
        # workspace_dir = os.path.join(outer_workspace_dir, resolution)
        # if not os.path.exists(workspace_dir):
        #     os.makedirs(workspace_dir)
        # aligned_inputs = process_pollination_service_rasters(
        #     workspace_dir, service_raster_list, resolution)
        aligned_inputs = {
            'kba_raster': "F:/carbon_lpj_guess_workspace/aligned_inputs/Global_KBA_10km.tif",
            'countries_mask': "F:/carbon_lpj_guess_workspace/aligned_inputs/onekmgaul_country__world.asc",
            'area_raster': "F:/carbon_lpj_guess_workspace/aligned_inputs/pixel_area_km2.tif",
            'service_raster_list': [],
        }

        countries_raster_path = aligned_inputs['countries_mask']
        kba_raster_path = aligned_inputs['kba_raster']
        area_raster_path = aligned_inputs['area_raster']

        countries_raster = gdal.OpenEx(countries_raster_path)
        countries_band = countries_raster.GetRasterBand(1)

        kba_raster = gdal.OpenEx(kba_raster_path)
        kba_band = kba_raster.GetRasterBand(1)

        area_raster = gdal.OpenEx(area_raster_path)
        area_band = area_raster.GetRasterBand(1)

        try:
            total_land_area = 0
            total_kba_area = 0
            last_blocksize = None
            for block_offset in pygeoprocessing.iterblocks(
                    (countries_raster_path, 1), offset_only=True):
                blocksize = (
                    block_offset['win_ysize'], block_offset['win_xsize'])

                if last_blocksize != blocksize:
                    countries_array = numpy.zeros(
                        blocksize,
                        dtype=pygeoprocessing._gdal_to_numpy_type(
                            countries_band))
                    kba_array = numpy.zeros(
                        blocksize,
                        dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))
                    area_array = numpy.zeros(
                        blocksize,
                        dtype=pygeoprocessing._gdal_to_numpy_type(area_band))

                countries_data = block_offset.copy()
                countries_data['buf_obj'] = countries_array
                countries_band.ReadAsArray(**countries_data)

                kba_data = block_offset.copy()
                kba_data['buf_obj'] = kba_array
                kba_band.ReadAsArray(**kba_data)

                area_data = block_offset.copy()
                area_data['buf_obj'] = area_array
                area_band.ReadAsArray(**area_data)

                country_mask = (countries_array > 0)
                kba_mask = (
                    (countries_array > 0) &
                    (kba_array > 0))
                block_land_area = numpy.sum(area_array[country_mask])
                block_kba_area = numpy.sum(area_array[kba_mask])

                total_land_area += block_land_area
                total_kba_area += block_kba_area
            area_dict['spatial_resolution'].append(resolution)
            area_dict['total_land_area_km2'].append(total_land_area)
            area_dict['KBA_area_km2'].append(total_kba_area)
            area_dict['KBA_percent_of_area'].append(
                (float(total_kba_area) / total_land_area) * 100)
        finally:
            countries_band = None
            kba_band = None
            gdal.Dataset.__swig_destroy__(countries_raster)
            gdal.Dataset.__swig_destroy__(kba_raster)

    area_summary_df = pandas.DataFrame(data=area_dict)
    area_summary_df.to_csv(os.path.join(
        outer_workspace_dir, 'KBA_area_summary.csv'), index=False)


def carbon_workflow():
    """Summarize carbon in KBAs and total global carbon.

    Carbon data are stored in tiled rasters. Instead of mosaicking the tiles
    together into one global raster, process each tile separately.

    Parameters:

    Returns:
        None
    """
    # directory containing tiles: aboveground live woody biomass in 2000
    alwb_dir = "F:/GFW_ALWBD_2000"
    hansen_loss_dir = "F:/Hansen_lossyear"
    hansen_base_name = 'Hansen_GFC-2017-v1.5_lossyear_<loc_string>.tif'
    kba_shp = "F:/Global_KBA_poly.shp"

    # aligned rasters go in the temp_dir, but intermediate files go in
    # persistent directory in case there is an interruption
    intermediate_save_dir = "F:/carbon_intermediate_files"
    if not os.path.exists(intermediate_save_dir):
        os.makedirs(intermediate_save_dir)
    temp_dir = tempfile.mkdtemp()
    tile_list = [
        os.path.join(alwb_dir, f) for f in
        os.listdir(alwb_dir) if f.endswith('.tif')]

    summary_dict = {
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'resolution': [],
        'n_pixels': [],
        'n_KBA_pixels': [],
        'biomass_tile': [],
    }
    # native resolution
    combine_summary_results()
    done_df_csv = [
        f for f in os.listdir(intermediate_save_dir) if
        f.startswith('biomass_KBA_summary')]
    done_df = pandas.read_csv(
        os.path.join(intermediate_save_dir, done_df_csv[0]))
    done_df = done_df[done_df.n_pixels.notnull()]
    done_tiles = done_df[
        done_df['resolution'] == '30m']['biomass_tile'].values
    current_object = 1
    for biomass_tile_path in tile_list:
        print("processing tile {} of 280".format(current_object))
        if os.path.basename(biomass_tile_path) in done_tiles:
            current_object += 1
            continue
        loc_string = os.path.basename(biomass_tile_path)[:8]
        loss_path = os.path.join(
            hansen_loss_dir,
            hansen_base_name.replace('<loc_string>', loc_string))
        kba_raster_path = os.path.join(temp_dir, 'kba.tif')
        pygeoprocessing.new_raster_from_base(
            biomass_tile_path, kba_raster_path, gdal.GDT_Int16,
            [_TARGET_NODATA], fill_value_list=[0])
        # create aligned KBA raster
        print("rasterizing aligned KBA raster")
        pygeoprocessing.rasterize(kba_shp, kba_raster_path, burn_values=[1])
        # calculate zonal stats for tile
        print("calculating zonal stats for tile")
        tile_dict = carbon_kba_summary(
            biomass_tile_path, loss_path, kba_raster_path)
        summary_dict['global_service_sum'].append(
            tile_dict['global_service_sum'])
        summary_dict['global_service_sum_in_KBA'].append(
            tile_dict['global_service_sum_in_KBA'])
        summary_dict['n_pixels'].append(tile_dict['n_pixels'])
        summary_dict['n_KBA_pixels'].append(tile_dict['n_KBA_pixels'])
        summary_dict['resolution'].append('30m')
        summary_dict['biomass_tile'].append(
            os.path.basename(biomass_tile_path))
        current_object += 1
        summary_df = pandas.DataFrame(data=summary_dict)
        csv_path = os.path.join(
            intermediate_save_dir, 'summary_dict_{}.csv'.format(
                current_object))
        summary_df.to_csv(csv_path, index=False)

    # 10 km resolution
    summary_dict = {
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'resolution': [],
        'n_pixels': [],
        'n_KBA_pixels': [],
        'biomass_tile': [],
    }
    kba_10km_path = "C:/Users/ginge/Dropbox/KBA_ES/Global_KBA_10km.tif"
    target_pixel_size = pygeoprocessing.get_raster_info(
        kba_10km_path)['pixel_size']
    # biomass masked and aggregated to ~10 km in R
    biomass_resampled_dir = "C:/Users/ginge/Desktop/biomass_working_dir/GFW_ALWBD_2015_10km_resample"
    aligned_dir = os.path.join(temp_dir, 'aligned')
    os.makedirs(aligned_dir)

    tile_basename_list = [
        f for f in os.listdir(biomass_resampled_dir) if f.endswith('.tif')]
    current_object = 1
    done_tiles = done_df[
        done_df['resolution'] == '10km']['biomass_tile'].values
    for biomass_tile_bn in tile_basename_list:
        if biomass_tile_bn in done_tiles:
            continue
        native_path = os.path.join(alwb_dir, biomass_tile_bn)
        resampled_path = os.path.join(biomass_resampled_dir, biomass_tile_bn)
        input_path_list = [native_path, resampled_path, kba_10km_path]

        aligned_native_path = os.path.join(aligned_dir, 'native_biomass.tif')
        aligned_resampled_path = os.path.join(aligned_dir, biomass_tile_bn)
        aligned_kba_path = os.path.join(aligned_dir, 'kba.tif')
        target_raster_path_list = [
            aligned_native_path, aligned_resampled_path, aligned_kba_path]

        print("processing 10km tile {} of {}".format(
            current_object, len(tile_basename_list)))
        align_bounding_box = pygeoprocessing.get_raster_info(
            native_path)['bounding_box']
        pygeoprocessing.align_and_resize_raster_stack(
            input_path_list, target_raster_path_list,
            ['near'] * len(input_path_list), target_pixel_size,
            bounding_box_mode=align_bounding_box)

        tile_dict = carbon_kba_summary_no_loss(
            aligned_resampled_path, aligned_kba_path)
        summary_dict['global_service_sum'].append(
            tile_dict['global_service_sum'])
        summary_dict['global_service_sum_in_KBA'].append(
            tile_dict['global_service_sum_in_KBA'])
        summary_dict['n_pixels'].append(tile_dict['n_pixels'])
        summary_dict['n_KBA_pixels'].append(tile_dict['n_KBA_pixels'])
        summary_dict['resolution'].append('10km')
        summary_dict['biomass_tile'].append(biomass_tile_bn)
        current_object += 1
        # if current_object % 10 == 0:
        summary_df = pandas.DataFrame(data=summary_dict)
        csv_path = os.path.join(
            intermediate_save_dir, 'summary_dict_{}.csv'.format(
                current_object))
        summary_df.to_csv(csv_path, index=False)

    combine_summary_results()


def combine_summary_results():
    outer_outdir = "F:/carbon_intermediate_files"
    existing_csv = [
        os.path.join(outer_outdir, f) for f in os.listdir(outer_outdir) if
        f.startswith('summary_dict_') or f.startswith('biomass_KBA_summary_')]
    existing_df = [pandas.read_csv(csv) for csv in existing_csv]
    for df in existing_df:
        if 'n_pixels' not in df.columns.values:
            df['n_pixels'] = 'NA'
        if 'n_KBA_pixels' not in df.columns.values:
            df['n_KBA_pixels'] = 'NA'
    if len(existing_df) > 1:
        done_df = pandas.concat(existing_df)
        done_df.drop_duplicates(inplace=True)
        now_str = datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
        save_as = os.path.join(
            outer_outdir,
            'biomass_KBA_summary_{}.csv'.format(now_str))
        done_df.to_csv(save_as, index=False)
        for item in existing_csv:
            os.remove(item)


def process_ndr_service_rasters(workspace_dir):
    """Calculate N retention and align with KBA and countries rasters.

    Resample ~10km N retention rasters to align with KBA and countries rasters.

    Parameters:
        workspace_dir (string): path to directory where intermediate and
            output datasets will be written

    Returns:
        dictionary of aligned inputs
    """
    def calc_N_retention(load, export):
        """Calculate N retention from load and export."""
        valid_mask = (
            (load != load_nodata) &
            (export != export_nodata))
        retention = numpy.empty(load.shape, dtype=numpy.float32)
        retention[:] = load_nodata
        retention[valid_mask] = load[valid_mask] - export[valid_mask]
        return retention

    data_dict = base_data_dict('10km')
    aligned_raster_dir = os.path.join(workspace_dir, 'aligned_inputs')
    if not os.path.exists(aligned_raster_dir):
        os.makedirs(aligned_raster_dir)

    # paths to raw load and export rasters (resampled with block statistics)
    scenario_raster_dict = {
        'cur': {
            'load': 'worldclim_2015_modified_load_compressed_md5_e3072705a87b0db90e7620abbc0d75f1.tif',
            'export': 'worldclim_2015_n_export_compressed_md5_fa15687cc4d4fdc5e7a6351200873578.tif',
        },
        'ssp1': {
            'load': 'worldclim_2050_ssp1_modified_load_compressed_md5_a5f1db75882a207636546af94cde6549.tif',
            'export': 'worldclim_2050_ssp1_n_export_compressed_md5_4b2b0a4ac6575fde5aca00de4f788494.tif',
        },
        'ssp3': {
            'load': 'worldclim_2050_ssp3_modified_load_compressed_md5_e49e578ed025c0bc796e55b7f27f82f1.tif',
            'export': 'worldclim_2050_ssp3_n_export_compressed_md5_b5259ac0326b0dcef8a34f2086e8339b.tif',
        },
        'ssp5': {
            'load': 'worldclim_2050_ssp5_modified_load_compressed_md5_7337576433238f70140be9ec5b588fd1.tif',
            'export': 'worldclim_2050_ssp5_n_export_compressed_md5_12b9caecc29058d39748e13bf5b5f150.tif',
        }
    }
    retention_raster_list = [
        os.path.join(workspace_dir, 'N_retention_{}.tif'.format(scenario))
        for scenario in scenario_raster_dict.keys()]
    aligned_inputs_dict = {
        'kba_raster': os.path.join(
            aligned_raster_dir, os.path.basename(data_dict['kba_raster'])),
        'countries_mask': os.path.join(
            aligned_raster_dir, os.path.basename(data_dict['countries_mask'])),
        'service_raster_list': [
            os.path.join(aligned_raster_dir, os.path.basename(r)) for r in
            retention_raster_list],
    }
    existing_processed_inputs = [
        os.path.join(workspace_dir, 'aligned_inputs', os.path.basename(r))
        for r in retention_raster_list]
    if all([os.path.exists(p) for p in existing_processed_inputs]):
        return aligned_inputs_dict

    # use block statistics to aggregate service to approximate KBA resolution
    # I did this in R because ArcMap ran out of memory
    blockstat_dir = os.path.join(workspace_dir, 'block_statistic_aggregate')

    # calculate N retention for each scenario
    for scenario in scenario_raster_dict.keys():
        load_raster_path = os.path.join(
            blockstat_dir, scenario_raster_dict[scenario]['load'])
        export_raster_path = os.path.join(
            blockstat_dir, scenario_raster_dict[scenario]['export'])
        retention_raster_path = os.path.join(
            workspace_dir, 'N_retention_{}.tif'.format(scenario))
        load_nodata = pygeoprocessing.get_raster_info(
            load_raster_path)['nodata'][0]
        export_nodata = pygeoprocessing.get_raster_info(
            export_raster_path)['nodata'][0]
        print("Calculating N retention: ")
        print(retention_raster_path)
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in [
                load_raster_path, export_raster_path]],
            calc_N_retention, retention_raster_path, gdal.GDT_Float32,
            load_nodata)

    kba_pixel_size = pygeoprocessing.get_raster_info(
        data_dict['kba_raster'])['pixel_size']
    service_pixel_size = pygeoprocessing.get_raster_info(
        retention_raster_list[0])['pixel_size']

    input_path_list = (
        [data_dict['kba_raster'], data_dict['countries_mask']] +
        retention_raster_list)
    aligned_path_list = [
        os.path.join(aligned_raster_dir, os.path.basename(r)) for r in
        input_path_list]
    print("Aligning retention rasters: ")
    pygeoprocessing.align_and_resize_raster_stack(
        input_path_list, aligned_path_list,
        ['near'] * len(aligned_path_list), kba_pixel_size,
        bounding_box_mode="union", raster_align_index=0)
    return aligned_inputs_dict


def ndr_workflow(workspace_dir):
    """Summarize N retention inside KBAs for current conditions and scenarios.

    Parameters:
        workspace_dir (string): path to folder where outputs will be
            created

    Returns:
        none
    """
    global_summary_dict = {
        'service': [],
        'scenario': [],
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'global_service_percent_in_KBA': [],
        }
    aligned_inputs = process_ndr_service_rasters(workspace_dir)
    result_dir = os.path.join(workspace_dir, 'summary_tables_and_maps')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for service_raster in aligned_inputs['service_raster_list']:
        service = 'N retention'
        scenario = re.search('N_retention_(.+?).tif', service_raster).group(1)
        # global summary of service sum and sum in KBAs
        summary_dict = global_kba_summary_country_mask(
            service_raster, aligned_inputs['kba_raster'],
            aligned_inputs['countries_mask'])
        global_summary_dict['service'].append(service)
        global_summary_dict['scenario'].append(scenario)
        global_summary_dict['global_service_sum'].append(
            summary_dict['global_service_sum'])
        global_summary_dict['global_service_sum_in_KBA'].append(
            summary_dict['global_service_sum_in_KBA'])
        global_summary_dict['global_service_percent_in_KBA'].append(
            summary_dict['global_service_percent_in_KBA'])

        # proportion of service in KBAs by country
        service_by_country = zonal_stats(
            service_raster, aligned_inputs['countries_mask'])
        service_in_KBA_by_country = nested_zonal_stats(
            service_raster, aligned_inputs['countries_mask'],
            aligned_inputs['kba_raster'],
            "F:/ndr_workspace/aligned_inputs/pixel_area_km2.tif")
        country_zonal_stat_csv = os.path.join(
            result_dir, "zonal_stats_by_country_{}_{}.csv".format(
                service, scenario))
        summarize_nested_zonal_stats(
            service_by_country, service_in_KBA_by_country,
            country_zonal_stat_csv)
        proportion_in_kba_by_country_raster = os.path.join(
            result_dir,
            "proportion_sum_service_in_kba_by_country_{}_{}.tif".format(
                service, scenario))
        # proportion of service in KBAs, by country
        zonal_stat_to_raster(
            country_zonal_stat_csv, aligned_inputs['countries_mask'],
            'proportion_service_in_KBAs',
            proportion_in_kba_by_country_raster)
        # service relative to area, by country
        service_relative_to_area_by_country_raster = os.path.join(
            result_dir,
            "service_relative_to_area_by_country_{}_{}.tif".format(
                service, scenario))
        zonal_stat_to_raster(
            country_zonal_stat_csv, aligned_inputs['countries_mask'],
            'service_relative_to_area',
            service_relative_to_area_by_country_raster)
    global_summary_df = pandas.DataFrame(data=global_summary_dict)
    global_summary_df.to_csv(os.path.join(
        result_dir, 'global_service_summary.csv'), index=False)


def process_LPJ_GUESS_rasters(workspace_dir):
    """Align and resample LPJ-GUESS rasters with KBA and countries rasters.

    Raw outputs containing carbon in vegetation from LPJ-GUESS are at 1 degree
    resolution.  Resample these to align exactly with the 10 km KBA and
    countries rasters.

    Parameters:
        workspace_dir (string): path to directory where intermediate and
            output datasets will be written

    Returns:
        dictionary of aligned inputs
    """
    data_dict = base_data_dict('10km')
    aligned_raster_dir = os.path.join(workspace_dir, 'resampled_kg_m-2')
    if not os.path.exists(aligned_raster_dir):
        os.makedirs(aligned_raster_dir)

    raw_tif_dir = r'F:/LPJ_Guess_carbon_scenarios/exported_geotiff'
    LPJ_GUESS_basename_list = [
        'LPJ-GUESS_rcp2p6_IMAGE_cVeg_2015_1x1.tif',
        'LPJ-GUESS_rcp2p6_IMAGE_cVeg_2050_1x1.tif',
        'LPJ-GUESS_rcp6p0_AIM_cVeg_2015_1x1.tif',
        'LPJ-GUESS_rcp6p0_AIM_cVeg_2050_1x1.tif',
        'LPJ-GUESS_rcp8p5_MAGPIE_cVeg_2015_1x1.tif',
        'LPJ-GUESS_rcp8p5_MAGPIE_cVeg_2050_1x1.tif']
    LPJ_GUESS_path_list = [
        os.path.join(raw_tif_dir, b) for b in LPJ_GUESS_basename_list]
    results_dir = os.path.join(workspace_dir, 'aligned_inputs')
    aligned_inputs_dict = {
        'kba_raster': os.path.join(
            results_dir, os.path.basename(data_dict['kba_raster'])),
        'countries_mask': os.path.join(
            results_dir, os.path.basename(data_dict['countries_mask'])),
        'service_raster_list': [
            os.path.join(results_dir, os.path.basename(r)) for r in
            LPJ_GUESS_path_list],
    }
    existing_processed_inputs = [
        os.path.join(results_dir, os.path.basename(r))
        for r in LPJ_GUESS_path_list]
    if all([os.path.exists(p) for p in existing_processed_inputs]):
        return aligned_inputs_dict

    kba_pixel_size = pygeoprocessing.get_raster_info(
        data_dict['kba_raster'])['pixel_size']
    service_pixel_size = pygeoprocessing.get_raster_info(
        LPJ_GUESS_path_list[0])['pixel_size']
    input_path_list = (
        [data_dict['kba_raster'], data_dict['countries_mask']] +
        LPJ_GUESS_path_list)
    aligned_path_list = [
        os.path.join(aligned_raster_dir, os.path.basename(r)) for r in
        input_path_list]
    print("Aligning LPJ-GUESS carbon rasters: ")
    pygeoprocessing.align_and_resize_raster_stack(
        input_path_list, aligned_path_list,
        ['near'] * len(aligned_path_list), kba_pixel_size,
        bounding_box_mode="union", raster_align_index=0)
    raise ValueError("Carbon rasters must be converted to Gt in R")


def LPJ_carbon_workflow(workspace_dir):
    """Summarize carbon in KBAs from LPJ-GUESS scenario analysis.

    Parameters:
        workspace_dir (string): path to folder where outputs will be
            created

    Returns:
        none
    """
    global_summary_dict = {
        'source': [],
        'year': [],
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'global_service_percent_in_KBA': [],
        'global_num_pixels': [],
        }
    aligned_inputs = process_LPJ_GUESS_rasters(workspace_dir)
    result_dir = os.path.join(workspace_dir, 'summary_tables_and_maps')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for service_raster in aligned_inputs['service_raster_list']:
        source = re.search('LPJ-GUESS_(.+?)_cVeg', service_raster).group(1)
        year = re.search('cVeg_(.+?)_1x1', service_raster).group(1)
        # global summary of service sum and sum in KBAs
        summary_dict = global_kba_summary_country_mask(
            service_raster, aligned_inputs['kba_raster'],
            aligned_inputs['countries_mask'])
        global_summary_dict['source'].append(source)
        global_summary_dict['year'].append(year)
        global_summary_dict['global_service_sum'].append(
            summary_dict['global_service_sum'])
        global_summary_dict['global_service_sum_in_KBA'].append(
            summary_dict['global_service_sum_in_KBA'])
        global_summary_dict['global_service_percent_in_KBA'].append(
            summary_dict['global_service_percent_in_KBA'])
        global_summary_dict['global_num_pixels'].append(
            summary_dict['global_num_pixels'])
    global_summary_df = pandas.DataFrame(data=global_summary_dict)
    global_summary_df.to_csv(os.path.join(
        result_dir, 'global_service_summary.csv'), index=False)

    # sum and average of service values within each country
    for service_raster in aligned_inputs['service_raster_list']:
        source = re.search('LPJ-GUESS_(.+?)_cVeg', service_raster).group(1)
        year = re.search('cVeg_(.+?)_1x1', service_raster).group(1)
        service_by_country = zonal_stats(
            service_raster, aligned_inputs['countries_mask'])
        service_in_KBA_by_country = nested_zonal_stats(
            service_raster, aligned_inputs['countries_mask'],
            aligned_inputs['kba_raster'],
            "F:/carbon_lpj_guess_workspace/aligned_inputs/pixel_area_km2.tif")
        country_zonal_stat_csv = os.path.join(
            result_dir, "zonal_stats_by_country_{}_{}.csv".format(
                source, year))
        summarize_nested_zonal_stats(
            service_by_country, service_in_KBA_by_country,
            country_zonal_stat_csv)
        proportion_in_kba_by_country_raster = os.path.join(
            result_dir,
            "proportion_sum_service_in_kba_by_country_{}_{}.tif".format(
                source, year))
        # proportion of service in KBAs, by country
        zonal_stat_to_raster(
            country_zonal_stat_csv, aligned_inputs['countries_mask'],
            'proportion_service_in_KBAs',
            proportion_in_kba_by_country_raster)
        # total service by country
        sum_by_country_raster = os.path.join(
            result_dir,
            "sum_service_by_country_{}_{}.tif".format(
                source, year))
        zonal_stat_to_raster(
            country_zonal_stat_csv, aligned_inputs['countries_mask'],
            'total_service_sum', sum_by_country_raster)
        service_relative_to_area_by_country_raster = os.path.join(
            result_dir,
            "service_relative_to_area_by_country_{}_{}.tif".format(
                source, year))
        zonal_stat_to_raster(
            country_zonal_stat_csv, aligned_inputs['countries_mask'],
            'service_relative_to_area',
            service_relative_to_area_by_country_raster)


def raster_sum_under_land_area(value_raster_path, land_area_path):
    """Calculate sum of values in a raster falling under a land area mask.

    The two rasters must be aligned, with identical dimensions, pixel size,
    and block size.

    Parameters:
        value_raster_path (string): path to raster whose sum should be
            summarized
        land_area_path (string): path to raster indicating land area

    Returns:
        the sum of valid pixels in `value_raster_path` intersecting valid
            pixels in `land_area_path`

    """
    value_nodata = pygeoprocessing.get_raster_info(
        value_raster_path)['nodata'][0]
    land_area_nodata = pygeoprocessing.get_raster_info(
        land_area_path)['nodata'][0]

    value_raster = gdal.OpenEx(value_raster_path)
    value_band = value_raster.GetRasterBand(1)

    land_raster = gdal.OpenEx(land_area_path)
    land_band = land_raster.GetRasterBand(1)

    value_sum = 0
    try:
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (value_raster_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                value_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(value_band))
                land_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(land_band))
                last_blocksize = blocksize

            value_data = block_offset.copy()
            value_data['buf_obj'] = value_array
            value_band.ReadAsArray(**value_data)

            land_data = block_offset.copy()
            land_data['buf_obj'] = land_array
            land_band.ReadAsArray(**land_data)

            valid_mask = (
                (~numpy.isclose(value_array, value_nodata)) &
                (~numpy.isclose(land_array, land_area_nodata)))
            block_sum = numpy.sum(value_array[valid_mask])
            value_sum = value_sum + block_sum
    finally:
        value_band = None
        land_band = None
        gdal.Dataset.__swig_destroy__(value_raster)
        gdal.Dataset.__swig_destroy__(land_raster)

    return value_sum


def calculate_land_area():
    area_raster = "F:/ESA_landcover_2015/pixel_area_km2.tif"
    land_area_mask = "F:/ESA_landcover_2015/land_mask.tif"
    total_land_area = raster_sum_under_land_area(
        area_raster, land_area_mask)
    print("Total land area: {} km2".format(total_land_area))


def raster_summary(path_list, save_as):
    """Summarize spatial characteristics of a list of rasters.

    Parameters:
        path_list (list): list of paths to rasters that should be summarized
        save_as (string): location to save the summary table

    Side effects:
        creates a csv table at `save_as` containing pixel size, raster
            dimensions, projection, and bounding box of the rasters in
            `path_list`

    Returns:
        None

    """
    raster_info_dict = {
        os.path.basename(path): pygeoprocessing.get_raster_info(
            path) for path in path_list
    }
    df_dict = {
        'service_basename': [*raster_info_dict],
        'pixel_size': [
            raster_info_dict[key]['pixel_size'] for key in raster_info_dict],
        'raster_size': [
            raster_info_dict[key]['raster_size'] for key in raster_info_dict],
        'projection': [
            raster_info_dict[key]['projection'] for key in raster_info_dict],
        'bounding_box': [
            raster_info_dict[key]['bounding_box'] for key in raster_info_dict],
    }
    raster_info_df = pandas.DataFrame(df_dict)
    raster_info_df.to_csv(save_as, index=False)


def extract_by_mask(value_raster_path, mask_raster_path, save_as):
    """Extract the values from one raster lying under valid pixels in mask.

    The two rasters must be aligned and share pixel size.

    Parameters:
        value_raster_path (string): path to raster containing values to extract
        mask_raster_path (string): path to raster containing mask, identified
            by any valid pixels
        save_as (string): path to location where extracted raster should be
            saved

    Side effects:
        creates new raster at the location specified by `save_as`, containing
            values from `value_raster_path` intersecting. It will have the same
            data type and nodata value as the value raster

    Returns:
        None

    """
    def extract_op(value_ar, mask_ar):
        valid_mask = (
            (~numpy.isclose(value_ar, value_nodata)) &
            (~numpy.isclose(mask_ar, mask_nodata)))
        extracted = numpy.empty(value_ar.shape, dtype=target_numpy_datatype)
        extracted[:] = value_nodata
        extracted[valid_mask] = value_ar[valid_mask]
        return extracted

    # make sure the two rasters are aligned
    raster_info_list = [
        pygeoprocessing.get_raster_info(path)
        for path in [value_raster_path, mask_raster_path]]
    size_set = set()
    for raster_info in raster_info_list:
        size_set.add(raster_info['raster_size'])
    if len(size_set) > 1:
        raise ValueError("Input Rasters are not the same dimensions")
    # pixel_set = set()
    # for raster_info in raster_info_list:
    #     pixel_set.add(raster_info['pixel_size'])
    # if len(pixel_set) > 1:
    #     raise ValueError("Input Rasters do not have same pixel size")

    value_nodata = pygeoprocessing.get_raster_info(
        value_raster_path)['nodata'][0]
    value_datatype = pygeoprocessing.get_raster_info(
        value_raster_path)['datatype']
    mask_nodata = pygeoprocessing.get_raster_info(
        mask_raster_path)['nodata'][0]

    # get GDAL data type
    value_raster = gdal.OpenEx(value_raster_path)
    value_band = value_raster.GetRasterBand(1)
    target_numpy_datatype = pygeoprocessing._gdal_to_numpy_type(value_band)
    value_band = None
    gdal.Dataset.__swig_destroy__(value_raster)

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [value_raster_path, mask_raster_path]],
        extract_op, save_as, value_datatype, value_nodata)


def extract_area_by_habitat():
    """Throwaway."""
    area_raster = "F:/Data_marine_coastal_habitat/aligned/area_km2.tif"
    mask_raster = "F:/Data_marine_coastal_habitat/habitat_mosaic.tif"
    save_as = "F:/Data_marine_coastal_habitat/pixel_area_habitat_mosaic.tif"
    extract_by_mask(area_raster, mask_raster, save_as)


def mosaic_habitat_rasters(target_path):
    """Mosaic a list of rasters together into one mosaic raster.

    Given a list of rasters depicting habitat, mosaic the rasters together. The
    result will be a mosaic raster of integer type containing the value "1" in
    pixels where any of the input rasters has a valid value (i.e., a value
    other than nodata).

    Parameters:
        target_path (string): location to save the result

    Side effects:
        creates a raster at `target_path` containing the value 1 in pixels
            where any of the input rasters contain a valid value

    Returns:
        None

    """
    def mosaic_valid_pixels(*raster_list):
        """Mosaic valid pixels from a list of rasters.

        Where any raster in raster_list has a value other than nodata, the
        value in the mosaic should be 1.

        Returns:
            A raster with the same extent as the rasters in raster_list,
                containing the value 1 in pixels where any of the rasters in
                raster_list has a valid value
        """
        result = numpy.empty(raster_list[0].shape, dtype=numpy.int16)
        result[:] = 1
        invalid_mask = numpy.all(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        result[invalid_mask] = _TARGET_NODATA
        return result

    input_dir = "F:/Data_marine_coastal_habitat"
    # basenames of rasters to mosaic together
    habitat_raster_bn_list = [
        '1_2000_mask_md5_91e7f997e1197e4a2abf064095e2179e.tif',
        '2_1000_mask_md5_f428433ea05cf1c7960ec7ff3995c5aa.tif',
        '2_2000_mask_md5_1ffc23cd09f748e1fe5a996b72df3757.tif',
        '4_500_mask_md5_6f48797ca1ab8e953e32288efd0536a4.tif',
        'ipbes-cv_mangrove_md5_2205f546ab3eb92f9901b3e57258b998.tif',
        'ipbes-cv_reef_wgs84_compressed_md5_96d95cc4f2c5348394eccff9e8b84e6b.tif',
        'ipbes-cv_saltmarsh_md5_203d8600fd4b6df91f53f66f2a011bcd.tif',
        'ipbes-cv_seagrass_md5_a9cc6d922d2e74a14f74b4107c94a0d6.tif']
    input_path_list = [
        os.path.join(input_dir, bn) for bn in habitat_raster_bn_list]
    raster_info_list = [
        pygeoprocessing.get_raster_info(path) for path in input_path_list]
    # pixel_size_set = set([info['pixel_size'] for info in raster_info_list])
    # if len(pixel_size_set) > 1:
    #     raise ValueError("Input rasters contain more than one pixel size")
    input_pixel_size = pygeoprocessing.get_raster_info(
        input_path_list[0])['pixel_size']
    nodata_value_set = set([info['nodata'][0] for info in raster_info_list])
    if len(nodata_value_set) > 1:
        raise ValueError("Input rasters contain more than one nodata value")
    input_nodata = list(nodata_value_set)[0]

    aligned_dir = os.path.join(input_dir, 'aligned')
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)
    aligned_path_list = [
        os.path.join(aligned_dir, bn) for bn in habitat_raster_bn_list]
    # pygeoprocessing.align_and_resize_raster_stack(
    #     input_path_list, aligned_path_list, ['near'] * len(input_path_list),
    #     input_pixel_size, 'union')
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in aligned_path_list], mosaic_valid_pixels,
        target_path, gdal.GDT_Int16, _TARGET_NODATA)



def merge_data_frame_list(df_path_list, fid_field, save_as):
    """Merge the data frames in `df_path_list` and save as one data frame.

    Merge each data frame in `df_path_list` into a single data frame. Save this
    merged data frame as `save_as`.

    Parameters:
        df_path_list (list): list of file paths indicating locations of data
            frames that should be merged. Each must include a column fid_field
            identifying unique watersheds or monitoring stations, and a column
            of covariate data
        fid_field (string): field that identifies features
        save_as (string): path to location on disk where the result should be
            saved

    Returns:
        None

    """
    combined_df = pandas.read_csv(df_path_list[0])
    df_i = 1
    while df_i < len(df_path_list):
        combined_df = combined_df.merge(
            pandas.read_csv(df_path_list[df_i]), on=fid_field,
            suffixes=(False, False), validate="one_to_one")
        df_i = df_i + 1
    combined_df.to_csv(save_as, index=False)


def map_FID_to_field(shp_path, field):
    """Map FID of each feature, according to GetFID(), to the given field.

    This allows for mapping of a dictionary of zonal statistics, where keys
    correspond to FID according to GetFID(), to another field that is preferred
    to identify features.

    Parameters:
        shp_path (string): path to shapefile
        field (string): the field to map to FID

    Returns:
        dictionary indexed by the FID of each feature retrieved with GetFID(),
            and values are the value of `field` for the feature

    """
    vector = gdal.OpenEx(shp_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    FID_to_field = {
        feature.GetFID(): feature.GetField(field) for feature in layer}

    # clean up
    vector = None
    layer = None
    return FID_to_field


def service_zonal_stats(aggregate_vector_path, fid_field, output_dir, save_as):
    """Calculate zonal stats from service rasters within polygon features.

    Parameters:
        aggregate_vector_path (string): a path to a polygon vector containing
            zones to summarize services within
        fid_field (string): field in aggregate_vector_path that identifies
            features
        output_dir (string): path to directory to store results and
            intermediate files
        save_as (string): file location to save summarized zonal statistics

    Returns:
        None

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    service_raster_dir = "F:/Data_service_rasters"
    service_raster_bn_list = [
        f for f in os.listdir(service_raster_dir) if f.endswith('.tif')]
    service_raster_bn_list.remove(
        'realized_natureaccess10_nathab_md5_af07e76ecea7fb5be0fa307dc7ff4eed.tif')
    service_raster_bn_list.remove(
        'realized_natureaccess100_nathab_md5_ac72bb0f6c0460d7d48f7ee25e161b0f.tif')
    service_raster_path_list = [
        os.path.join(service_raster_dir, bn) for bn in service_raster_bn_list]
    service_raster_path_list.append(
        "F:/Data_service_rasters/natureaccess_aligned/realized_natureaccess100_nathab_md5_ac72bb0f6c0460d7d48f7ee25e161b0f.tif")
    service_raster_path_list.append(
        "F:/Data_service_rasters/natureaccess_aligned/realized_natureaccess10_nathab_md5_af07e76ecea7fb5be0fa307dc7ff4eed.tif")
    fid_to_objectid = map_FID_to_field(aggregate_vector_path, fid_field)
    df_path_list = []
    for raster_path in service_raster_path_list:
        if os.path.basename(raster_path).startswith('realized_nature'):
            colname = os.path.basename(raster_path)[9:24]
        else:
            colname = os.path.basename(raster_path)[9:15]
        intermediate_path = os.path.join(
            output_dir, 'zonal_stat_biome_{}.csv'.format(colname))
        df_path_list.append(intermediate_path)
        if not os.path.exists(intermediate_path):
            print("processing {} under {}".format(
                colname, aggregate_vector_path))
            zonal_stat_dict = pygeoprocessing.zonal_statistics(
                (raster_path, 1), aggregate_vector_path,
                polygons_might_overlap=False)
            objectid_zonal_stats_dict = {
                objectid: zonal_stat_dict[fid] for (fid, objectid) in
                fid_to_objectid.items()
            }
            zonal_df = pandas.DataFrame(
                {
                    fid_field: [
                        key for key, value in sorted(
                            objectid_zonal_stats_dict.items())],
                    '{}_sum'.format(colname): [
                        value['sum'] for key, value in
                        sorted(objectid_zonal_stats_dict.items())],
                    '{}_count'.format(colname): [
                        value['count'] for key, value in
                        sorted(objectid_zonal_stats_dict.items())]
                })
            zonal_df['{}_mean'.format(colname)] = (
                zonal_df['{}_sum'.format(colname)] /
                zonal_df['{}_count'.format(colname)])
            zonal_df.to_csv(intermediate_path, index=False)
    merge_data_frame_list(df_path_list, fid_field, save_as)


def biome_zonal_stats():
    """Calculate zonal stats from service rasters within biomes."""
    biome_shp_path = "F:/Data_terrestrial_ecoregions/wwf_terr_ecos_diss.shp"
    fid_field = 'OBJECTID'
    output_dir = "C:/Users/ginge/Dropbox/NatCap_backup/KBA+ES/processing_2020/zonal_stats_biome"
    save_as = os.path.join(output_dir, 'zonal_stat_biome_combined.csv')
    service_zonal_stats(biome_shp_path, fid_field, output_dir, save_as)


def country_zonal_stats():
    """Calculate zonal stats from service rasters within countries."""
    countries_shp_path = "F:/Data_service_rasters/service_by_country/TM_WORLD_BORDERS-0.3.shp"
    fid_field = 'ISO3'
    output_dir = "C:/Users/ginge/Dropbox/NatCap_backup/KBA+ES/processing_2020/zonal_stats_countries"
    save_as = os.path.join(output_dir, 'zonal_stat_countries_combined.csv')
    service_zonal_stats(countries_shp_path, fid_field, output_dir, save_as)


if __name__ == '__main__':
    pollination_workspace = "F:/"
    # pollination_workflow(pollination_workspace)
    cv_workspace = "F:/cv_workspace"
    # cv_habitat_attribution_workflow(cv_workspace)
    # coastal_vulnerability_workflow(cv_workspace)
    # cv_habitat_in_kbas(cv_workspace)
    ndr_workspace = 'F:/ndr_workspace'
    # ndr_workflow(ndr_workspace)
    lpjguess_workspace = 'F:/carbon_lpj_guess_workspace/'
    # LPJ_carbon_workflow(lpjguess_workspace)
    # area_of_kbas(lpjguess_workspace)
    # extract_area_by_land()
    target_path = "F:/Data_marine_coastal_habitat/habitat_mosaic.tif"
    # mosaic_habitat_rasters(target_path)
    # extract_area_by_habitat()
    biome_zonal_stats()
    country_zonal_stats()
