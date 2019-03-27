"""Geoprocessing workflow: Intersection of ES with KBA.

Calculate the spatial intersection of ecosystem service (ES) provision from
IP-BES results with areas protected as Key Biodiversity Areas (KBA).
"""
import os
import tempfile

from osgeo import gdal
import numpy
import pandas
from datetime import datetime
import urllib2
# import arcpy

import pygeoprocessing
# from arcpy import sa

# arcpy.CheckOutExtension("Spatial")

_TARGET_NODATA = -9999


def base_data_dict(resolution):
    """Base data input locations on disk."""
    if resolution == '10km':
        data_dict = {
            'kba_raster': 'C:/Users/ginge/Dropbox/KBA_ES/Global_KBA_10km.tif',
            'countries_mask': 'C:/Users/ginge/Dropbox/KBA_ES/Mark_Mulligan_data/onekmgaul_country__world/onekmgaul_country__world.asc',
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
    }
    summary_df = pandas.DataFrame(data=country_dict)
    summary_df['proportion_service_in_KBAs'] = (
        summary_df.service_sum_in_KBAs / summary_df.total_service_sum)
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
        service_raster_path, country_raster_path, kba_raster_path):
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

            country_values = numpy.unique(
                country_array[country_array != country_nodata])
            for country_id in country_values:
                valid_mask = (
                    (service_array != service_nodata) &
                    (kba_array > 0) &
                    (country_array == country_id))
                valid_block = service_array[valid_mask]
                country_in_kba_sum = numpy.sum(valid_block)
                if country_in_kba_sum > 0:
                    country_in_kba_avg = country_in_kba_sum / valid_block.size
                else:
                    country_in_kba_avg = 0
                if country_id in zonal_stat_dict:
                    zonal_stat_dict[country_id]['sum'] += country_in_kba_sum
                    zonal_stat_dict[country_id]['average'] += (
                        country_in_kba_avg)
                else:
                    zonal_stat_dict[country_id] = {
                        'sum': country_in_kba_sum,
                        'average': country_in_kba_avg,
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
        """Extract values from value_raster under mask_raster."""
        valid_mask = (
            (value_raster != value_nodata) &
            (mask_raster != mask_nodata))
        result = numpy.empty(value_raster.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = value_raster[valid_mask]
        return result

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

    cv_results_shp = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/CV_outputs_11.28.18/cv_11_28_18.gdb/cv_outputs_proj"
    service_field = "Service_GK"
    habitat_dict = {
        'mangrove': {
            'risk': 1,
            'effect_distance': 1000,
            'habitat_shp': r"C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/habitat/DataPack-14_001_WCMC010_MangrovesUSGS2011_v1_3/01_Data/14_001_WCMC010_MangroveUSGS2011_v1_3.shp",
        },
        'saltmarsh': {
            'risk': 2,
            'effect_distance': 1000,
            'habitat_shp': r"C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/habitat/datapack-14_001_wcmc027_saltmarsh2017_v4/01_Data/14_001_WCMC027_Saltmarsh_py_v4.shp",
        },
        'coralreef': {
            'risk': 1,
            'effect_distance': 2000,
            'habitat_shp': r"C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/habitat/DataPack-14_001_WCMC008_CoralReef2010_v1_3/01_Data/14_001_WCMC008_CoralReef2010_v1_3.shp",
        },
        'seagrass': {
            'risk': 4,
            'effect_distance': 500,
            'habitat_shp': r"C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/habitat/datapack-14_001_wcmc013_014_seagrassptpy_v4/01_Data/WCMC_013_014_SeagrassesPy_v4.shp",
        },
    }
    # moving window analysis: mean of service values within effect distance
    # of each pixel
    masked_habitat_list = []
    for habitat_type in habitat_dict.iterkeys():
        effect_distance = convert_m_to_deg_equator(
                habitat_dict[habitat_type]['effect_distance'])
        target_pixel_size = effect_distance / 2
        point_stat_raster_path = os.path.join(
            workspace_dir, 'pt_st_{}.tif'.format(habitat_type[:3]))
        if not os.path.exists(point_stat_raster_path):
            neighborhood = arcpy.sa.NbrCircle(effect_distance, "MAP")
            point_statistics_raster = arcpy.sa.PointStatistics(
                cv_results_shp, field=service_field,
                cell_size=target_pixel_size, neighborhood=neighborhood,
                statistics_type="MEAN")
            point_statistics_raster.save(point_stat_raster_path)

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
            arcpy.PolygonToRaster_conversion(
                habitat_dict[habitat_type]['habitat_shp'], raster_field,
                habitat_raster_path, cell_assignment="MAXIMUM_AREA")

        # align habitat raster with point stat raster exactly
        rasters_to_align = [point_stat_raster_path, habitat_raster_path]
        aligned_rasters = [
            os.path.join(workspace_dir, '{}_aligned.tif'.format(
                os.path.basename(r)[:-4])) for r in rasters_to_align]
        if not all([os.path.exists(p) for p in aligned_rasters]):
            pygeoprocessing.align_and_resize_raster_stack(
                rasters_to_align, aligned_rasters,
                ['near'] * len(rasters_to_align),
                (target_pixel_size, target_pixel_size),
                bounding_box_mode="union")

        # mask out habitat pixels from point statistics raster
        point_stat_habitat_path = os.path.join(
            workspace_dir, 'pt_st_hab_{}.tif'.format(habitat_type[:3]))
        if not os.path.exists(point_stat_habitat_path):
            value_raster = os.path.join(
                workspace_dir, '{}_aligned.tif'.format(
                    os.path.basename(point_stat_raster_path)[:-4]))
            mask_raster = os.path.join(
                workspace_dir, '{}_aligned.tif'.format(
                    os.path.basename(habitat_raster_path)[:-4]))
            value_nodata = pygeoprocessing.get_raster_info(
                value_raster)['nodata'][0]
            mask_nodata = pygeoprocessing.get_raster_info(
                mask_raster)['nodata'][0]
            pygeoprocessing.raster_calculator(
                [(path, 1) for path in [
                    value_raster, mask_raster]],
                extract_by_mask, point_stat_habitat_path, gdal.GDT_Float32,
                _TARGET_NODATA)
        masked_habitat_list.append(point_stat_habitat_path)

    # try saving memory by saving to int
    int_raster_list = []
    for habitat_raster in masked_habitat_list:
        int_raster_path = os.path.join(
            workspace_dir,
            '{}_x1000.tif'.format(os.path.basename(habitat_raster)[:-4]))
        if not os.path.exists(int_raster_path):
            pygeoprocessing.raster_calculator(
                [(habitat_raster, 1)], multiply_by_1000, int_raster_path,
                gdal.GDT_Int16, _TARGET_NODATA)
        int_raster_list.append(int_raster_path)

    # do it in Arc
    pygeoprocessing.new_raster_from_base(
        os.path.join(workspace_dir, 'pt_st_hab_cor_x1000.tif'),
        os.path.join(workspace_dir, 'service_mosaic.tif'), gdal.GDT_Int16,
        band_nodata_list=[_TARGET_NODATA], fill_value_list=[_TARGET_NODATA])
    # arcpy.MosaicToNewRaster_management(input_rasters="pt_st_hab_sea_x1000.tif;
    #   pt_st_hab_cor_x1000.tif;pt_st_hab_sal_x1000.tif;pt_st_hab_man_x1000.tif",
    #   output_location="C:/Users/ginge/Desktop/CV",
    #   raster_dataset_name_with_extension="service_mosaic.tif",
    #   coordinate_system_for_the_raster="GEOGCS[
    #     'GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,
    #     298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',
    #     0.0174532925199433]]", pixel_type="16_BIT_SIGNED", cellsize="",
    #   number_of_bands="1", mosaic_method="MAXIMUM",
    #   mosaic_colormap_mode="FIRST")
    return

    # combine masked habitat pixels across habitat types
    attributed_service_path = os.path.join(
        workspace_dir, 'attributed_service.tif')
    aligned_dir = os.path.join(workspace_dir, 'aligned_habitat')
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)
    aligned_habitat_list = [
        os.path.join(
            aligned_dir, os.path.basename(r)) for r in int_raster_list]
    if not all([os.path.exists(p) for p in aligned_habitat_list]):
        max_pixel_size = max(
            [habitat_dict[i]['effect_distance'] / 2 for i in
                habitat_dict.iterkeys()])
        pygeoprocessing.align_and_resize_raster_stack(
            int_raster_list, aligned_habitat_list,
            ['near'] * len(int_raster_list), (max_pixel_size, max_pixel_size),
            bounding_box_mode="union", raster_align_index=1)
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in aligned_habitat_list], mosaic_max_value,
        attributed_service_path, gdal.GDT_Float32, _TARGET_NODATA)


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
        workspace_dir, service_raster_list, resolution):
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
    service_raster_list = [
        "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_en_10s_ESACCI_LC_L4_LCSS.tif",
        "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_fo_10s_ESACCI_LC_L4_LCSS.tif",
        "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_va_10s_ESACCI_LC_L4_LCSS.tif",
    ]
    global_summary_dict = {
        'spatial_resolution': [],
        'service': [],
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'global_service_percent_in_KBA': [],
        }
    for resolution in ['10kmMark', '10km', '300m']:
        workspace_dir = os.path.join(outer_workspace_dir, resolution)
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        aligned_inputs = process_pollination_service_rasters(
            workspace_dir, service_raster_list, resolution)
        # rescale to 0 - 1?
        # combine the 3 nutrients?

        result_dir = os.path.join(workspace_dir, 'summary_tables_and_maps')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # spatial summaries
        for service_raster in aligned_inputs['service_raster_list']:
            # global summary of service sum and sum in KBAs
            summary_dict = global_kba_summary(
                service_raster, aligned_inputs['kba_raster'])
            global_summary_dict['spatial_resolution'].append(resolution)
            global_summary_dict['service'].append(
                os.path.basename(service_raster))
            global_summary_dict['global_service_sum'].append(
                summary_dict['global_service_sum'])
            global_summary_dict['global_service_sum_in_KBA'].append(
                summary_dict['global_service_sum_in_KBA'])
            global_summary_dict['global_service_percent_in_KBA'].append(
                summary_dict['global_service_percent_in_KBA'])

            # sum and average of service values within each KBA
            service_by_KBA = zonal_stats(
                service_raster, aligned_inputs['kba_raster'])
            KBA_zonal_stat_csv = os.path.join(
                result_dir, "zonal_stats_by_KBA_{}_{}.csv".format(
                    os.path.basename(service_raster), resolution))
            # zonal_stats_to_csv(service_by_KBA, 'KBA_id', KBA_zonal_stat_csv)
            avg_by_KBA_raster = os.path.join(
                result_dir, "average_service_by_KBA_{}.tif".format(
                    os.path.basename(service_raster)))
            zonal_stat_to_raster(
                KBA_zonal_stat_csv, aligned_inputs['kba_raster'], 'average',
                avg_by_KBA_raster)
            sum_by_KBA_raster = os.path.join(
                result_dir, "sum_service_by_KBA_{}.tif".format(
                    os.path.basename(service_raster)))
            zonal_stat_to_raster(
                KBA_zonal_stat_csv, aligned_inputs['kba_raster'], 'sum',
                sum_by_KBA_raster)

            # sum and average of service values within each country
            service_by_country = zonal_stats(
                service_raster, aligned_inputs['countries_mask'])
            service_in_KBA_by_country = nested_zonal_stats(
                service_raster, aligned_inputs['countries_mask'],
                aligned_inputs['kba_raster'])
            country_zonal_stat_csv = os.path.join(
                result_dir, "zonal_stats_by_country_{}_{}.csv".format(
                    os.path.basename(service_raster), resolution))
            summarize_nested_zonal_stats(
                service_by_country, service_in_KBA_by_country,
                country_zonal_stat_csv)
            proportion_in_kba_by_country_raster = os.path.join(
                result_dir,
                "proportion_sum_service_in_kba_by_country_{}_{}.tif".format(
                    os.path.basename(service_raster), resolution))
            zonal_stat_to_raster(
                country_zonal_stat_csv, aligned_inputs['countries_mask'],
                'proportion_service_in_KBAs',
                proportion_in_kba_by_country_raster)
    global_summary_df = pandas.DataFrame(data=global_summary_dict)
    global_summary_df.to_csv(os.path.join(
        outer_workspace_dir, 'global_service_summary.csv'), index=False)


def coastal_vulnerability_workflow(outer_workspace_dir):
    """Summarize coastal vulnerability service in and outside KBAs.

    Parameters:
        outer_workspace_dir (string): path to folder where outputs
            will be created

    Returns:
        None
    """
    service_raster = "C:/Users/ginge/Dropbox/KBA_ES/CV_summary_11.30.18/habitat_attribution_rasters/service_mosaic.tif"
    global_summary_dict = {
        'spatial_resolution': [],
        'service': [],
        'global_service_sum': [],
        'global_service_sum_in_KBA': [],
        'global_service_percent_in_KBA': [],
        }
    for resolution in ['10km', '300m']:
        workspace_dir = os.path.join(outer_workspace_dir, resolution)
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        aligned_inputs = process_cv_rasters(
            workspace_dir, service_raster, resolution)
        result_dir = os.path.join(workspace_dir, 'summary_tables_and_maps')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # global summary of service sum and sum in KBAs
        summary_dict = global_kba_summary(
            aligned_inputs['service_raster'], aligned_inputs['kba_raster'])
        global_summary_dict['spatial_resolution'].append(resolution)
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
        outer_workspace_dir, 'global_service_summary_TEST.csv'), index=False)


def process_cv_rasters(workspace_dir, service_raster, resolution):
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
    for pollination service). The habitat raster is a mosaic of all marine
    habitat types, where I did the mosaic operation in Arc.

    Returns:
        none
    """
    area_dict = {
        'total_habitat_pixels': [],
        'habitat_in_KBA_pixels': [],
        'KBA_percent_of_habitat_pixels': [],
    }
    habitat_raster_path = r"C:/Users/ginge/Dropbox/KBA_ES/CV_summary_11.30.18/habitat_attribution_rasters/habitat_mosaic_10km.tif"
    kba_raster_path = r"C:/Users/ginge/Dropbox/KBA_ES/CV_summary_11.30.18/10km/aligned_inputs/Global_KBA_10km.tif"

    habitat_raster = gdal.OpenEx(habitat_raster_path)
    habitat_band = habitat_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(kba_raster_path)
    kba_band = kba_raster.GetRasterBand(1)

    try:
        total_habitat_pixels = 0
        total_kba_pixels = 0
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (habitat_raster_path, 1), offset_only=True):
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
        outer_workspace_dir, 'KBA_area_summary.csv'), index=False)


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
        'total_land_pixels': [],
        'KBA_pixels': [],
        'KBA_percent_of_land_pixels': [],
    }
    service_raster_list = [
        "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_en_10s_ESACCI_LC_L4_LCSS.tif"]
    for resolution in ['10kmMark', '10km', '300m']:
        workspace_dir = os.path.join(outer_workspace_dir, resolution)
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        aligned_inputs = process_pollination_service_rasters(
            workspace_dir, service_raster_list, resolution)

        countries_raster_path = aligned_inputs['countries_mask']
        kba_raster_path = aligned_inputs['kba_raster']

        countries_raster = gdal.OpenEx(countries_raster_path)
        countries_band = countries_raster.GetRasterBand(1)

        kba_raster = gdal.OpenEx(kba_raster_path)
        kba_band = kba_raster.GetRasterBand(1)

        try:
            total_land_pixels = 0
            total_kba_pixels = 0
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

                countries_data = block_offset.copy()
                countries_data['buf_obj'] = countries_array
                countries_band.ReadAsArray(**countries_data)

                kba_data = block_offset.copy()
                kba_data['buf_obj'] = kba_array
                kba_band.ReadAsArray(**kba_data)

                country_mask = (countries_array > 0)
                kba_mask = (
                    (countries_array > 0) &
                    (kba_array > 0))
                block_land_pixels = countries_array[country_mask].size
                block_kba_pixels = countries_array[kba_mask].size

                total_land_pixels += block_land_pixels
                total_kba_pixels += block_kba_pixels
            area_dict['spatial_resolution'].append(resolution)
            area_dict['total_land_pixels'].append(total_land_pixels)
            area_dict['KBA_pixels'].append(total_kba_pixels)
            area_dict['KBA_percent_of_land_pixels'].append(
                (float(total_kba_pixels) / total_land_pixels) * 100)
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
        print "processing tile {} of 280".format(current_object)
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
        print "rasterizing aligned KBA raster"
        pygeoprocessing.rasterize(kba_shp, kba_raster_path, burn_values=[1])
        # calculate zonal stats for tile
        print "calculating zonal stats for tile"
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

        print "processing 10km tile {} of {}".format(
            current_object, len(tile_basename_list))
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


def download_data():
    """Download the latest IPBES results."""
    download_table = r"C:\Users\ginge\Dropbox\NatCap_backup\KBA+ES\ipbes_download_table_3.24.19.csv"
    save_dir = r"F:\IPBES_data_layers_3.24.19"

    dl_df = pandas.read_csv(download_table).set_index('DATASET DESCRIPTION')
    for row_i in xrange(dl_df.shape[0]):
        url = dl_df.ix[row_i]['DOWNLOAD LINK']
        local_destination = os.path.join(
            save_dir, os.path.basename(url))
        if not os.path.exists(local_destination):
            print "downloading {}".format(os.path.basename(url))
            data = urllib2.urlopen(url)
            with open(local_destination,'w') as f:
                while True:
                    tmp = data.read(1024)
                    if not tmp:
                        break
                    f.write(tmp)


if __name__ == '__main__':
    # outer_workspace_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/pollination_summary_11.30.18"
    # pollination_workflow(workspace_dir)
    # area_of_kbas(outer_workspace_dir)
    # outer_workspace_dir = r"C:/Users/ginge/Dropbox/KBA_ES/CV_summary_11.30.18"
    # coastal_vulnerability_workflow(outer_workspace_dir)
    # cv_habitat_in_kbas(outer_workspace_dir)
    # carbon_workflow()
    # download_data()
