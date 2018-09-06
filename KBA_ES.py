"""Geoprocessing workflow: Intersection of ES with KBA.

Calculate the spatial intersection of ecosystem service (ES) provision from
IP-BES results with areas protected as Key Biodiversity Areas (KBA).
"""
import os

from osgeo import gdal
import numpy
import pandas
import arcpy

import pygeoprocessing
from arcpy import sa

_TARGET_NODATA = -9999


def base_data_dict():
    """Base data input locations on disk."""
    data_dict = {
        'kba_raster': 'C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/Mark_Mulligan_data/kbas1k/kbas1k.asc',
        'countries_mask': 'C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/Mark_Mulligan_data/onekmgaul_country__world/onekmgaul_country__world.asc',
    }
    return data_dict


def calc_masked_sum(service_raster_path, global_land_mask):
    """Calculate the global sum across pixels in a service raster.

    Parameters:
        service_raster_path (string): path to a raster of service provision
            values
        global_land_mask_path (string): path to a raster containing a mask of
            global land areas.

    Returns:
        the sum of all valid pixels across the service raster within valid
            areas of the global land mask
    """
    service_nodata = pygeoprocessing.get_raster_info(
        service_raster_path)['nodata'][0]
    land_mask_nodata = pygeoprocessing.get_raster_info(
        global_land_mask)['nodata'][0]

    service_raster = gdal.OpenEx(service_raster_path)
    service_band = service_raster.GetRasterBand(1)

    land_raster = gdal.OpenEx(global_land_mask)
    land_band = land_raster.GetRasterBand(1)

    try:
        masked_sum = 0
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                service_raster_path, offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                service_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(service_band))
                land_mask_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(land_band))
                last_blocksize = blocksize

            service_data = block_offset.copy()
            service_data['buf_obj'] = service_array
            service_band.ReadAsArray(**service_data)

            land_mask_data = block_offset.copy()
            land_mask_data['buf_obj'] = land_mask_array
            land_band.ReadAsArray(**land_mask_data)

            valid_mask = (
                (service_array != service_nodata) &
                (land_mask_array != land_mask_nodata))

            valid_block = service_array[valid_mask]
            masked_sum += numpy.sum(valid_block)
    finally:
        service_band = None
        land_band = None
        gdal.Dataset.__swig_destroy__(service_raster)
        gdal.Dataset.__swig_destroy__(land_raster)

    return masked_sum


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
                zonal_raster_path, offset_only=True):
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
                country_raster_path, offset_only=True):
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


def cv_habitat_attribution_workflow(workspace_dir):
    """Use focal statistics to attribute cv service value to marine habitat.

    For each marine habitat type, use a moving window analysis to calculate
    average service value of points falling within a search radius of habitat
    pixels. The search radius for each habitat type corresponds to the effect
    distance for that habitat type used by Rich for the IPBES analysis.

    Parameters:

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

    cv_results_shp = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/coastal_vulnerability/CV_outputs_8.28.18/CV_outputs.shp"
    service_field = "NCP_c"
    habitat_dict = {
        'mangrove': {
            'risk': 1,
            'effect_distance': 1000,
        },
        'saltmarsh': {
            'risk': 2,
            'effect_distance': 1000,
        },
        'coralreef': {
            'risk': 1,
            'effect_distance': 2000,
        },
        'seagrass': {
            'risk': 4,
            'effect_distance': 500,
        },
    }
    # moving window analysis: mean of service values within effect distance
    # of each pixel
    for habitat_type in habitat_dict.iterkeys():
        effect_distance = convert_m_to_deg_equator(
            habitat_dict[habitat_type]['effect_distance'])
        target_pixel_size = effect_distance / 2
        neighborhood = arcpy.sa.NbrCircle(effect_distance, "MAP")
        save_as = os.path.join(
            workspace_dir, 'point_statistics_{}.tif'.format(habitat_type))
        point_statistics_raster = arcpy.sa.PointStatistics(
            cv_results_shp, field=service_field, cell_size=target_pixel_size,
            neighborhood=neighborhood, statistics_type="MEAN")
        point_statistics_raster.save(save_as)

    # mask out habitat pixels only




def process_pollination_service_rasters(workspace_dir, service_raster_list):
    """Process service rasters for summarizing via KBA and country zonal stats.

    First reclassify areas of nodata in the service rasters under the country
    mask to zero. Then resample service rasters to match KBA and country
    rasters.

    Parameters:
        workspace_dir (string): path to directory where intermediate and output
            datasets will be written
        service_raster_list (list): list of paths to raw service rasters that
            should be pre-processed prior to summarizing via country and KBA
            zonal stats

    Returns:
        dictionary of aligned inputs
    """
    data_dict = base_data_dict()
    aligned_raster_dir = os.path.join(workspace_dir, 'aligned_inputs')
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

    # align service rasters with KBA raster via nearest neighbor resampling
    input_path_list = (
        [data_dict['kba_raster'], data_dict['countries_mask']] +
        [os.path.join(blockstat_dir, os.path.basename(r)) for r in
            service_raster_list])
    aligned_path_list = [
        os.path.join(aligned_raster_dir, os.path.basename(r)) for r in
        input_path_list]
    pygeoprocessing.align_and_resize_raster_stack(
        input_path_list, aligned_path_list,
        ['nearest'] * len(aligned_path_list), kba_pixel_size,
        bounding_box_mode="union", raster_align_index=0)

    return aligned_inputs_dict


def pollination_workflow(workspace_dir):
    """Frame out workflow to summarize one service raster.

    Parameters:
        workspace_dir (string): path to folder where outputs will be created

    Returns:
        none
    """
    service_raster_list = [
        "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_en_10s_cur.tif",
        "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_fo_10s_cur.tif",
        "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/IPBES_data_layers/pollination/prod_poll_dep_realized_va_10s_cur.tif",
    ]
    aligned_inputs = process_pollination_service_rasters(
        workspace_dir, service_raster_list)

    # TODO rescale to 0 - 1?
    # TODO how to combine the 3 nutrients?

    result_dir = os.path.join(workspace_dir, 'summary_tables_and_maps')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # spatial summaries
    for service_raster in aligned_inputs['service_raster_list']:
        # sum and average of service values within each KBA
        service_by_KBA = zonal_stats(
            service_raster, aligned_inputs['kba_raster'])
        KBA_zonal_stat_csv = os.path.join(
            result_dir, "zonal_stats_by_KBA_{}.csv".format(
                os.path.basename(service_raster)))
        zonal_stats_to_csv(service_by_KBA, 'KBA_id', KBA_zonal_stat_csv)
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
            result_dir, "zonal_stats_by_country_{}.csv".format(
                os.path.basename(service_raster)))
        summarize_nested_zonal_stats(
            service_by_country, service_in_KBA_by_country,
            country_zonal_stat_csv)
        proportion_in_kba_by_country_raster = os.path.join(
            result_dir,
            "proportion_sum_service_in_kba_by_country_{}.tif".format(
                os.path.basename(service_raster)))
        zonal_stat_to_raster(
            country_zonal_stat_csv, aligned_inputs['countries_mask'],
            'proportion_service_in_KBAs', proportion_in_kba_by_country_raster)
        sum_by_country_raster = os.path.join(
            result_dir, "sum_service_by_country_{}.tif".format(
                os.path.basename(service_raster)))


def area_of_kbas():
    """Calculate the area of KBAs relative to land mask."""
    countries_raster_path = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/Mark_Mulligan_data/onekmgaul_country__world/onekmgaul_country__world.asc"
    kba_raster_path = "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/Mark_Mulligan_data/kbas1k/kbas1k.asc"

    countries_raster = gdal.OpenEx(countries_raster_path)
    countries_band = countries_raster.GetRasterBand(1)

    kba_raster = gdal.OpenEx(kba_raster_path)
    kba_band = kba_raster.GetRasterBand(1)

    try:
        total_country_pixels = 0
        total_kba_pixels = 0
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                countries_raster_path, offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                countries_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(countries_band))
                kba_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(kba_band))

            countries_data = block_offset.copy()
            countries_data['buf_obj'] = countries_array
            countries_band.ReadAsArray(**countries_data)

            kba_data = block_offset.copy()
            kba_data['buf_obj'] = kba_array
            kba_band.ReadAsArray(**kba_data)

            country_mask = (country_array > 0)
            kba_mask = (country_array > 0 & kba_array > 0)
            block_country_pixels = country_array[country_mask].size()  # TODO count # pixels??
            block_kba_pixels = country_array[kba_mask].size()

            total_country_pixels += block_country_pixels
            total_kba_pixels += block_kba_pixels
        print "total country pixels: {}".format(total_country_pixels)
        print "total kba pixels: {}".format(total_kba_pixels)
        print "% kba by area: {}".format(
            (total_kba_pixels / total_country_pixels) * 100)
    finally:
        countries_band = None
        kba_band = None
        gdal.Dataset.__swig_destroy__(countries_raster)
        gdal.Dataset.__swig_destroy__(kba_raster)


if __name__ == '__main__':
    workspace_dir = "F:/pollination_summary_300m"
    pollination_workflow(workspace_dir)
    # cv_habitat_attribution_workflow(workspace_dir)


