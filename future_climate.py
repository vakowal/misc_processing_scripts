"""Summarize future climate scenarios for Mongolia."""
import os
import tempfile
import collections

import numpy
import pandas

from osgeo import gdal
from osgeo import ogr

import pygeoprocessing


def raster_values_at_points(
        point_shp_path, shp_id_field, raster_path, band, raster_field_name):
    """Collect values from a raster intersecting points in a shapefile.

    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates and must have a field identifying sites, shp_id_field
        shp_id_field (string): field in point_shp_path identifying features
        raster_path (string): path to raster containing values that should be
            extracted at points
        band (int): band index of the raster to analyze
        raster_field_name (string): name to assign to the field in the data
            frame that contains values extracted from the raster

    Returns:
        a pandas data frame with one column 'shp_id_field' containing values
            of the `shp_id_field` from point features, and one column
            'raster_field_name' containing values from the raster at the point
            location

    """
    raster_nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    point_vector = ogr.Open(point_shp_path)
    point_layer = point_vector.GetLayer()

    # build up a list of the original field names so we can copy it to report
    point_field_name_list = [shp_id_field]

    # this will hold (x,y) coordinates for each point in its iterator order
    point_coord_list = []
    # this maps fieldnames to a list of the values associated with that
    # fieldname in the order that the points are read in and written to
    # `point_coord_list`.
    feature_attributes_fieldname_map = collections.defaultdict(list)
    for point_feature in point_layer:
        sample_point_geometry = point_feature.GetGeometryRef()
        for field_name in point_field_name_list:
            feature_attributes_fieldname_map[field_name].append(
                point_feature.GetField(field_name))
        point_coord_list.append(
            (sample_point_geometry.GetX(), sample_point_geometry.GetY()))
    point_layer = None
    point_vector = None

    # each element will hold the point samples for each raster in the order of
    # `point_coord_list`
    sampled_precip_data_list = []
    for field_name in point_field_name_list:
        sampled_precip_data_list.append(
            pandas.Series(
                data=feature_attributes_fieldname_map[field_name],
                name=field_name))
    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(band)
    geotransform = raster.GetGeoTransform()
    sample_list = []
    for point_x, point_y in point_coord_list:
        raster_x = int((
            point_x - geotransform[0]) / geotransform[1])
        raster_y = int((
            point_y - geotransform[3]) / geotransform[5])
        try:
            sample_list.append(
                band.ReadAsArray(raster_x, raster_y, 1, 1)[0, 0])
        except TypeError:  # access window out of range
            sample_list.append('NA')
    sampled_precip_data_list.append(
        pandas.Series(data=sample_list, name=raster_field_name))

    raster = None
    band = None

    # set nodata values to NA
    report_table = pandas.DataFrame(data=sampled_precip_data_list)
    report_table = report_table.transpose()
    try:
        report_table.loc[
            numpy.isclose(report_table[raster_field_name], raster_nodata),
            raster_field_name] = None
    except TypeError:
        try:
            report_table[raster_field_name] = pandas.to_numeric(
                report_table[raster_field_name], errors='coerce')
            report_table.loc[
                numpy.isclose(report_table[raster_field_name], raster_nodata),
                raster_field_name] = None
        except TypeError:
            report_table.loc[
                pandas.isnull(report_table[raster_field_name]),
                raster_field_name] = None
    return report_table


def average_band_value_in_aoi(raster_path, aoi_path):
    """Calculate the average value across bands of a raster inside an aoi.

    Args:
        raster_path (string): path to raster with 12 bands corresponding to 12
            months of the year
        aoi_path (string): path to polygon vector defining the aoi. Should have
            a single feature

    Returns:
        average value across pixels within the aoi across bands in
            raster_path
    """
    running_sum = 0
    running_count = 0
    for band in range(1, 13):
        zonal_stat_dict = pygeoprocessing.zonal_statistics(
            (raster_path, band), aoi_path)
        if len([*zonal_stat_dict]) > 1:
            raise ValueError("Vector path contains >1 feature")
        running_sum = running_sum + zonal_stat_dict[0]['sum']
        running_count = running_count + zonal_stat_dict[0]['count']
    try:
        mean_value = float(running_sum) / running_count
    except ZeroDivisionError:
        mean_value = 'NA'
    return mean_value


def average_raster_value_in_aoi(raster_list, aoi_path):
    """Calculate the average value in a list of rasters inside an aoi.

    Args:
        raster_list (list): list of paths to rasters that should be summarized
        aoi_path (string): path to polygon vector defining the aoi. Should have
            a single feature

    Returns:
        average value across pixels within the aoi across rasters in
            raster_list
    """
    running_sum = 0
    running_count = 0
    for path in raster_list:
        zonal_stat_dict = pygeoprocessing.zonal_statistics((path, 1), aoi_path)
        if len([*zonal_stat_dict]) > 1:
            raise ValueError("Vector path contains >1 feature")
        running_sum = running_sum + zonal_stat_dict[0]['sum']
        running_count = running_count + zonal_stat_dict[0]['count']
    try:
        mean_value = float(running_sum) / running_count
    except ZeroDivisionError:
        mean_value = 'NA'
    return mean_value


def raster_band_sum(raster_path, input_nodata, target_path, target_nodata):
    """Calculate the sum per pixel across bands in a raster.

    Sum the values in bands of `raster_path` element-wise, treating nodata as
    zero. Areas where all inputs are nodata will be nodata in the output.

    Args:
        raster_path (string): path to raster with 12 bands corresponding to 12
            months of the year
        input_nodata (float or int): nodata value in the input rasters
        target_path (string): path to location to store the result
        target_nodata (float or int): nodata value for the result raster

    Side effects:
        modifies or creates the raster indicated by `target_path`

    Returns:
        None

    """
    def raster_sum_op_nodata_remove(*band_list):
        """Add the rasters in band_list, treating nodata as zero."""
        invalid_mask = numpy.all(
            numpy.isclose(numpy.array(band_list), input_nodata), axis=0)
        for r in band_list:
            numpy.place(r, numpy.isclose(r, input_nodata), [0])
        sum_of_rasters = numpy.sum(band_list, axis=0)
        sum_of_rasters[invalid_mask] = target_nodata
        return sum_of_rasters

    pygeoprocessing.raster_calculator(
        [(raster_path, band) for band in range(1, 13)],
        raster_sum_op_nodata_remove, target_path, gdal.GDT_Float32,
        target_nodata)


def summarize_future_climate():
    """Summarize future climate scenarios for a large part of Mongolia."""
    climate_summary_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/climate/Worldclim_current+future_scenario_summary.csv"
    mongolia_shp_path = "E:/GIS_local/Mongolia/boundaries_etc/Mongolia.shp"
    steppe_shp_path = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop/southeastern_aimags_aoi_WGS84.shp"
    future_dir = "E:/GIS_local_archive/General_useful_data/Worldclim_future_climate/cmip6"
    scenario = 'ssp370'
    resolution = '2.5m'
    time_period = '2061-2080'
    raster_path_bn = 'wc2.1_{}_{}_{}_{}_{}.tif'
    output_list = ['tmin', 'tmax']  # ,'prec'
    full_model_list = [
        f for f in os.listdir(
            os.path.join(
                future_dir, 'share', 'spatial03', 'worldclim', 'cmip6',
                '7_fut', '2.5m'))]
    existing_df = pandas.read_csv(climate_summary_path)
    existing_models = set(
        existing_df[existing_df['output'] == 'tmin']['model'])
    model_list = set(full_model_list).difference(existing_models)
    summary_dict = {
        'model': [],
        'output': [],
        'aoi': [],
        'summary_method': [],
        'mean_value': [],
    }
    for model in model_list:
        for output in output_list:
            # average value across months across pixels in Mongolia
            output_path = os.path.join(
                future_dir, 'share', 'spatial03', 'worldclim', 'cmip6',
                '7_fut', resolution, model, scenario, raster_path_bn.format(
                    resolution, output, model, scenario, time_period))
            print("calc average value in Mongolia: {}".format(
                os.path.basename(output_path)))
            mean_val = average_band_value_in_aoi(
                output_path, mongolia_shp_path)
            summary_dict['model'].append(model)
            summary_dict['output'].append(output)
            summary_dict['aoi'].append('Mongolia')
            summary_dict['summary_method'].append(
                'Monthly average across pixels')
            summary_dict['mean_value'].append(mean_val)

            # average value across months across pixels in Eastern Steppe
            output_path = os.path.join(
                future_dir, 'share', 'spatial03', 'worldclim', 'cmip6',
                '7_fut', resolution, model, scenario, raster_path_bn.format(
                    resolution, output, model, scenario, time_period))
            print("calc average value in Eastern Steppe: {}".format(
                os.path.basename(output_path)))
            mean_val = average_band_value_in_aoi(output_path, steppe_shp_path)
            summary_dict['model'].append(model)
            summary_dict['output'].append(output)
            summary_dict['aoi'].append('Eastern_steppe')
            summary_dict['summary_method'].append(
                'Monthly average across pixels')
            summary_dict['mean_value'].append(mean_val)

        # calculate annual precip
        # with tempfile.NamedTemporaryFile(prefix='raster_sum') as sum_temp_file:
        #     band_sum_path = sum_temp_file.name
        # output_path = os.path.join(
        #         future_dir, 'share', 'spatial03', 'worldclim', 'cmip6',
        #         '7_fut', resolution, model, scenario, raster_path_bn.format(
        #             resolution, 'prec', model, scenario, time_period))
        # nodata_val = pygeoprocessing.get_raster_info(output_path)['nodata'][0]
        # raster_band_sum(output_path, nodata_val, band_sum_path, nodata_val)

        # # average annual precip across pixels in Mongolia
        # sum_zonal_stat = pygeoprocessing.zonal_statistics(
        #     (band_sum_path, 1), mongolia_shp_path)
        # mean_val = float(sum_zonal_stat[0]['sum']) / sum_zonal_stat[0]['count']
        # summary_dict['model'].append(model)
        # summary_dict['output'].append('prec')
        # summary_dict['aoi'].append('Mongolia')
        # summary_dict['summary_method'].append(
        #     'Average annual sum across pixels')
        # summary_dict['mean_value'].append(mean_val)

        # # average annual precip across pixels in Eastern steppe
        # sum_zonal_stat = pygeoprocessing.zonal_statistics(
        #     (band_sum_path, 1), steppe_shp_path)
        # mean_val = float(sum_zonal_stat[0]['sum']) / sum_zonal_stat[0]['count']
        # summary_dict['model'].append(model)
        # summary_dict['output'].append('prec')
        # summary_dict['aoi'].append('Eastern_steppe')
        # summary_dict['summary_method'].append(
        #     'Average annual sum across pixels')
        # summary_dict['mean_value'].append(mean_val)
    summary_df = pandas.DataFrame(summary_dict)
    combined_df = pandas.concat([summary_df, existing_df])
    combined_df.to_csv(climate_summary_path, index=False)


def add_current_climate():
    """Add summary of current climate conditions to the summary of future."""
    mongolia_shp_path = "E:/GIS_local/Mongolia/boundaries_etc/Mongolia.shp"
    steppe_shp_path = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop/southeastern_aimags_aoi_WGS84.shp"
    current_dir = "E:/GIS_local_archive/General_useful_data/Worldclim_2.0"
    summary_dict = {
        'model': [],
        'output': [],
        'aoi': [],
        'summary_method': [],
        'mean_value': [],
    }
    # tmin
    tmin_dir = os.path.join(current_dir, 'temperature_min')
    tmin_path_list = [
        os.path.join(tmin_dir, f) for f in os.listdir(tmin_dir)
        if f.endswith('.tif')]
    mean_val = average_raster_value_in_aoi(tmin_path_list, mongolia_shp_path)
    summary_dict['model'].append('current')
    summary_dict['output'].append('tmin')
    summary_dict['aoi'].append('Mongolia')
    summary_dict['summary_method'].append('Monthly average across pixels')
    summary_dict['mean_value'].append(mean_val)

    mean_val = average_raster_value_in_aoi(tmin_path_list, steppe_shp_path)
    summary_dict['model'].append('current')
    summary_dict['output'].append('tmin')
    summary_dict['aoi'].append('Eastern_steppe')
    summary_dict['summary_method'].append('Monthly average across pixels')
    summary_dict['mean_value'].append(mean_val)

    # tmax
    tmax_dir = os.path.join(current_dir, 'temperature_max')
    tmax_path_list = [
        os.path.join(tmax_dir, f) for f in os.listdir(tmax_dir)
        if f.endswith('.tif')]
    mean_val = average_raster_value_in_aoi(tmax_path_list, mongolia_shp_path)
    summary_dict['model'].append('current')
    summary_dict['output'].append('tmax')
    summary_dict['aoi'].append('Mongolia')
    summary_dict['summary_method'].append('Monthly average across pixels')
    summary_dict['mean_value'].append(mean_val)

    mean_val = average_raster_value_in_aoi(tmax_path_list, steppe_shp_path)
    summary_dict['model'].append('current')
    summary_dict['output'].append('tmax')
    summary_dict['aoi'].append('Eastern_steppe')
    summary_dict['summary_method'].append('Monthly average across pixels')
    summary_dict['mean_value'].append(mean_val)

    # prec
    prec_dir = os.path.join(current_dir, 'worldclim_precip')
    prec_path_list = [
        os.path.join(prec_dir, f) for f in os.listdir(prec_dir)
        if f.endswith('.tif')]
    mean_val = average_raster_value_in_aoi(prec_path_list, mongolia_shp_path)
    summary_dict['model'].append('current')
    summary_dict['output'].append('prec')
    summary_dict['aoi'].append('Mongolia')
    summary_dict['summary_method'].append('Monthly average across pixels')
    summary_dict['mean_value'].append(mean_val)

    mean_val = average_raster_value_in_aoi(prec_path_list, steppe_shp_path)
    summary_dict['model'].append('current')
    summary_dict['output'].append('prec')
    summary_dict['aoi'].append('Eastern_steppe')
    summary_dict['summary_method'].append('Monthly average across pixels')
    summary_dict['mean_value'].append(mean_val)

    # average annual precip
    sum_precip_path = os.path.join(current_dir, 'annual_precip.tif')
    # across pixels in Mongolia
    sum_zonal_stat = pygeoprocessing.zonal_statistics(
        (sum_precip_path, 1), mongolia_shp_path)
    mean_val = float(sum_zonal_stat[0]['sum']) / sum_zonal_stat[0]['count']
    summary_dict['model'].append('current')
    summary_dict['output'].append('prec')
    summary_dict['aoi'].append('Mongolia')
    summary_dict['summary_method'].append(
        'Average annual sum across pixels')
    summary_dict['mean_value'].append(mean_val)

    # across pixels in Eastern steppe
    sum_zonal_stat = pygeoprocessing.zonal_statistics(
        (sum_precip_path, 1), steppe_shp_path)
    mean_val = float(sum_zonal_stat[0]['sum']) / sum_zonal_stat[0]['count']
    summary_dict['model'].append('current')
    summary_dict['output'].append('prec')
    summary_dict['aoi'].append('Eastern_steppe')
    summary_dict['summary_method'].append(
        'Average annual sum across pixels')
    summary_dict['mean_value'].append(mean_val)

    current_df = pandas.DataFrame(summary_dict)
    future_df = pandas.read_csv("C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/climate/Worldclim_future_scenario_summary.csv")
    combined_df = pandas.concat([current_df, future_df])
    save_as = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/climate/Worldclim_current+future_scenario_summary.csv"
    combined_df.to_csv(save_as, index=False)


def time_series_at_points():
    """Make precip time series from future and current at a few points."""
    point_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/time_series_points.shp"
    precip_pattern_dict = {
        'current': "E:/GIS_local/Mongolia/Worldclim/Worldclim_baseline/wc2.0_30s_prec_2016_{}.tif",
        'CanESM5_ssp370_2061-2080': "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/CanESM5_ssp370_2061-2080/tmpjyovbsmf/precip_2016_{}.tif",
    }
    df_list = []
    for time_period in precip_pattern_dict:
        for m in range(1, 13):
            precip_path = precip_pattern_dict[time_period].format(m)
            precip_df = raster_values_at_points(
                point_shp_path, 'id', precip_path, 1, 'precip')
            precip_df['time_period'] = time_period
            precip_df['month'] = m
            df_list.append(precip_df)
    sum_df = pandas.concat(df_list)
    out_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/climate"
    save_as = os.path.join(out_dir, 'Worldclim_time_series_points.csv')
    sum_df.to_csv(save_as, index=False)


if __name__ == "__main__":
    summarize_future_climate()
    # add_current_climate()
    # time_series_at_points()
