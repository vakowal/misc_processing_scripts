# coding=UTF-8
# -----------------------------------------------
# Generated by InVEST 3.9.0 on Thu Jan 21 16:34:22 2021
# Model: Sediment Delivery Ratio Model (SDR)

import os
import logging
import sys
import re
import tempfile

import numpy
import pandas
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import natcap.invest.sdr.sdr
import natcap.invest.seasonal_water_yield.seasonal_water_yield
import natcap.invest.utils
import pygeoprocessing

TARGET_NODATA = -1

LOGGER = logging.getLogger(__name__)
root_logger = logging.getLogger()

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt=natcap.invest.utils.LOG_FMT,
    datefmt='%m/%d/%Y %H:%M:%S ')
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])

# common data inputs used for both SDR and SWY
_AOI = 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/projected/Chaglla_dam_watershed.shp'
_DEM = 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/projected/HydroSHEDS_CON_Chaglla_UTM18S.tif'
_TFA_VAL = '1000'
_CURRENT_LULC = 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/SEALS_lulc/lulc_current.tif'
_LULC_PATTERN = 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/SEALS_lulc/lulc_RCP{}_year{}.tif'

# biophysical table that includes CN-III values
_BIOPHYS_CNIII = 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/biophysical_table_Chaglla_SEALS_simplified.csv'
# biophysical table that contains CN-II values, or "default" CN
_BIOPHYS_CNII = 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/biophysical_table_Chaglla_SEALS_simplified_CNII.csv'


def outputs_to_dict(watershed_results_path, field_list):
    """Read results from watershed results and return as a dict."""
    field_list.append('WS_ID')
    model_dict = {field: [] for field in field_list}
    ws_vector = ogr.Open(watershed_results_path)
    ws_layer = ws_vector.GetLayer()
    for ws_feature in ws_layer:
        for field_name in model_dict:
            model_dict[field_name].append(ws_feature.GetField(field_name))

    ws_layer = None
    ws_layer = None
    return model_dict


def convert_flow_outputs(swy_args):
    """Convert flow outputs from mm per pixel per year to m3/sec."""
    def mm_per_year_to_m3_per_sec(mm_raster, pixel_area_m):
        """Calculate average m3 per second from mm per year.

        Use a static multiplier to convert mm/year to meters per second, then
        multiply that by pixel area to calculate cubic meters per second.

        Parameters:
            mm_raster (numpy.ndarray): raster whose values are in units of mm per
                pixel per year
            pixel_area_m (float): pixel area in square meters

        Returns:
            raster with values in cubic meter per second

        """
        valid_mask = (mm_raster != input_nodata)
        result = numpy.empty(mm_raster.shape, dtype=numpy.float32)
        result[:] = input_nodata
        result[valid_mask] = (
            mm_raster[valid_mask] * 3.17098E-11 * pixel_area_m)
        return result

    output_pixel_area = pygeoprocessing.get_raster_info(
        swy_args['dem_raster_path'])['pixel_size'][0]**2

    baseflow_mm_per_year_path = os.path.join(
        swy_args['workspace_dir'], 'B.tif')
    input_nodata = pygeoprocessing.get_raster_info(
        baseflow_mm_per_year_path)['nodata'][0]
    baseflow_m3_per_sec_path = os.path.join(
        swy_args['workspace_dir'], 'B_m3_per_sec.tif')
    pygeoprocessing.raster_calculator(
        [(baseflow_mm_per_year_path, 1), (output_pixel_area, 'raw')],
        mm_per_year_to_m3_per_sec, baseflow_m3_per_sec_path,
        gdal.GDT_Float32, input_nodata)

    quickflow_mm_per_year_path = os.path.join(
        swy_args['workspace_dir'], 'QF.tif')
    input_nodata = pygeoprocessing.get_raster_info(
        quickflow_mm_per_year_path)['nodata'][0]
    quickflow_m3_per_sec_path = os.path.join(
        swy_args['workspace_dir'], 'QF_m3_per_sec.tif')
    pygeoprocessing.raster_calculator(
        [(quickflow_mm_per_year_path, 1), (output_pixel_area, 'raw')],
        mm_per_year_to_m3_per_sec, quickflow_m3_per_sec_path,
        gdal.GDT_Float32, input_nodata)


def swy_runs():
    """Run SWY for current and future scenarios."""
    args = {
        'alpha_m': '1/12',
        'aoi_path': _AOI,
        'beta_i': '1',
        'dem_raster_path': _DEM,
        'gamma': '1',
        'monthly_alpha': False,
        'results_suffix': '',
        'soil_group_path': 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/projected/hydrologic_soil_group.tif',
        'threshold_flow_accumulation': _TFA_VAL,
        'user_defined_climate_zones': False,
        'user_defined_local_recharge': False,
    }
    result_dir = 'C:/SWY_workspace'
    processing_dir = os.path.join(result_dir, 'SWY_intermediate')
    if not os.path.exists(processing_dir):
        os.makedirs(processing_dir)
    precip_dir_pattern = 'F:/Moore_Amazon_backups/precipitation/year_{}/rcp_{}'
    et0_dir_pattern = 'F:/Moore_Amazon_backups/ET0/year_{}/rcp_{}'
    rain_events_table_pattern = "F:/Moore_Amazon_backups/rain_events/year_{}_rcp_{}.csv"

    df_list = []
    for cn_option in ['CN-II', 'CN-III']:
        if cn_option == 'CN-II':
            args['biophysical_table_path'] = _BIOPHYS_CNII
        else:
            args['biophysical_table_path'] = _BIOPHYS_CNIII

        # current
        args['workspace_dir'] = os.path.join(result_dir, cn_option, 'current')
        args['lulc_raster_path'] = _CURRENT_LULC
        args['precip_dir'] = 'F:/Moore_Amazon_backups/precipitation/current'
        args['et0_dir'] = 'F:/Moore_Amazon_backups/ET0/current'
        args['rain_events_table_path'] = 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/rain_events_Chaglla_centroid_iwmi_water_atlas.csv'
        watershed_results_path = os.path.join(
            args['workspace_dir'], 'aggregated_results_swy.shp')
        if not os.path.isfile(watershed_results_path):
            natcap.invest.seasonal_water_yield.seasonal_water_yield.execute(args)
        baseflow_df = attribute_baseflow_by_precip(args)
        quickflow_df = monthly_quickflow_sum(args)
        combined_df = baseflow_df.merge(
            quickflow_df, on='month', suffixes=(False, False))
        combined_df['scenario'] = 'current'
        combined_df['year'] = 2015
        combined_df['CN_option'] = cn_option
        df_list.append(combined_df)

        # future
        for year in ['2050', '2070']:
            for rcp in ['2.6', '6.0', '8.5']:
                args['workspace_dir'] = os.path.join(
                    result_dir, cn_option, 'year_{}'.format(year),
                    'rcp_{}'.format(rcp))
                args['lulc_path'] = _LULC_PATTERN.format(rcp, year)
                args['precip_dir'] = precip_dir_pattern.format(year, rcp)
                args['et0_dir'] = et0_dir_pattern.format(year, rcp)
                args['rain_events_table_path'] = rain_events_table_pattern.format(
                    year, rcp)
                watershed_results_path = os.path.join(
                    args['workspace_dir'], 'aggregated_results_swy.shp')
                if not os.path.isfile(watershed_results_path):
                    natcap.invest.seasonal_water_yield.seasonal_water_yield.execute(args)
                baseflow_df = attribute_baseflow_by_precip(args)
                quickflow_df = monthly_quickflow_sum(args)
                combined_df = baseflow_df.merge(
                    quickflow_df, on='month', suffixes=(False, False))
                combined_df['scenario'] = rcp
                combined_df['year'] = year
                combined_df['CN_option'] = cn_option
                df_list.append(combined_df)

    summary_df = pandas.concat(df_list)
    summary_df.to_csv(
        os.path.join(result_dir, 'seasonal_streamflow_summary.csv'),
        index=False)


def monthly_quickflow_sum(swy_args):
    """Collect the mean of monthly quickflow values."""
    def mm_per_month_to_m3_per_sec(mm_raster):
        """Calculate average m3 per second from mm per month.

        Use a static multiplier to convert mm/month to meters per second, then
        multiply that by pixel area to calculate cubic meters per second.

        Parameters:
            mm_raster (numpy.ndarray): raster whose values are in units of mm
                per pixel per month

        Returns:
            raster with values in cubic meter per second

        """
        valid_mask = (mm_raster != input_nodata)
        result = numpy.empty(mm_raster.shape, dtype=numpy.float32)
        result[:] = input_nodata
        result[valid_mask] = (
            mm_raster[valid_mask] * 3.81E-10 * pixel_area_m)
        return result

    quickflow_dict = {'month': [], 'quickflow': []}
    for month_index in range(1, 13):
        qf_path = os.path.join(
            swy_args['workspace_dir'], 'intermediate_outputs',
            'qf_{}.tif'.format(month_index))
        pixel_area_m = pygeoprocessing.get_raster_info(
            qf_path)['pixel_size'][0]**2
        input_nodata = pygeoprocessing.get_raster_info(qf_path)['nodata'][0]
        with tempfile.NamedTemporaryFile(
                prefix='quickflow_m3', delete=False, suffix='.tif') as qf_file:
            qf_m3_per_sec_path = qf_file.name
        pygeoprocessing.raster_calculator(
            [(qf_path, 1)], mm_per_month_to_m3_per_sec, qf_m3_per_sec_path,
            gdal.GDT_Float32, input_nodata)
        qf_sum = pygeoprocessing.zonal_statistics(
            (qf_m3_per_sec_path, 1), swy_args['aoi_path'])[0]['sum']
        quickflow_dict['month'].append(month_index)
        quickflow_dict['quickflow'].append(qf_sum)
    quickflow_df = pandas.DataFrame(quickflow_dict)
    return quickflow_df


def attribute_baseflow_by_precip(swy_args):
    """Attribute annual baseflow according to distribution of precip.

    Following the method in Hamel et al 2020, "Modeling seasonal water yield
    for landscape management: Applications in Peru and Myanmar", Journal of
    Environmental Management.
    calculate mean annual baseflow across pixels.
    summarize precipitation inside the watershed: monthly sum across pixels,
    and annual sum across months. Apply these relative values to the annual
    baseflow number, shifted by one month. I.e. baseflow in February is
    proportional to annual baseflow as precipitation in January is proportional
    to annual precipitation.

    Args:
        swy_args (dict): dictionary of inputs for Seasonal Water Yield

    Returns:
        pandas data frame containing two columns, month and baseflow arriving
            at outlet in that month

    """
    def mm_per_year_to_m3_per_sec(mm_raster):
        """Calculate average m3 per second from mm per year.

        Use a static multiplier to convert mm/year to meters per second, then
        multiply that by pixel area to calculate cubic meters per second.

        Parameters:
            mm_raster (numpy.ndarray): raster whose values are in units of mm
                per pixel per year

        Returns:
            raster with values in cubic meter per second

        """
        valid_mask = (mm_raster != input_nodata)
        result = numpy.empty(mm_raster.shape, dtype=numpy.float32)
        result[:] = input_nodata
        result[valid_mask] = (
            mm_raster[valid_mask] * 3.17098E-11 * pixel_area_m)
        return result

    # convert annual baseflow to m3/sec and calculate watershed sum
    baseflow_path = os.path.join(swy_args['workspace_dir'], 'B.tif')
    pixel_area_m = pygeoprocessing.get_raster_info(
        baseflow_path)['pixel_size'][0]**2
    input_nodata = pygeoprocessing.get_raster_info(baseflow_path)['nodata'][0]
    with tempfile.NamedTemporaryFile(
            prefix='baseflow_m3', delete=False,
            suffix='.tif') as baseflow_file:
        baseflow_m3_per_sec_path = baseflow_file.name
    pygeoprocessing.raster_calculator(
        [(baseflow_path, 1)],mm_per_year_to_m3_per_sec,
        baseflow_m3_per_sec_path, gdal.GDT_Float32, input_nodata)
    baseflow_sum = pygeoprocessing.zonal_statistics(
        (baseflow_m3_per_sec_path, 1), swy_args['aoi_path'])[0]['sum']

    # find precip rasters and calculate sum of precip in watershed
    precip_dict = {'month_p': [], 'precip': []}
    precip_dir_list = [
        os.path.join(swy_args['precip_dir'], f) for f in os.listdir(
            swy_args['precip_dir'])]
    for month_index in range(1, 13):
        month_file_match = re.compile(r'.*[^\d]%d\.[^.]+$' % month_index)
        file_list = [
            month_file_path for month_file_path in precip_dir_list
            if month_file_match.match(month_file_path)]
        if len(file_list) > 1:
            raise ValueError(
                "Ambiguous set of files found for month %d: %s" %
                (month_index, file_list))
        monthly_precip = pygeoprocessing.zonal_statistics(
            (file_list[0], 1), swy_args['aoi_path'])[0]['sum']
        precip_dict['month_p'].append(month_index)
        precip_dict['precip'].append(monthly_precip)
    precip_df = pandas.DataFrame(precip_dict)
    precip_df['proportion'] = precip_df['precip'] / sum(precip_df['precip'])
    precip_df['month'] = precip_df['month_p'] + 1
    precip_df.loc[precip_df['month_p'] == 12, 'month'] = 1
    precip_df['baseflow'] = baseflow_sum * precip_df['proportion']
    baseflow_df = precip_df[['month', 'baseflow']]
    return baseflow_df


def sdr_runs():
    """Run SDR for current and future scenarios."""
    # outer_dir = 'C:/Users/ginge/Documents/NatCap/GIS_local/Moore_Amazon'
    # result_dir = os.path.join(outer_dir, 'SDR_workspace')
    args = {
        'dem_path': _DEM,
        'drainage_path': '',
        'erodibility_path': 'C:/Users/ginge/Dropbox/NatCap_backup/Moore_Amazon/SDR_SWY_data_inputs/projected/erodibility_ISRIC_30arcseconds.tif',
        'ic_0_param': '0.5',
        'k_param': '2',
        'results_suffix': '',
        'sdr_max': '0.8',
        'threshold_flow_accumulation': _TFA_VAL,
        'watersheds_path': _AOI,
        'biophysical_table_path': _BIOPHYS_CNIII,
    }

    result_dir = 'C:/SDR_workspace'
    sdr_fields = ['usle_tot', 'sed_export', 'sed_retent', 'sed_dep']
    erosivity_pattern = "F:/Moore_Amazon_backups/precipitation/erosivity_Riquetti/erosivity_year{}_rcp{}.tif"

    df_list = []

    # SEALS landcover, "Riquetti" erosivity
    args['workspace_dir'] = os.path.join(result_dir, 'current')
    args['lulc_path'] = _CURRENT_LULC
    args['erosivity_path'] = 'F:/Moore_Amazon_backups/precipitation/erosivity_Riquetti/erosivity_current.tif'
    # natcap.invest.sdr.sdr.execute(args)

    sdr_dict = outputs_to_dict(
        os.path.join(args['workspace_dir'], 'watershed_results_sdr.shp'),
        sdr_fields)
    sdr_dict['year'] = ['2015' * len(sdr_dict['WS_ID'])]
    sdr_dict['scenario'] = ['current' * len(sdr_dict['WS_ID'])]
    sdr_dict['erosivity'] = ['Riquetti_current' * len(sdr_dict['WS_ID'])]
    sdr_df = pandas.DataFrame(sdr_dict)
    df_list.append(sdr_df)

    for year in ['2050', '2070']:
        for rcp in ['2.6', '6.0', '8.5']:
            # lulc only reflecting future conditions
            args['workspace_dir'] = os.path.join(
                result_dir, 'year_{}'.format(year), 'rcp_{}'.format(rcp),
                'current_erosivity')
            args['lulc_path'] = _LULC_PATTERN.format(rcp, year)
            natcap.invest.sdr.sdr.execute(args)

            sdr_dict = outputs_to_dict(
                os.path.join(args['workspace_dir'],
                'watershed_results_sdr.shp'), sdr_fields)
            sdr_dict['year'] = [year * len(sdr_dict['WS_ID'])]
            sdr_dict['scenario'] = [rcp * len(sdr_dict['WS_ID'])]
            sdr_dict['erosivity'] = [
                'current' * len(sdr_dict['WS_ID'])]
            sdr_df = pandas.DataFrame(sdr_dict)
            df_list.append(sdr_df)

            # lulc and erosivity reflecting future conditions
            args['workspace_dir'] = os.path.join(
                result_dir, 'year_{}'.format(year), 'rcp_{}'.format(rcp))
            args['erosivity_path'] = erosivity_pattern.format(year[2:], rcp)
            # natcap.invest.sdr.sdr.execute(args)

            sdr_dict = outputs_to_dict(
                os.path.join(args['workspace_dir'],
                'watershed_results_sdr.shp'), sdr_fields)
            sdr_dict['year'] = [year * len(sdr_dict['WS_ID'])]
            sdr_dict['scenario'] = [rcp * len(sdr_dict['WS_ID'])]
            sdr_dict['erosivity'] = [
                'Riquetti_future_precip' * len(sdr_dict['WS_ID'])]
            sdr_df = pandas.DataFrame(sdr_dict)
            df_list.append(sdr_df)

    summary_df = pandas.concat(df_list)
    summary_df.to_csv(
        os.path.join(result_dir, 'watershed_results_summary.csv'), index=False)


def inspect_precip():
    """make a table of precip in the watershed from scenario inputs."""
    def get_precip_df(precip_dir, aoi_path):
        """Get sum of monthly precip inside the watershed for each month"""
        precip_dict = {'month_p': [], 'precip': []}
        precip_dir_list = [
            os.path.join(precip_dir, f) for f in os.listdir(precip_dir)]
        for month_index in range(1, 13):
            month_file_match = re.compile(r'.*[^\d]%d\.[^.]+$' % month_index)
            file_list = [
                month_file_path for month_file_path in precip_dir_list
                if month_file_match.match(month_file_path)]
            if len(file_list) > 1:
                raise ValueError(
                    "Ambiguous set of files found for month %d: %s" %
                    (month_index, file_list))
            monthly_precip = pygeoprocessing.zonal_statistics(
                (file_list[0], 1), aoi_path)[0]['sum']
            precip_dict['month_p'].append(month_index)
            precip_dict['precip'].append(monthly_precip)
        precip_df = pandas.DataFrame(precip_dict)
        return precip_df

    result_dir = 'C:/SWY_workspace'
    precip_dir_pattern = 'F:/Moore_Amazon_backups/precipitation/year_{}/rcp_{}'

    df_list = []
    # current
    precip_df = get_precip_df(
        'F:/Moore_Amazon_backups/precipitation/current', _AOI)
    precip_df['scenario'] = 'current'
    precip_df['year'] = 2015
    df_list.append(precip_df)

    for year in ['2050', '2070']:
        for rcp in ['2.6', '6.0', '8.5']:
            precip_df = get_precip_df(
                precip_dir_pattern.format(year, rcp), _AOI)
            precip_df['year'] = year
            precip_df['scenario'] = rcp
            df_list.append(precip_df)
    summary_df = pandas.concat(df_list)
    summary_df.to_csv(
        os.path.join(result_dir, 'precip_summar.csv'), index=False)


if __name__ == '__main__':
    # sdr_runs()
    swy_runs()
    # inspect_precip()
