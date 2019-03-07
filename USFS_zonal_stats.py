"""Miscellaneous small data processing steps for USFS Hemlock decline."""
import os
import numpy
from osgeo import gdal
import pygeoprocessing
import pandas

# nodata for precip scenario rasters
_TARGET_NODATA = -1


def calc_reference_evapotranspiration():
    """Use the modified Hargreaves equation to calc ET0."""
    def calc_ET0(RA, Tavg, TD, P):
        """Modified Hargreaves from Droogers and Allen 2002.

        Parameters:
            RA (numpy.ndarray): daily extraterrestrial radiation
            Tavg (numpy.ndarray): average temperature
            TD (numpy.ndarray): difference between minimum and maximum
                temperature
            P (numpy.ndarray): monthly precipitation

        Returns:
            monthly reference evapotranspiration (mm)

        """
        valid_mask = (
            (RA != _TARGET_NODATA) &
            (Tavg != tavg_nodata) &
            (TD != td_nodata) &
            (P != _TARGET_NODATA))
        result = numpy.empty(RA.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            0.0013 * 0.408 * RA[valid_mask] * (Tavg[valid_mask] + 17.) *
            (numpy.power((TD[valid_mask] - 0.0123 * P[valid_mask]), 0.76)) *
            29.5)
        return result
    workspace_dir = r"C:\Users\ginge\Documents\NatCap\GIS_local\USFS\NCEI_climate"
    RA_df = pandas.read_csv(os.path.join(workspace_dir, "RA_FAO.csv"))
    Tavg_dir = os.path.join(workspace_dir, "temp_ave_rasters")
    TD_dir = os.path.join(workspace_dir, "tdiff_rasters")
    for scenario in ['high_precip_scenario', 'low_precip_scenario']:
        output_dir = os.path.join(
            r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\input_data\model_inputs",
            scenario)
        ET0_dir = os.path.join(output_dir, 'ET0')
        P_dir = os.path.join(output_dir, "precip_rasters")
    # # testing
    # ET0_dir = os.path.join(workspace_dir, 'ET0_test')
    # P_dir = os.path.join(workspace_dir, 'precip_rasters')
        for month in xrange(1, 13):
            Tavg_raster_path = os.path.join(
                Tavg_dir, 'tavg_{}.tif'.format(month))
            TD_raster_path = os.path.join(
                TD_dir, 'tdiff_{}.tif'.format(month))
            P_raster_path = os.path.join(
                P_dir, 'precip_{}.tif'.format(month))
            tavg_nodata = pygeoprocessing.get_raster_info(
                Tavg_raster_path)['nodata'][0]
            td_nodata = pygeoprocessing.get_raster_info(
                TD_raster_path)['nodata'][0]
            RA_val = (
                RA_df.loc[RA_df['month'] == month, 'average_RA'].values[0])
            RA_raster_path = os.path.join(workspace_dir, 'RA.tif')
            pygeoprocessing.new_raster_from_base(
                P_raster_path, RA_raster_path, gdal.GDT_Float32,
                [_TARGET_NODATA], fill_value_list=[RA_val])
            ET0_raster_path = os.path.join(ET0_dir, 'ET_{}.tif'.format(month))
            pygeoprocessing.raster_calculator(
                [(path, 1) for path in [
                    RA_raster_path, Tavg_raster_path, TD_raster_path,
                    P_raster_path]],
                calc_ET0, ET0_raster_path, gdal.GDT_Float32, _TARGET_NODATA)


def generate_scenario_precip_rasters():
    """Generate precipitation rasters for low- and high-precip scenarios."""
    def down_to_50_perc(precip_raster):
        valid_mask = (precip_raster != precip_raster_nodata)
        result = numpy.empty(precip_raster.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = precip_raster[valid_mask] * 0.5
        return result

    def up_to_150_perc(precip_raster):
        valid_mask = (precip_raster != precip_raster_nodata)
        result = numpy.empty(precip_raster.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = precip_raster[valid_mask] * 1.5
        return result

    baseline_precip_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\input_data\model_inputs\precip_NCEI"
    high_precip_scenario_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\input_data\model_inputs\high_precip_scenario\precip_rasters"
    low_precip_scenario_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\input_data\model_inputs\low_precip_scenario\precip_rasters"

    example_precip_raster = os.path.join(
        baseline_precip_dir, 'precip_1.tif')
    precip_raster_nodata = pygeoprocessing.get_raster_info(
        example_precip_raster)['nodata'][0]
    for m in xrange(1, 13):
        baseline_raster_path = os.path.join(
            baseline_precip_dir, 'precip_{}.tif'.format(m))
        decreased_raster_path = os.path.join(
            low_precip_scenario_dir, 'precip_{}.tif'.format(m))
        pygeoprocessing.raster_calculator(
            [(baseline_raster_path, 1)], down_to_50_perc, decreased_raster_path,
            gdal.GDT_Float32, _TARGET_NODATA)

        increased_raster_path = os.path.join(
            high_precip_scenario_dir, 'precip_{}.tif'.format(m))
        pygeoprocessing.raster_calculator(
            [(baseline_raster_path, 1)], up_to_150_perc, increased_raster_path,
            gdal.GDT_Float32, _TARGET_NODATA)


def zonal_stats_workflow():
    """Calculate monthly quickflow sum within aoi."""
    save_as = "C:/Users/ginge/Documents/NatCap/GIS_local/USFS/replicate_4th_draft_12.4.18/summary/monthly_quickflow.csv"
    scenario_dict = {
        'pre-decline': "C:/Users/ginge/Documents/NatCap/GIS_local/USFS/replicate_4th_draft_12.4.18/pre_decline",
        'post-decline': "C:/Users/ginge/Documents/NatCap/GIS_local/USFS/replicate_4th_draft_12.4.18/post_decline",
    }
    df_list = []
    for scenario in scenario_dict.iterkeys():
        results_dict = {
            'scenario': [],
            'month': [],
            'sum_quickflow': [],
        }
        folder = scenario_dict[scenario]
        aoi_shp = os.path.join(folder, 'aggregated_results.shp')
        for month in xrange(1, 13):
            qf_raster = os.path.join(
                folder, 'intermediate_outputs', 'qf_{}.tif'.format(month))
            zonal_stats = pygeoprocessing.zonal_statistics(
                (qf_raster, 1), aoi_shp)
            sum_QF = zonal_stats[0]['sum']
            results_dict['scenario'].append(scenario)
            results_dict['month'].append(month)
            results_dict['sum_quickflow'].append(sum_QF)
        results_df = pandas.DataFrame(data=results_dict)
        df_list.append(results_df)
    combined_list = pandas.concat(df_list)
    combined_list.to_csv(save_as, index=False)


if __name__ == '__main__':
    calc_reference_evapotranspiration()
