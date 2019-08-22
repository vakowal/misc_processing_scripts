"""Miscellaneous small data processing steps for USFS Hemlock decline."""
import os
import numpy
from osgeo import gdal
import pygeoprocessing
import pandas

# nodata for precip scenario rasters
_TARGET_NODATA = -1


def generate_scenario_precip_inputs():
    """Generate precipitation rasters for low- and high-precip scenarios."""
    def multiply_precip(precip_raster, multiply_factor):
        valid_mask = (precip_raster != precip_raster_nodata)
        result = numpy.empty(precip_raster.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = precip_raster[valid_mask] * multiply_factor
        return result

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

    climate_dir = r"E:\GIS_local_3.6.19\USFS\NCEI_climate"
    RA_df = pandas.read_csv(os.path.join(climate_dir, "RA_FAO.csv"))
    Tavg_dir = os.path.join(climate_dir, "temp_ave_rasters")
    TD_dir = os.path.join(climate_dir, "tdiff_rasters")
    events_table = os.path.join(
        climate_dir, 'rain_events_table_Palmer_creek.csv')

    baseline_precip_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\input_data\model_inputs\precip_NCEI"
    outer_scenario_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\input_data\model_inputs\precip_scenarios"

    example_precip_raster = os.path.join(
        baseline_precip_dir, 'precip_1.tif')
    precip_raster_nodata = pygeoprocessing.get_raster_info(
        example_precip_raster)['nodata'][0]

    for multiply_factor in [0.5]:  # [0.7, 0.9, 1.1, 1.3]:
        scenario_dir = os.path.join(
            outer_scenario_dir, '{}x'.format(multiply_factor))
        modified_event_table = os.path.join(scenario_dir, 'rain_events.csv')
        ET0_dir = os.path.join(scenario_dir, 'ET0_rasters')
        precip_dir = os.path.join(scenario_dir, 'precip_rasters')
        if not os.path.exists(ET0_dir):
            os.makedirs(ET0_dir)
        if not os.path.exists(precip_dir):
            os.makedirs(precip_dir)
        event_df = pandas.read_csv(events_table)
        event_df.events = event_df.events * multiply_factor
        event_df.to_csv(modified_event_table)
        for month in xrange(1, 13):
            baseline_raster_path = os.path.join(
                baseline_precip_dir, 'precip_{}.tif'.format(month))
            modified_precip_path = os.path.join(
                precip_dir, 'precip_{}.tif'.format(month))
            pygeoprocessing.raster_calculator(
                [(baseline_raster_path, 1), (multiply_factor, 'raw')],
                multiply_precip, modified_precip_path,
                gdal.GDT_Float32, _TARGET_NODATA)

            Tavg_raster_path = os.path.join(
                Tavg_dir, 'tavg_{}.tif'.format(month))
            TD_raster_path = os.path.join(
                TD_dir, 'tdiff_{}.tif'.format(month))
            tavg_nodata = pygeoprocessing.get_raster_info(
                Tavg_raster_path)['nodata'][0]
            td_nodata = pygeoprocessing.get_raster_info(
                TD_raster_path)['nodata'][0]
            RA_val = (
                RA_df.loc[RA_df['month'] == month, 'average_RA'].values[0])
            RA_raster_path = os.path.join(climate_dir, 'RA.tif')
            pygeoprocessing.new_raster_from_base(
                modified_precip_path, RA_raster_path, gdal.GDT_Float32,
                [_TARGET_NODATA], fill_value_list=[RA_val])
            ET0_raster_path = os.path.join(ET0_dir, 'ET_{}.tif'.format(month))
            pygeoprocessing.raster_calculator(
                [(path, 1) for path in [
                    RA_raster_path, Tavg_raster_path, TD_raster_path,
                    modified_precip_path]],
                calc_ET0, ET0_raster_path, gdal.GDT_Float32, _TARGET_NODATA)


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


def run_precip_scenarios(save_as):
    """Run the seasonal water yield model under a range of total precip."""
    import natcap.invest.seasonal_water_yield.seasonal_water_yield

    outer_scenario_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\input_data\model_inputs\precip_scenarios"
    outer_results_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\model_runs\precip_scenarios"

    outer_workspace_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\model_runs"
    pre_decline_lulc = "C:/Users/ginge/Dropbox/NatCap_backup/USFS/input_data/model_inputs/Madden_overstory_proj_filled_lucode.tif"
    post_decline_lulc = "C:/Users/ginge/Dropbox/NatCap_backup/USFS/input_data/model_inputs/post_decline_lulc.tif"

    model_args = {
        u'alpha_m': u'1/12',
        u'aoi_path': u'C:/Users/ginge/Dropbox/NatCap_backup/USFS/input_data/model_inputs/Palmer_creek_HUC12_proj.shp',
        u'beta_i': u'1',
        u'biophysical_table_path': u'C:/Users/ginge/Dropbox/NatCap_backup/USFS/model_runs/fourth_draft/biophysical_table.csv',
        u'dem_raster_path': u'C:/Users/ginge/Dropbox/NatCap_backup/USFS/input_data/model_inputs/DEM_1_arc_sec_proj_fill.tif',
        u'gamma': u'1',
        u'monthly_alpha': False,
        u'results_suffix': u'',
        u'soil_group_path': u'C:/Users/ginge/Dropbox/NatCap_backup/USFS/input_data/model_inputs/hydro_soils.tif',
        u'threshold_flow_accumulation': u'90',
        u'user_defined_climate_zones': False,
        u'user_defined_local_recharge': False,
    }
    summary_dict = {
        'precip_multiply_factor': [],
        'lulc_scenario': [],
        'sum_B': [],
        'sum_aet': [],
        'sum_QF': [],
    }
    for multiply_factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        scenario_dir = os.path.join(
            outer_scenario_dir, '{}x'.format(multiply_factor))
        results_dir = os.path.join(
            outer_results_dir, '{}x'.format(multiply_factor))
        predecline_results_dir = os.path.join(results_dir, 'pre_decline')
        postdecline_results_dir = os.path.join(results_dir, 'post_decline')
        if not os.path.exists(results_dir):
            model_args['rain_events_table_path'] = os.path.join(
                scenario_dir, 'rain_events.csv')
            model_args['et0_dir'] = os.path.join(scenario_dir, 'ET0_rasters')
            model_args['precip_dir'] = os.path.join(
                scenario_dir, 'precip_rasters')

            # pre-decline run
            model_args['lulc_raster_path'] = pre_decline_lulc
            model_args['workspace_dir'] = predecline_results_dir
            print "run me"
            # natcap.invest.seasonal_water_yield.seasonal_water_yield.execute(
            #     model_args)

            # post-decline run
            model_args['lulc_raster_path'] = post_decline_lulc
            model_args['workspace_dir'] = postdecline_results_dir
            print "run me"
            # natcap.invest.seasonal_water_yield.seasonal_water_yield.execute(
            #     model_args)

        # collect zonal stats: pre-decline
        summary_dict['precip_multiply_factor'].append(multiply_factor)
        summary_dict['lulc_scenario'].append('pre-decline')
        b_raster = os.path.join(predecline_results_dir, 'B.tif')
        summary_dict['sum_B'].append(
            pygeoprocessing.zonal_statistics(
                (b_raster, 1), model_args['aoi_path'])[0]['sum'])
        qf_raster = os.path.join(predecline_results_dir, 'QF.tif')
        summary_dict['sum_QF'].append(
            pygeoprocessing.zonal_statistics(
                (qf_raster, 1), model_args['aoi_path'])[0]['sum'])
        aet_raster = os.path.join(
            predecline_results_dir, 'intermediate_outputs', 'aet.tif')
        summary_dict['sum_aet'].append(
            pygeoprocessing.zonal_statistics(
                (aet_raster, 1), model_args['aoi_path'])[0]['sum'])

        # collect zonal stats: post-decline
        summary_dict['precip_multiply_factor'].append(multiply_factor)
        summary_dict['lulc_scenario'].append('post-decline')
        b_raster = os.path.join(postdecline_results_dir, 'B.tif')
        summary_dict['sum_B'].append(
            pygeoprocessing.zonal_statistics(
                (b_raster, 1), model_args['aoi_path'])[0]['sum'])
        qf_raster = os.path.join(postdecline_results_dir, 'QF.tif')
        summary_dict['sum_QF'].append(
            pygeoprocessing.zonal_statistics(
                (qf_raster, 1), model_args['aoi_path'])[0]['sum'])
        aet_raster = os.path.join(
            postdecline_results_dir, 'intermediate_outputs', 'aet.tif')
        summary_dict['sum_aet'].append(
            pygeoprocessing.zonal_statistics(
                (aet_raster, 1), model_args['aoi_path'])[0]['sum'])
    summary_df = pandas.DataFrame.from_dict(summary_dict)
    summary_df.to_csv(save_as)


def diff_l_sum_avail():
    """Calculate the pixel-level difference in L_sum_avail between pre- and
    post-decline landscapes, for each precip scenario."""
    def subtract(raster1, raster2):
        """Subtract raster2 from raster1."""
        valid_mask = (
            (raster1 != nodata) &
            (raster2 != nodata))
        result = numpy.empty(raster1.shape, dtype=numpy.float32)
        result[:] = nodata
        result[valid_mask] = raster1[valid_mask] - raster2[valid_mask]
        return result

    outer_results_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\model_runs\precip_scenarios"
    example_raster = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\model_runs\precip_scenarios\0.5x\post_decline\L_sum_avail.tif"
    nodata = pygeoprocessing.get_raster_info(example_raster)['nodata'][0]
    for multiply_factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        results_dir = os.path.join(
            outer_results_dir, '{}x'.format(multiply_factor))
        predecline_lavail_sum = os.path.join(
            results_dir, 'pre_decline', 'L_sum_avail.tif')
        postdecline_lavail_sum = os.path.join(
            results_dir, 'post_decline', 'L_sum_avail.tif')
        target_path = os.path.join(
            results_dir, "L_sum_avail_post_minus_pre.tif")
        pygeoprocessing.raster_calculator(
            [(postdecline_lavail_sum, 1), (predecline_lavail_sum, 1)],
            subtract, target_path, gdal.GDT_Float32, nodata)


if __name__ == '__main__':
    # generate_scenario_precip_inputs()
    test_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\USFS\model_runs\precip_scenarios\precip_scenario_summary.csv"
    # run_precip_scenarios(test_csv)
    diff_l_sum_avail()
