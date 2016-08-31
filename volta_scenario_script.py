""""
Volta SDR: generate scenarios, run model, collect results.
"""

import os
import natcap.invest.scenario_gen_proximity
import natcap.invest.sdr
from osgeo import gdal
from osgeo import ogr
import pygeoprocessing.geoprocessing
import pandas
from arcpy import *

def merge_rasters(rasters_to_merge, save_as):
    """Mosaic positive values from several rasters of the same shape together.
    If positive regions overlap, those later in the list will cover those
    earlier so that the last raster in the list will not be covered.  Saves the
    merged raster as save_as and returns nothing."""

    def merge_op(*rasters):
        raster_list = list(rasters)
        # assume rasters share size
        result_raster = np.full((raster_list[0].shape), -9999)
        # assume nodata (should not be copied) is 0
        for raster in raster_list:
            np.copyto(result_raster, raster, where=raster > 0)
        return result_raster
    
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            rasters_to_merge[0])
    pygeoprocessing.geoprocessing.vectorize_datasets(
            rasters_to_merge, merge_op, save_as,
            gdal.GDT_Int32, -9999, out_pixel_size, "union",
            dataset_to_align_index=0, vectorize_op=False)

def generate_buf_scenario(lulc_path, buffer_int, buffer_area,
                          percent_to_convert):
    """Main function to generate the flexible buffer area scenario.  Percent
    to convert is the percent of full buffer area that should be converted to
    riparian buffer."""
    
    # inputs will be:
    # - your lulc with the full buffer coded as the integer buffer_int
    # - percent of the buffer area to convert
    # - buffer_area is the area of the buffer in ha.  We should calculate this
    # inside the script but I don't know how to, so for now we'll calculate by
    # hand outside the script!
    
    # stream cell just inside reservoir inlet was reclassified
    # as the integer 'focal_landcover' (1000)
    
    # step 1: calculate area of full buffer
    # cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(lulc_path)
    # cell_area = cell_size * cell_size
    full_buffer_area = buffer_area  # calculate in hectares
    convertible_area = full_buffer_area * percent_to_convert
    
    replacement_lucode = 1001  # arbitrary integer to identify 'filled' buffer
    # (make sure this is included in the biophysical table)
    convertible_landcover = buffer_int  # integer that identifies the full buffer in lulc
    
    # ASHLEY change these filepaths to point to your inputs
    # and check all other inputs
    scen_gen_args = {
            u'aoi_uri': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/SDR_workspace_7.2.16/watershed_results_sdr_default.shp',
            u'area_to_convert': str(convertible_area),
            u'base_lulc_uri': lulc_path,
            u'convert_farthest_from_edge': False,
            u'convert_nearest_to_edge': True,
            u'convertible_landcover_codes': str(convertible_landcover),
            u'focal_landcover_codes': u'1000',
            u'n_fragmentation_steps': u'1',
            u'replacment_lucode': str(replacement_lucode),
            u'workspace_dir': u'C:\\Users\\Ginger/Documents/scenario_proximity_workspace',
    }
    natcap.invest.scenario_gen_proximity.execute(scen_gen_args)
    # TODO return result of scenario generator

def generate_scenarios_from_table(scenario_csv, data_dir):
    """Put pieces together to create lulc scenarios."""
    
    scenario_dict = {'scenario': [], 'scen_name': [], 'lulc_raster': []}
    scen_df = pandas.read_csv(scenario_csv)
    for row in xrange(len(scen_df)):
        scenario_dict['scenario'].append(scen_df.iloc[row].scenario)
        scen_name = scen_df.iloc[row].comment
        scenario_dict['scen_name'].append(comment)
        if scen_name == 'perc_stream_buffer':
            # result_ras = generate_buf_scenario 
            # (must add fields to table for args to this function)
            # scenario_dict['lulc_raster'].append(result_ras)
            # for now:
            continue
        else:    
            merge_l = scen_df.iloc[row].lulc_merge_list
            rasters_to_merge = [os.path.join(data_dir, f) for f in merge_l]
            result_ras = os.path.join(data_dir, '%d.tif' % scen_name)
            merge_rasters(rasters_to_merge, result_ras)
            scenario_dict['lulc_raster'].append(result_ras)
    run_df = pandas.data_frame(scenario_dict)        
    return run_df

def remove_reservoir_area(sed_export_ras, lulc_ras, watersheds):
    """Reservoirs are coded as bare ground, but we need to remove their
    contributions to usle and sediment export.  Assume that reservoirs are
    coded in lulc raster as 999 and 1000."""
    
    # first set sed export of reservoir extents to 0
    def reclassify_result(result_ras, lulc_ras):
        result_ras[lulc_ras == 999] = 0
        result_ras[lulc_ras == 1000] = 0
        return result_ras
    
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            sed_export_ras)
    sed_export_cor = os.path.join(os.path.dirname(sed_export_ras), 
                                  '%s_cor.tif' % 
                                  os.path.basename(sed_export_ras)[:-4])
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [sed_export_ras, lulc_ras], reclassify_result, sed_export_cor,
            gdal.GDT_Float32, -9999, out_pixel_size, "union",
            dataset_to_align_index=0, vectorize_op=False)    
    
    # recalculate summary shapefile attributes
    field_summaries = {
        'sed_expcor': pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
                                    sed_export_cor, watersheds, 'ws_id').total,
        }
    esri_driver = ogr.GetDriverByName('ESRI Shapefile')
    original_datasource = ogr.Open(watersheds)
    watershed_output_datasource_uri = os.path.join(
                              os.path.dirname(watersheds), 
                              '%s_cor.shp' % os.path.basename(watersheds)[:-4])
    #If there is already an existing shapefile with the same name and path, delete it
    #Copy the input shapefile into the designated output folder
    if os.path.isfile(watershed_output_datasource_uri):
        os.remove(watershed_output_datasource_uri)
    datasource_copy = esri_driver.CopyDataSource(original_datasource, watershed_output_datasource_uri)
    layer = datasource_copy.GetLayer()

    for field_name in field_summaries:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        layer.CreateField(field_def)

    #Initialize each feature field to 0.0
    for feature_id in xrange(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        for field_name in field_summaries:
            try:
                ws_id = feature.GetFieldAsInteger('ws_id')
                feature.SetField(field_name, float(field_summaries[field_name][ws_id]))
            except KeyError:
                LOGGER.warning('unknown field %s' % field_name)
                feature.SetField(field_name, 0.0)
        #Save back to datasource
        layer.SetFeature(feature)

    original_datasource.Destroy()
    datasource_copy.Destroy()
        
def launch_sdr_collect_results(run_df, TFA_dict, results_csv):
    
    sdr_args = { 
        'biophysical_table_path': u'C:/Users/Ginger/Downloads/Ginger_new_data_8.25.16/biophysical_table_MODIS_8.31.16.csv',
        u'dem_path': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/DEM_strm90m_subset_fill.tif',
        u'drainage_path': u'',
        u'erodibility_path': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/erodibility_ISRICSoilGrids250m_7.5arcseconds_subset_prj.tif',
        u'erosivity_path': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/annual_prec_vb_erosivity_proj.tif',
        u'ic_0_param': u'0.5',
        u'k_param': u'2',
        u'lulc_path': '',
        u'results_suffix': '',
        u'sdr_max': u'0.8',
        u'threshold_flow_accumulation': '',
        u'watersheds_path': u'C:/Users/Ginger/Downloads/Volta_data_8.25.16/watershed_results_sdr_merge.shp',
        u'workspace_dir': 'C:/Users/Ginger/Desktop/test_sdr.8.31.16',
    }
    
    results_dict = {'scen_name': [], 'lulc_raster': [], 'Res_name': [],
                    'TFA': [], 'ws_id': [], 'sediment_export': []}

    for row in xrange(len(run_df)):
        scen_name = run_df.iloc[row].scen_name
        lulc_raster = run_df.iloc[row].lulc_raster
        sdr_args['lulc_path'] = lulc_raster
        for TFA_val in TFA_dict.keys():
            sdr_args['threshold_flow_accumulation'] = TFA_val
            sdr_args['watersheds_path'] = TFA_dict[TFA_val]
            sdr_args['results_suffix'] = '%s_TFA%d' % (scen_name, TFA_val)
            # if TFA_val != 25:
                # natcap.invest.sdr.execute(sdr_args)
            
            # post-process: remove contribution of reservoir area to sediment
            # export
            sed_export_ras = os.path.join(sdr_args['workspace_dir'],
                                          'sed_export_%s.tif' %
                                          sdr_args['results_suffix'])
            lulc_ras = lulc_raster
            watersheds = os.path.join(sdr_args['workspace_dir'],
                                      'watershed_results_sdr_%s.shp' %
                                      sdr_args['results_suffix'])
            remove_reservoir_area(sed_export_ras, lulc_ras, watersheds)
            corrected_summary_shp = os.path.join(os.path.dirname(watersheds), 
                              '%s_cor.shp' % os.path.basename(watersheds)[:-4])
            fields = ['ws_id', 'sed_expcor', 'Res_name']
            num_rows = 0
            with da.SearchCursor(corrected_summary_shp, fields) as cursor:
                for row in cursor:
                    ws_id = row[0]
                    sed_export = row[1]
                    res_name = row[2]
                    results_dict['ws_id'].append(ws_id)
                    results_dict['sediment_export'].append(sed_export)
                    results_dict['Res_name'].append(res_name)
                    num_rows += 1
            while num_rows > 0:
                results_dict['scen_name'].append(scen_name)
                results_dict['lulc_raster'].append(lulc_raster)
                results_dict['TFA'].append(TFA_val)
                num_rows -= 1        
    results_df = pandas.DataFrame(results_dict)
    results_df.to_csv(results_csv)

def whole_shebang(scenario_csv, data_dir, results_csv):
    TFA100_watersheds = r"C:\Users\Ginger\Downloads\Ginger_new_data_8.25.16\watersheds_tfa100.shp"
    TFA25_watersheds = r"C:\Users\Ginger\Downloads\Ginger_new_data_8.25.16\watersheds_tfa25.shp"
    TFA_dict = {25: TFA25_watersheds, 100: TFA100_watersheds}
    run_df = generate_scenarios_from_table(scenario_csv, data_dir)
    launch_sdr_collect_results(run_df, TFA_dict, results_csv)

def test_sdr_run():
    test_run_df = r"C:\Users\Ginger\Downloads\test_run_df.csv"
    TFA100_watersheds = r"C:\Users\Ginger\Downloads\Ginger_new_data_8.25.16\watersheds_tfa100.shp"
    TFA25_watersheds = r"C:\Users\Ginger\Downloads\Ginger_new_data_8.25.16\watersheds_tfa25.shp"
    TFA_dict = {25: TFA25_watersheds, 100: TFA100_watersheds}
    run_df = pandas.read_csv(test_run_df)
    results_csv = 'C:/Users/Ginger/Desktop/scenario_results_8.30.16.csv'
    launch_sdr_collect_results(run_df, TFA_dict, results_csv)
    
if __name__ == '__main__':
    # scenario_csv = 'C:/Users/Ginger/Desktop/scenario_table_8.30.16.csv'
    # data_dir = 'C:/Users/Ginger/Desktop/Volta_data'
    # results_csv = 'C:/Users/Ginger/Desktop/scenario_results_8.30.16.csv'
    # whole_shebang(scenario_csv, data_dir, results_csv)
    test_sdr_run()
    