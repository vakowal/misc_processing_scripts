""""
Volta SDR: generate scenarios, run model, collect results.
"""

import os
import numpy as np
import natcap.invest.scenario_gen_proximity
import natcap.invest.sdr
from osgeo import gdal
from osgeo import ogr
import pygeoprocessing.geoprocessing
import pandas
from datetime import datetime

def count_lulc_composition(watersheds, lulc):
    atest = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
                                                    lulc, watersheds, 'ws_id')
    print atest
    
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
            raster[raster == 255] = 0
            np.copyto(result_raster, raster, where=raster > 0)
        return result_raster
    
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            rasters_to_merge[0])
    pygeoprocessing.geoprocessing.vectorize_datasets(
            rasters_to_merge, merge_op, save_as,
            gdal.GDT_Int32, -9999, out_pixel_size, "union",
            dataset_to_align_index=0, vectorize_op=False)
  
def generate_buf_scenario(substrate_lulc, aoi_shp, aoi_dir, percent_to_convert,
                          result_raster):
    """Main function to generate the flexible buffer area scenario.  Percent
    to convert is the percent of full buffer area that should be converted to
    riparian buffer. aoi_shp is a path to a shapefile containing all
    watersheds; aoi_dir is a directory that contains separate shapefiles for
    each watershed.  substrate_lulc must contain full buffer area
    coded as 222."""
    
    # stream cell just inside reservoir inlet was reclassified
    # as the integer 'focal_landcover' (1000)
                
    # get area within each aoi of pixel type 222
    # first: reclassify to keep just pixel type 222
    ras_222 = os.path.join(os.path.dirname(substrate_lulc), '222_only.tif')
    if not os.path.exists(ras_222):
        def reclassify_to_222(substrate_lulc):
            result_raster = np.full((substrate_lulc.shape), 255, dtype=np.int)
            np.copyto(result_raster, substrate_lulc,
                      where=substrate_lulc == 222)
            return result_raster
        out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            substrate_lulc)
        pygeoprocessing.geoprocessing.vectorize_datasets(
                [substrate_lulc], reclassify_to_222, ras_222,
                gdal.GDT_Int32, 255, out_pixel_size, "union",
                dataset_to_align_index=0, vectorize_op=False)
    sum_dict = pygeoprocessing.aggregate_raster_values_uri(
                                   ras_222, aoi_shp, 'ws_id').total
    area_dict = {key: (sum_dict[key] / 222) * 0.865994 for key in
                 sum_dict.keys()}
    result_list = []
    for ws_id in area_dict.keys():
        aoi = os.path.join(aoi_dir, 'ws_id_%d.shp' % ws_id)
        convertible_area = percent_to_convert * area_dict[ws_id]
        scen_gen_args = {
                u'aoi_path': aoi,
                u'area_to_convert': str(convertible_area),
                u'base_lulc_path': substrate_lulc,
                u'convert_farthest_from_edge': False,
                u'convert_nearest_to_edge': True,
                u'convertible_landcover_codes': '222',
                u'focal_landcover_codes': u'1000',
                u'n_fragmentation_steps': u'1',
                u'replacment_lucode': '1001',
                u'workspace_dir': r"C:\Users\Ginger\Desktop\scenario_generator",
        }
        natcap.invest.scenario_gen_proximity.execute(scen_gen_args)
        scen_result_ras = os.path.join(scen_gen_args['workspace_dir'],
                                       'nearest_to_edge.tif')
    
        # remove areas of stream buffer that were not filled, reclassify filled
        # buffer as 222
        def remove_unfilled_buffer(scen_result_ras):
            scen_result_ras[scen_result_ras == 222] = -9999
            scen_result_ras[scen_result_ras == 255] = -9999
            scen_result_ras[scen_result_ras == 1001] = 222
            return scen_result_ras
        out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                scen_result_ras)
        save_as = os.path.join(scen_gen_args['workspace_dir'],
                               'buf_cor_%d.tif' % ws_id)
        pygeoprocessing.geoprocessing.vectorize_datasets(
                [scen_result_ras], remove_unfilled_buffer, save_as,
                gdal.GDT_Int32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, vectorize_op=False)
        result_list.append(save_as)
    # mosaic all constituent pieces together
    merge_rasters(result_list, result_raster)

def generate_scenarios_from_table(scenario_csv, data_dir):
    """Put pieces together to create lulc scenarios."""
    
    scenario_lulc_folder = os.path.join(data_dir, 'scenario_lulc')
    if not os.path.exists(scenario_lulc_folder):
        os.makedirs(scenario_lulc_folder)
    scenario_dict = {'scenario': [], 'scen_name': [], 'lulc_raster': []}
    scen_df = pandas.read_csv(scenario_csv)
    for row in xrange(len(scen_df)):
        scenario_dict['scenario'].append(scen_df.iloc[row].scenario)
        scen_name = scen_df.iloc[row].scen_name
        scenario_dict['scen_name'].append(scen_name)
        merge_l = scen_df.iloc[row].lulc_merge_list.split(', ')
        rasters_to_merge = [os.path.join(data_dir, f) for f in merge_l]
        result_ras = os.path.join(scenario_lulc_folder, '%s.tif' % scen_name)
        scenario_dict['lulc_raster'].append(result_ras)
        if not os.path.exists(result_ras):
            merge_rasters(rasters_to_merge, result_ras)
    run_df = pandas.DataFrame(scenario_dict)
    now_str = datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
    run_df.to_csv(os.path.join(data_dir, 'run_table_%s.csv' % now_str),
                  index=False)
    return run_df

def remove_reservoir_area(sed_export_ras, usle_ras, lulc_ras, watersheds):
    """Reservoirs are coded as bare ground, but we need to remove their
    contributions to usle and sediment export in post-processing.  Assume that
    reservoirs are coded in lulc raster as 999 and 1000."""
    
    def reclassify_result(result_ras, lulc_ras):
        result_ras[lulc_ras == 999] = 0
        result_ras[lulc_ras == 1000] = 0
        return result_ras
    
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            sed_export_ras)
    
    # set sediment export of reservoir extents to 0
    sed_export_cor = os.path.join(os.path.dirname(sed_export_ras), 
                                  '%s_cor.tif' % 
                                  os.path.basename(sed_export_ras)[:-4])
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [sed_export_ras, lulc_ras], reclassify_result, sed_export_cor,
            gdal.GDT_Float32, -1, out_pixel_size, "union",
            dataset_to_align_index=0, vectorize_op=False)
    
    # set usle of reservoir extents to 0
    usle_cor = os.path.join(os.path.dirname(usle_ras), 
                                  '%s_cor.tif' % 
                                  os.path.basename(usle_ras)[:-4])
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [usle_ras, lulc_ras], reclassify_result, usle_cor,
            gdal.GDT_Float32, -1, out_pixel_size, "union",
            dataset_to_align_index=0, vectorize_op=False)    
    
    # recalculate summary shapefile attributes
    field_summaries = {
        'sed_expcor': pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
                                    sed_export_cor, watersheds, 'ws_id').total,
        'usle_totco': pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
                                    usle_cor, watersheds, 'ws_id').total,
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
    kb_ic_list = [(20, 0.1)]  # [(10, -0.3), (10, -0.5), (10, -1.)]
    
    # erosivity options:
    # u'C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/erosivity_30s_prj_ex.tif'
    # u'C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/annual_prec_vb_erosivity_proj.tif'
    # u"C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/erosivity_MFI_GK_1.6.17.tif"
    
    sdr_args = { 
        'biophysical_table_path': u"C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/scenario_data_8.31.16/biophysical_table_MODIS_12.20.16.csv",
        u'dem_path': u"C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/DEM_strm90m_subset_fill.tif",
        u'drainage_path': u'',
        u'erodibility_path': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/erodibility_ISRICSoilGrids250m_7.5arcseconds_subset_prj.tif',
        u'erosivity_path': u"C:/Users/Ginger/Documents/NatCap/GIS_local/Corinne/Volta/erosivity_MFI_GK_1.6.17.tif",
        u'ic_0_param': '',
        u'k_param': '',
        u'lulc_path': '',
        u'results_suffix': '',
        u'sdr_max': u'1.0',
        u'threshold_flow_accumulation': '',
        u'watersheds_path': '',
        u'workspace_dir': 'C:/Users/Ginger/Desktop/scenarios_1.9.17',
    }
    
    results_dict = {'scen_name': [], 'lulc_raster': [], 'Res_name': [],
                    'TFA': [], 'ws_id': [], 'sediment_export': [],
                    'usle_tot': [], 'kb': [], 'ic_0': []}
    for kb, ic0 in kb_ic_list:
        sdr_args['k_param'] = kb
        sdr_args['ic_0_param'] = ic0
        for row in xrange(len(run_df)):
            scen_name = run_df.iloc[row].scen_name
            lulc_raster = run_df.iloc[row].lulc_raster
            sdr_args['lulc_path'] = lulc_raster
            for TFA_val in TFA_dict.keys():
                sdr_args['threshold_flow_accumulation'] = TFA_val
                sdr_args['watersheds_path'] = TFA_dict[TFA_val]
                sdr_args['results_suffix'] = '%s_TFA%d_kb%d_ic%.2f' % (
                                              scen_name, TFA_val, kb, ic0)
                watersheds = os.path.join(sdr_args['workspace_dir'],
                                          'watershed_results_sdr_%s.shp' %
                                          sdr_args['results_suffix'])
                corrected_summary_shp = os.path.join(os.path.dirname(watersheds), 
                                  '%s_cor.shp' % os.path.basename(watersheds)[:-4])
                # if not os.path.isfile(corrected_summary_shp):
                natcap.invest.sdr.execute(sdr_args)
                
                # post-process: remove contribution of reservoir area to
                # sediment export and usle
                sed_export_ras = os.path.join(sdr_args['workspace_dir'],
                                              'sed_export_%s.tif' %
                                              sdr_args['results_suffix'])
                usle_ras = os.path.join(sdr_args['workspace_dir'],
                                              'usle_%s.tif' %
                                              sdr_args['results_suffix'])
                # if not os.path.exists(corrected_summary_shp):
                # try:
                remove_reservoir_area(sed_export_ras, usle_ras, lulc_raster,
                                      watersheds)
                # except:
                    # continue
                num_rows = 0
                shpf = ogr.Open(corrected_summary_shp)
                layer = shpf.GetLayer(0)
                num_features = layer.GetFeatureCount()
                for i in range(num_features):
                    feature = layer.GetFeature(i)
                    results_dict['ws_id'].append(feature.GetField("ws_id"))
                    results_dict['sediment_export'].append(
                                            feature.GetField("sed_expcor"))
                    results_dict['usle_tot'].append(
                                            feature.GetField("usle_totco"))
                    results_dict['Res_name'].append(feature.GetField("Res_name"))
                    results_dict['scen_name'].append(scen_name)
                    results_dict['lulc_raster'].append(lulc_raster)
                    results_dict['TFA'].append(TFA_val)
                    results_dict['kb'].append(kb)
                    results_dict['ic_0'].append(ic0)
    results_df = pandas.DataFrame(results_dict)
    results_df.to_csv(results_csv, index=False)       
    
def whole_shebang(scenario_csv, data_dir, results_csv):
    TFA100_watersheds = os.path.join(data_dir, "watersheds_tfa100.shp")
    TFA25_watersheds = os.path.join(data_dir, "watersheds_tfa25.shp")
    TFA_dict = {25: TFA25_watersheds, 100: TFA100_watersheds}
    
    # mosaic lulcs for all scenarios
    print "............. mosaic-ing lulc for all scenarios ................"
    run_df = generate_scenarios_from_table(scenario_csv, data_dir)
    
    # launch sdr and collect results
    print "........... launching sdr, collecting results .................."
    launch_sdr_collect_results(run_df, TFA_dict, results_csv)

def launch_buffer_creation(data_dir):
    substrate_lulc = os.path.join(data_dir, "stream_buffer_93m_inlet.tif")
    aoi_shp = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\scenario_data_8.31.16\all_watersheds.shp"
    aoi_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\scenario_data_8.31.16\watersheds_separate"
    for percent_to_convert in [0.2, 0.5]:
        result_raster = os.path.join(data_dir,
                                     'stream_buffer_93m_%.2fperc.tif' %
                                     percent_to_convert)
        generate_buf_scenario(substrate_lulc, aoi_shp, aoi_dir,
                              percent_to_convert, result_raster)
                          
if __name__ == '__main__':
    scenario_csv = "C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\scenario_data_8.31.16\scenario_table_12.21.16.csv"
    # launch_buffer_creation(data_dir)
    data_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\scenario_data_8.31.16"
    results_csv = r"C:\Users\Ginger\Desktop\scenario_results_1.9.17.csv"
    whole_shebang(scenario_csv, data_dir, results_csv)
    # watersheds = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\scenario_data_8.31.16\all_watersheds.shp"
    # lulc = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\scenario_data_8.31.16\scenario_lulc\baseline_MODIS_2010.tif"
    # count_lulc_composition(watersheds, lulc)
