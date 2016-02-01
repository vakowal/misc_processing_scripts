""""
This is a saved model run from natcap.invest.globio.
Generated: 10/22/15 18:20:52
InVEST version: 3.3.0a1.post35+nb15b643bebc2
"""

import glob
import os

from osgeo import ogr
import natcap.invest.globio

output_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results\GLOBIO"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

iowa_args = {
        u'aoi_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/Iowa_state_extent/Becky_GLOBIO_inputs_IA_1.20.16/IA_aoi.shp',
        u'infrastructure_dir': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/Iowa_state_extent/Becky_GLOBIO_inputs_IA_1.20.16/Infrastructure_dir',
        u'intensification_fraction': u'0.9',
        u'lulc_to_globio_table_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/Iowa_state_extent/Becky_GLOBIO_inputs_IA_1.20.16/lulc_conversion_table.csv',
        u'lulc_uri': 'to_be_scripted',
        u'msa_parameters_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/Iowa_state_extent/Becky_GLOBIO_inputs_IA_1.20.16/msa_parameters.csv',
        u'pasture_threshold': u'0.5',
        u'pasture_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/Iowa_state_extent/Becky_GLOBIO_inputs_IA_1.20.16/pasture.tif',
        u'potential_vegetation_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/Iowa_state_extent/Becky_GLOBIO_inputs_IA_1.20.16/potential_veg.tif',
        u'predefined_globio': False,
        u'primary_threshold': u'0.25',
        u'results_suffix': '',
        u'workspace_dir': os.path.join(output_dir, "Iowa"),
}

mg_args = {
        u'aoi_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/MatoGrosso_state_extent/Becky_GLOBIO_inputs_MT_1.20.16/MT_aoi.shp',
        u'infrastructure_dir': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/MatoGrosso_state_extent/Becky_GLOBIO_inputs_MT_1.20.16/infrastructure_dir_p',
        u'intensification_fraction': u'0.5',
        u'lulc_to_globio_table_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/MatoGrosso_state_extent/Becky_GLOBIO_inputs_MT_1.20.16/lulc_conversion_table.csv',
        u'lulc_uri': 'to_be_scripted',
        u'msa_parameters_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/MatoGrosso_state_extent/Becky_GLOBIO_inputs_MT_1.20.16/msa_parameters.csv',
        u'pasture_threshold': u'0.75',
        u'pasture_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/MatoGrosso_state_extent/Becky_GLOBIO_inputs_MT_1.20.16/pasturep.tif',
        u'potential_vegetation_uri': u'C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/MatoGrosso_state_extent/Becky_GLOBIO_inputs_MT_1.20.16/potential_vegetationp.tif',
        u'predefined_globio': False,
        u'primary_threshold': u'0.25',
        u'results_suffix': '',
        u'workspace_dir': os.path.join(output_dir, "Mato_Grosso"),
}

if __name__ == '__main__':
    ## Iowa
    aggregate_table = open(os.path.join(output_dir, 'Grand_results_IA.csv'), 'wb')
    aggregate_table.write('scenario,msa_mean\n')
    for lulc_path in glob.glob(r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Iowa_state_extent\Becky_GLOBIO_inputs_IA_1.20.16\2016-02-01_IA_GLOBIO\*.tif"):
    
        print "PROCESSING SCENARIO %s" % lulc_path
        suffix = os.path.splitext(os.path.basename(lulc_path))[0]
        print suffix
        
        args = iowa_args
        
        args['lulc_uri'] = lulc_path
        args['results_suffix'] = suffix

        natcap.invest.globio.execute(args)
        
        # should be an AOI at workspace_dir + aoi_summary + suffix + .shp
        aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
        aoi_vector = ogr.Open(aoi_path)
        aoi_layer = aoi_vector.GetLayer()
        aggregate_table.write(suffix)
        for feature in aoi_layer:
            aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
        aggregate_table.write('\n')
    
    aggregate_table.close()
    
    ## Mato Grosso
    aggregate_table = open(os.path.join(output_dir, 'Grand_results_MT.csv'), 'wb')
    aggregate_table.write('scenario,msa_mean\n')
    for lulc_path in glob.glob(r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\MatoGrosso_state_extent\Becky_GLOBIO_inputs_MT_1.20.16\2016-02-01_MT_GLOBIO\*.tif"):
    
        print "PROCESSING SCENARIO %s" % lulc_path
        suffix = os.path.splitext(os.path.basename(lulc_path))[0]
        print suffix
        
        args = mg_args
        
        args['lulc_uri'] = lulc_path
        args['results_suffix'] = suffix

        natcap.invest.globio.execute(args)
        
        # should be an AOI at workspace_dir + aoi_summary + suffix + .shp
        aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
        aoi_vector = ogr.Open(aoi_path)
        aoi_layer = aoi_vector.GetLayer()
        aggregate_table.write(suffix)
        for feature in aoi_layer:
            aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
        aggregate_table.write('\n')
    
    aggregate_table.close()