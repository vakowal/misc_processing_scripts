""""
This is a saved model run from natcap.invest.globio.
Generated: 10/22/15 18:20:52
InVEST version: 3.3.0a1.post35+nb15b643bebc2
"""

import glob
import os

from osgeo import ogr
import natcap.invest.globio

output_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results\GLOBIO_uncertainty"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

iowa_args = {
        u'aoi_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/IA_globio/Iowa_state_aoi.shp',
        u'infrastructure_dir': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/IA_globio/infrastructure_dir',
        u'intensification_fraction': u'0.9',
        u'lulc_to_globio_table_uri': u"C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/Biophysical_Final/globio_lulc_conversion_table.csv",
        u'lulc_uri': 'to_be_scripted',
        u'msa_parameters_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/IA_globio/msa_parameters.csv',
        u'pasture_threshold': u'0.5',
        u'pasture_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/IA_globio/pasture.tif',
        u'potential_vegetation_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/IA_globio/potential_veg.tif',
        u'predefined_globio': False,
        u'primary_threshold': u'0.25',
        u'results_suffix': '',
        u'workspace_dir': os.path.join(output_dir, "Iowa"),
}

mg_args = {
        u'aoi_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/MT_globio/MT_state_aoi.shp',
        u'infrastructure_dir': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/MT_globio/infrastructure_dir',
        u'intensification_fraction': u'0.5',
        u'lulc_to_globio_table_uri': u"C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/Biophysical_Final/globio_lulc_conversion_table.csv",
        u'lulc_uri': 'to_be_scripted',
        u'msa_parameters_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/MT_globio/msa_parameters.csv',
        u'pasture_threshold': u'0.75',
        u'pasture_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/MT_globio/pasturep.tif',
        u'potential_vegetation_uri': u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/MT_globio/potential_vegetationp.tif',
        u'predefined_globio': False,
        u'primary_threshold': u'0.25',
        u'results_suffix': '',
        u'workspace_dir': os.path.join(output_dir, "Mato_Grosso"),
}

if __name__ == '__main__':
    uncertainty_scenarios = glob.glob(r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\2016-02-05_SCENARIOS_state-extent\uncertainty\*.tif")
    msa_minusSE = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final\msa_parameters_minusSE.csv"
    msa_plusSE = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final\msa_parameters_plusSE.csv"
    
    ## Iowa
    args = iowa_args
    scen_3 = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\2016-02-05_SCENARIOS_state-extent\IA_3_set0006_.tif"
    scen_0 = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\2016-02-05_SCENARIOS_state-extent\IA_0_set0006_.tif"
    # aggregate_table = open(os.path.join(output_dir, 'Grand_results_IA.csv'), 'wb')
    # aggregate_table.write('scenario,msa_mean\n')
    
    # run scenario 3 uncertainty
    # args['lulc_uri'] = scen_3
    # args[u'msa_parameters_uri'] = msa_minusSE
    # suffix = 'scen_3_minusSE'
    # args['results_suffix'] = suffix
    # natcap.invest.globio.execute(args)
    # aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
    # aoi_vector = ogr.Open(aoi_path)
    # aoi_layer = aoi_vector.GetLayer()
    # aggregate_table.write(suffix)
    # for feature in aoi_layer:
        # aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
    # aggregate_table.write('\n')
    
    args[u'msa_parameters_uri'] = msa_plusSE
    args['lulc_uri'] = scen_0
    suffix = 'scen_0_plusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    args[u'msa_parameters_uri'] = msa_plusSE
    args['lulc_uri'] = scen_3
    suffix = 'scen_3_plusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    # aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
    # aoi_vector = ogr.Open(aoi_path)
    # aoi_layer = aoi_vector.GetLayer()
    # aggregate_table.write(suffix)
    # for feature in aoi_layer:
        # aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
    # aggregate_table.write('\n')
    
    args[u'msa_parameters_uri'] = msa_minusSE
    args['lulc_uri'] = scen_0
    suffix = 'scen_0_minusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    args[u'msa_parameters_uri'] = msa_minusSE
    args['lulc_uri'] = scen_3
    suffix = 'scen_3_minusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    # run uncertainty scenarios
    # for lulc_path in uncertainty_scenarios:
        # print "PROCESSING SCENARIO %s" % lulc_path
        # suffix = os.path.splitext(os.path.basename(lulc_path))[0]
        # if suffix.startswith('MT'):
            # continue
        # print suffix

        # args['lulc_uri'] = lulc_path
        # args['results_suffix'] = suffix
        # args[u'msa_parameters_uri'] = u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/MT_globio/msa_parameters.csv'

        # natcap.invest.globio.execute(args)
        
        # should be an AOI at workspace_dir + aoi_summary + suffix + .shp
        # aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
        # aoi_vector = ogr.Open(aoi_path)
        # aoi_layer = aoi_vector.GetLayer()
        # aggregate_table.write(suffix)
        # for feature in aoi_layer:
            # aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
        # aggregate_table.write('\n')
    
    # aggregate_table.close()
    
    ## Mato Grosso
    args = mg_args
    scen_3 = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\2016-02-05_SCENARIOS_state-extent\MT_3_set0006_.tif"
    scen_0 = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\2016-02-05_SCENARIOS_state-extent\MT_0_set0006_.tif"
    # aggregate_table = open(os.path.join(output_dir, 'Grand_results_MT.csv'), 'wb')
    # aggregate_table.write('scenario,msa_mean\n')
    
    # run scenario 3 uncertainty
    # args['lulc_uri'] = scen_3
    # args[u'msa_parameters_uri'] = msa_minusSE
    # suffix = 'scen_3_minusSE'
    # args['results_suffix'] = suffix
    # natcap.invest.globio.execute(args)
    # aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
    # aoi_vector = ogr.Open(aoi_path)
    # aoi_layer = aoi_vector.GetLayer()
    # aggregate_table.write(suffix)
    # for feature in aoi_layer:
        # aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
    # aggregate_table.write('\n')
    
    args[u'msa_parameters_uri'] = msa_plusSE
    args['lulc_uri'] = scen_0
    suffix = 'scen_0_plusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    args[u'msa_parameters_uri'] = msa_plusSE
    args['lulc_uri'] = scen_3
    suffix = 'scen_3_plusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    # aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
    # aoi_vector = ogr.Open(aoi_path)
    # aoi_layer = aoi_vector.GetLayer()
    # aggregate_table.write(suffix)
    # for feature in aoi_layer:
        # aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
    # aggregate_table.write('\n')
    
    args[u'msa_parameters_uri'] = msa_minusSE
    args['lulc_uri'] = scen_0
    suffix = 'scen_0_minusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    args[u'msa_parameters_uri'] = msa_minusSE
    args['lulc_uri'] = scen_3
    suffix = 'scen_3_minusSE'
    args['results_suffix'] = suffix
    natcap.invest.globio.execute(args)
    
    # run uncertainty scenarios
    # for lulc_path in uncertainty_scenarios:
        # print "PROCESSING SCENARIO %s" % lulc_path
        # suffix = os.path.splitext(os.path.basename(lulc_path))[0]
        # if suffix.startswith('IA'):
            # continue
        # print suffix

        # args['lulc_uri'] = lulc_path
        # args['results_suffix'] = suffix
        # args[u'msa_parameters_uri'] = u'C:/Users/Ginger/Dropbox/Unilever_bioplastics_model_inputs/MT_globio/msa_parameters.csv'

        # natcap.invest.globio.execute(args)
        
        ## should be an AOI at workspace_dir + aoi_summary + suffix + .shp
        # aoi_path = os.path.join(args['workspace_dir'], 'aoi_summary_' + suffix + '.shp')
        # aoi_vector = ogr.Open(aoi_path)
        # aoi_layer = aoi_vector.GetLayer()
        # aggregate_table.write(suffix)
        # for feature in aoi_layer:
            # aggregate_table.write(',' + feature.GetFieldAsString('msa_mean'))
        # aggregate_table.write('\n')
        
    # aggregate_table.close()