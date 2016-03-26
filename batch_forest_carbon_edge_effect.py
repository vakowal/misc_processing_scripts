""""
This is a saved model run from natcap.invest.forest_carbon_edge_effect.
Generated: 10/07/15 11:17:02
InVEST version: 3.3.0a1.post14+n85026d0ba46d
"""

import os
import re
import natcap.invest.forest_carbon_edge_effect



if __name__ == '__main__':
    result_dir = "C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Batched_runs_results"

    MT_scenario_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\scenarios_Oct5\MT_state_extent"
    MT_workspace = os.path.join(result_dir, "MT_Carbon")
    if not os.path.exists(MT_workspace):
        os.makedirs(MT_workspace)

    MT_args = {
            u'aoi_uri': u'C:\\Users\\Ginger\\Dropbox\\NatCap_backup\\Unilever\\Unilever_model_inputs_10.6.15\\MatoGrosso_state_extent\\Mato_Grosso.shp',
            u'biomass_to_carbon_conversion_factor': u'0.47',
            u'biophysical_table_uri': '',
            u'lulc_uri': '',
            u'n_nearest_model_points': u'10',
            u'results_suffix': '',
            u'tropical_forest_edge_carbon_model_shape_uri': u'C:\\InVEST_3.3.0a1.post14+n85026d0ba46d_x86\\forest_carbon_edge_effect\\core_data\\forest_carbon_edge_regression_model_parameters.shp',
            u'workspace_dir': MT_workspace,
    }
    files = [f for f in os.listdir(MT_scenario_dir) if
                              os.path.isfile(os.path.join(MT_scenario_dir, f))]
    tifs = [f for f in files if re.search('.tif$', f)]
    MT_scenarios = [os.path.join(MT_scenario_dir, f) for f in tifs]
    
    # find and run baseline scenario
    baseline_l = [f for f in MT_scenarios if re.search('baseline', f)]
    assert len(baseline_l) == 1
    baseline = baseline_l[0]
    MT_scenarios.remove(baseline)
    
    MT_args[u'lulc_uri'] = baseline
    MT_args[u'results_suffix'] = os.path.basename(baseline)[3:-4]
    natcap.invest.forest_carbon_edge_effect.execute(MT_args)
    
    for scenario in MT_scenarios:
        MT_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        
        # run low intensity for this scenario
        suffix = lulc_name + '_low'
        ## how to represent low and high intensity??
        MT_args[u'results_suffix'] = suffix
        natcap.invest.forest_carbon_edge_effect.execute(MT_args)
        
        # run high intensity for this scenario
        suffix = lulc_name + '_high'
        MT_args[u'results_suffix'] = suffix
        natcap.invest.forest_carbon_edge_effect.execute(MT_args)
    
    # run Iowa
    IA_scenario_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\scenarios_Oct5\IA_state_extent"
    IA_workspace = os.path.join(result_dir, "IA_Carbon")
    if not os.path.exists(IA_workspace):
        os.makedirs(IA_workspace)

    IA_args = {
            u'aoi_uri': u'C:\\Users\\Ginger\\Dropbox\\NatCap_backup\\Unilever\\Unilever_model_inputs_10.6.15\\IA_state_extent\\Iowa_outline_proj.shp',
            u'biomass_to_carbon_conversion_factor': u'0.47',
            u'biophysical_table_uri': '',
            u'lulc_uri': '',
            u'n_nearest_model_points': u'10',
            u'results_suffix': '',
            u'tropical_forest_edge_carbon_model_shape_uri': u'C:\\InVEST_3.3.0a1.post14+n85026d0ba46d_x86\\forest_carbon_edge_effect\\core_data\\forest_carbon_edge_regression_model_parameters.shp',
            u'workspace_dir': IA_workspace,
    }
    files = [f for f in os.listdir(IA_scenario_dir) if
                              os.path.isfile(os.path.join(IA_scenario_dir, f))]
    tifs = [f for f in files if re.search('.tif$', f)]
    IA_scenarios = [os.path.join(IA_scenario_dir, f) for f in tifs]
    
    # find and run baseline scenario
    baseline_l = [f for f in IA_scenarios if re.search('baseline', f)]
    assert len(baseline_l) == 1
    baseline = baseline_l[0]
    IA_scenarios.remove(baseline)
    
    IA_args[u'lulc_uri'] = baseline
    IA_args[u'results_suffix'] = os.path.basename(baseline)[3:-4]
    natcap.invest.forest_carbon_edge_effect.execute(IA_args)
    
    for scenario in IA_scenarios:
        IA_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        
        # run low intensity for this scenario
        suffix = lulc_name + '_low'
        ## how to represent low and high intensity??
        IA_args[u'results_suffix'] = suffix
        natcap.invest.forest_carbon_edge_effect.execute(IA_args)
        
        # run high intensity for this scenario
        suffix = lulc_name + '_high'
        IA_args[u'results_suffix'] = suffix
        natcap.invest.forest_carbon_edge_effect.execute(IA_args)
