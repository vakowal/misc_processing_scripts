"""
The ugliest script ever written.
Run SDR and NDR for Mato Grosso and Iowa, Unilever, 10.7.15
"""

import os
import sys
import re
import shutil
import natcap.invest.sdr.sdr
import natcap.invest.ndr.ndr

if __name__ == '__main__':
    scenario_dir = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\2016-02-05_SCENARIOS_ws_extent\uncertainty"
    result_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    IA_scenario_files = ['IA_I_SGBP.tif',
                         'IA_J_SGBP.tif']
    IA_scenarios = [os.path.join(scenario_dir, f) for f in IA_scenario_files]
    IA_biophys_default = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final\biophysical_coeffs_Iowa_Unilever_Corn_ext-int.csv"
    IA_biophys_minus15NDR50SDR = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final\biophysical_coeffs_Iowa_Unilever_Corn_ext-int_minus15NDR50SDR.csv"
    IA_biophys_plus15NDR50SDR = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final\biophysical_coeffs_Iowa_Unilever_Corn_ext-int_plus15NDR50SDR.csv"
    
    ## SDR Iowa                        
    IA_SDR_workspace = os.path.join(result_dir, "IA_SDR")
    if not os.path.exists(IA_SDR_workspace):
        os.makedirs(IA_SDR_workspace)
        
    IA_inputs_dir = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Iowa_watershed_extent"

    IA_SDR_args = {
                u'biophysical_table_uri': 'fill_in',
                u'dem_uri': os.path.join(IA_inputs_dir,
                                         'DEM_SRTM_Iowa_HUC8_180m_fill.tif'),
                u'drainage_uri': u'',
                u'erodibility_uri': os.path.join(IA_inputs_dir,
                                             'erodibility_ISRIC_30arcsec.tif'),
                u'erosivity_uri': os.path.join(IA_inputs_dir,
                                          'erosivity_Iowa_CIAT_1km_prj.tif'),
                u'ic_0_param': u'0.5',
                u'k_param': u'2',
                u'lulc_uri': '',
                u'results_suffix': '',
                u'sdr_max': u'0.8',
                u'threshold_flow_accumulation': u'500',
                u'watersheds_uri': os.path.join(IA_inputs_dir,
                                           'HUC8_Iowa_intersect_dissolve.shp'),
                u'workspace_dir': IA_SDR_workspace,
        }
    
    # run default biophys for all scenarios and uncertainty
    for scenario in IA_scenarios:
        IA_SDR_args[u'biophysical_table_uri'] = IA_biophys_default
        IA_SDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        IA_SDR_args[u'results_suffix'] = lulc_name + "_default_biophys"
        print "******* executing SDR Iowa scenario " + lulc_name
        natcap.invest.sdr.sdr.execute(IA_SDR_args)
        shutil.rmtree(os.path.join(IA_SDR_workspace, 'intermediate'))
    
    # run sensitivity biophys for scenario 3 only
    scenario_3 = os.path.join(scenario_dir, 'IA_3_set0006_.tif')
    for biophys in [IA_biophys_minus15NDR50SDR, IA_biophys_plus15NDR50SDR]:
        IA_SDR_args[u'biophysical_table_uri'] = biophys
        IA_SDR_args[u'lulc_uri'] = scenario_3
        IA_SDR_args[u'results_suffix'] = "scen_3" + biophys[-18:-4]
        print "******* executing SDR Iowa scenario 3"
        # natcap.invest.sdr.sdr.execute(IA_SDR_args)
        # shutil.rmtree(os.path.join(IA_SDR_workspace, 'intermediate'))
    
    # NDR Iowa
    IA_NDR_workspace = os.path.join(result_dir, "IA_NDR")
    if not os.path.exists(IA_NDR_workspace):
        os.makedirs(IA_NDR_workspace)
    
    IA_NDR_args = {
            u'biophysical_table_uri': "fill_in",
            u'calc_n': True,
            u'calc_p': False,
            u'dem_uri': os.path.join(IA_inputs_dir,
                                     'DEM_SRTM_Iowa_HUC8_180m_fill.tif'),
            u'k_param': u'2',
            u'lulc_uri': '',
            u'results_suffix': '',
            u'subsurface_critical_length_n': u'1500',
            u'subsurface_eff_n': u'0.5',
            u'threshold_flow_accumulation': u'500',
            u'watersheds_uri': os.path.join(IA_inputs_dir,
                                           'HUC8_Iowa_intersect_dissolve.shp'),
            u'workspace_dir': IA_NDR_workspace,
        }
        
    # run default biophys for all scenarios and uncertainty
    for scenario in IA_scenarios:
        IA_NDR_args[u'biophysical_table_uri'] = IA_biophys_default
        IA_NDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        IA_NDR_args[u'results_suffix'] = lulc_name + "_default_biophys"
        print "******* executing NDR Iowa scenario " + lulc_name
        natcap.invest.ndr.ndr.execute(IA_NDR_args)
        shutil.rmtree(os.path.join(IA_NDR_workspace, 'intermediate'))
    
    # run sensitivity biophys for scenario 3 only
    scenario_3 = os.path.join(scenario_dir, 'IA_3_set0006_.tif')
    for biophys in [IA_biophys_minus15NDR50SDR, IA_biophys_plus15NDR50SDR]:
        IA_NDR_args[u'biophysical_table_uri'] = biophys
        IA_NDR_args[u'lulc_uri'] = scenario_3
        IA_NDR_args[u'results_suffix'] = "scen_3" + biophys[-18:-4]
        print "******* executing NDR Iowa scenario 3"
        # natcap.invest.ndr.ndr.execute(IA_NDR_args)
        # shutil.rmtree(os.path.join(IA_NDR_workspace, 'intermediate'))
    
    ## NDR Mato Grosso
    MT_scenario_files = ['MT_I_SGBP.tif',
                         'MT_J_SGBP.tif']
    MT_NDR_scenarios = [os.path.join(scenario_dir, f) for f in MT_scenario_files]
    MT_biophys_folder = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final"
    MT_def_biophys = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final\biophysical_coeffs_Brazil_Unilever_sugarcane_Sce3-LCA.csv"
                        
    MT_NDR_workspace = os.path.join(result_dir, "MT_NDR")
    if not os.path.exists(MT_NDR_workspace):
        os.makedirs(MT_NDR_workspace)

    MT_inputs_dir = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\MatoGrosso_watershed_extent"
    MT_NDR_args = {
            u'biophysical_table_uri': MT_def_biophys,
            u'calc_n': True,
            u'calc_p': False,
            u'dem_uri': os.path.join(MT_inputs_dir, 'dem_180_fill.tif'),
            u'k_param': u'2',
            u'lulc_uri': '',
            u'results_suffix': '',
            u'subsurface_critical_length_n': u'1500',
            u'subsurface_eff_n': u'0.5',
            u'threshold_flow_accumulation': u'600',
            u'watersheds_uri': os.path.join(MT_inputs_dir,
                                            'hydro_1k_dissolve.shp'),
            u'workspace_dir': MT_NDR_workspace,
        }

    for scenario in MT_NDR_scenarios:
        MT_NDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        MT_NDR_args[u'results_suffix'] = lulc_name
        print "******* executing NDR Mato Grosso scenario " + lulc_name
        natcap.invest.ndr.ndr.execute(MT_NDR_args)
        shutil.rmtree(os.path.join(MT_NDR_workspace, 'intermediate'))

    # run without intensification
    MT_scenario_files = ['MT_1_set0006_.tif',
                         'MT_2_set0006_.tif',
                         'MT_3_set0006_.tif',
                         'MT_4_set0006_.tif',
                         'MT_5_set0006_.tif']
    MT_scenarios = [os.path.join(scenario_dir, f) for f in MT_scenario_files]
    MT_biophys = r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\Biophysical_Final\biophysical_coeffs_Brazil_Unilever_sugarcane_no-intens.csv"
    for scenario in MT_scenarios:
        MT_NDR_args[u'lulc_uri'] = scenario
        MT_NDR_args[u'biophysical_table_uri'] = MT_biophys
        lulc_name = os.path.basename(scenario)[3:-4]
        MT_NDR_args[u'results_suffix'] = lulc_name + "_no-intens"
        print "******* executing NDR Mato Grosso scenario " + lulc_name
        # natcap.invest.ndr.ndr.execute(MT_NDR_args)
        # shutil.rmtree(os.path.join(MT_NDR_workspace, 'intermediate'))
        
    ## SDR Mato Grosso
    MT_SDR_workspace = os.path.join(result_dir, "MT_SDR")
    if not os.path.exists(MT_SDR_workspace):
        os.makedirs(MT_SDR_workspace)

    MT_SDR_scenarios = [r"C:\Users\Ginger\Dropbox\Unilever_bioplastics_model_inputs\2016-02-05_SCENARIOS_ws_extent\uncertainty\MT_B_set0007_.tif"]
    MT_SDR_args = {
                u'biophysical_table_uri': MT_def_biophys,
                u'dem_uri': os.path.join(MT_inputs_dir, 'dem_180_fill.tif'),
                u'drainage_uri': u'',
                u'erodibility_uri': os.path.join(MT_inputs_dir,
                                             'erodibility_ISRIC_30arcsec.tif'),
                u'erosivity_uri': os.path.join(MT_inputs_dir,
                                      'erosivity_MatoGrosso_CIAT_1km_prj.tif'),
                u'ic_0_param': u'0.5',
                u'k_param': u'2',
                u'lulc_uri': '',
                u'results_suffix': '',
                u'sdr_max': u'0.8',
                u'threshold_flow_accumulation': u'600',
                u'watersheds_uri': os.path.join(MT_inputs_dir,
                                                'hydro_1k_dissolve.shp'),
                u'workspace_dir': MT_SDR_workspace,
        }
    
    for scenario in MT_SDR_scenarios:
        MT_SDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        MT_SDR_args[u'results_suffix'] = lulc_name
        print "******* executing SDR scenario " + lulc_name
        natcap.invest.sdr.sdr.execute(MT_SDR_args)
        shutil.rmtree(os.path.join(MT_SDR_workspace, 'intermediate'))
