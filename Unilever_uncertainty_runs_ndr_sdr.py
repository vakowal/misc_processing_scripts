"""
The ugliest script ever written.
Run SDR and NDR for Mato Grosso and Iowa, Unilever, 10.7.15
"""

import os
import sys
import re
import natcap.invest.sdr.sdr
import natcap.invest.ndr.ndr


if __name__ == '__main__':
    scenario_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\scenarios_1.16.15"
    result_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results_1.16.15"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    IA_scenario_files = ['new_scen_0002_B_IA_m.tif']
    IA_scenarios = [os.path.join(scenario_dir, f) for f in IA_scenario_files]
    IA_biophys = r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Final per scenario\biophysical_coeffs_Iowa_Unilever_Corn_ext-int.csv"
    
    ## SDR Iowa                        
    IA_SDR_workspace = os.path.join(result_dir, "IA_SDR")
    if not os.path.exists(IA_SDR_workspace):
        os.makedirs(IA_SDR_workspace)
        
    inputs_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Iowa_watershed_extent"

    IA_SDR_args = {
                u'biophysical_table_uri': IA_biophys,
                u'dem_uri': os.path.join(inputs_dir,
                                         'DEM_SRTM_Iowa_HUC8_180m_fill.tif'),
                u'drainage_uri': u'',
                u'erodibility_uri': os.path.join(inputs_dir,
                                             'erodibility_ISRIC_30arcsec.tif'),
                u'erosivity_uri': os.path.join(inputs_dir,
                                          'erosivity_Iowa_CIAT_1km_prj.tif'),
                u'ic_0_param': u'0.5',
                u'k_param': u'2',
                u'lulc_uri': '',
                u'results_suffix': '',
                u'sdr_max': u'0.8',
                u'threshold_flow_accumulation': u'500',
                u'watersheds_uri': os.path.join(inputs_dir,
                                           'HUC8_Iowa_intersect_dissolve.shp'),
                u'workspace_dir': IA_SDR_workspace,
        }

    for scenario in IA_scenarios:
        IA_SDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        IA_SDR_args[u'results_suffix'] = lulc_name
        print "******* executing SDR Iowa scenario " + lulc_name
        natcap.invest.sdr.sdr.execute(IA_SDR_args)
    
    # NDR Iowa
    IA_NDR_workspace = os.path.join(result_dir, "IA_NDR")
    if not os.path.exists(IA_NDR_workspace):
        os.makedirs(IA_NDR_workspace)
    
    IA_NDR_args = {
            u'biophysical_table_uri': IA_biophys,
            u'calc_n': True,
            u'calc_p': False,
            u'dem_uri': os.path.join(inputs_dir,
                                     'DEM_SRTM_Iowa_HUC8_180m_fill.tif'),
            u'k_param': u'2',
            u'lulc_uri': '',
            u'results_suffix': '',
            u'subsurface_critical_length_n': u'1500',
            u'subsurface_eff_n': u'0.5',
            u'threshold_flow_accumulation': u'500',
            u'watersheds_uri': os.path.join(inputs_dir,
                                           'HUC8_Iowa_intersect_dissolve.shp'),
            u'workspace_dir': IA_NDR_workspace,
        }
    for scenario in IA_scenarios:
        IA_NDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        IA_NDR_args[u'results_suffix'] = lulc_name
        print "******* executing NDR Iowa scenario " + lulc_name
        natcap.invest.ndr.ndr.execute(IA_NDR_args)

    ## NDR Mato Grosso
    MT_scenario_files = ['new_scen_0002_B_MT_m.tif']
    MT_scenarios = [os.path.join(scenario_dir, f) for f in MT_scenario_files]
    MT_biophys = u"C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/Final per scenario/biophysical_coeffs_Brazil_Unilever_sugarcane_Sce3b.csv"
    
    MT_NDR_workspace = os.path.join(result_dir, "MT_NDR")
    if not os.path.exists(MT_NDR_workspace):
        os.makedirs(MT_NDR_workspace)

    inputs_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\MatoGrosso_watershed_extent"
    MT_NDR_args = {
            u'biophysical_table_uri': MT_biophys,
            u'calc_n': True,
            u'calc_p': False,
            u'dem_uri': os.path.join(inputs_dir, 'dem_180_fill.tif'),
            u'k_param': u'2',
            u'lulc_uri': '',
            u'results_suffix': '',
            u'subsurface_critical_length_n': u'1500',
            u'subsurface_eff_n': u'0.5',
            u'threshold_flow_accumulation': u'600',
            u'watersheds_uri': os.path.join(inputs_dir,
                                            'hydro_1k_dissolve.shp'),
            u'workspace_dir': MT_NDR_workspace,
        }
    for scenario in MT_scenarios:
        MT_NDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        MT_NDR_args[u'results_suffix'] = lulc_name
        print "******* executing NDR Mato Grosso scenario " + lulc_name
        natcap.invest.ndr.ndr.execute(MT_NDR_args)

    ## SDR Mato Grosso
    MT_SDR_workspace = os.path.join(result_dir, "MT_SDR")
    if not os.path.exists(MT_SDR_workspace):
        os.makedirs(MT_SDR_workspace)

    inputs_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\MatoGrosso_watershed_extent"

    MT_SDR_args = {
                u'biophysical_table_uri': MT_biophys,
                u'dem_uri': os.path.join(inputs_dir, 'dem_180_fill.tif'),
                u'drainage_uri': u'',
                u'erodibility_uri': os.path.join(inputs_dir,
                                             'erodibility_ISRIC_30arcsec.tif'),
                u'erosivity_uri': os.path.join(inputs_dir,
                                      'erosivity_MatoGrosso_CIAT_1km_prj.tif'),
                u'ic_0_param': u'0.5',
                u'k_param': u'2',
                u'lulc_uri': '',
                u'results_suffix': '',
                u'sdr_max': u'0.8',
                u'threshold_flow_accumulation': u'600',
                u'watersheds_uri': os.path.join(inputs_dir,
                                                'hydro_1k_dissolve.shp'),
                u'workspace_dir': MT_SDR_workspace,
        }
    
    MT_scenario_files = ['new_scen_0002_L_MT_m.tif',
                         'new_scen_0002_B_MT_m.tif']
    MT_scenarios = [os.path.join(scenario_dir, f) for f in MT_scenario_files]
    for scenario in MT_scenarios:
        MT_SDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        MT_SDR_args[u'results_suffix'] = lulc_name
        print "******* executing SDR scenario " + lulc_name
        natcap.invest.sdr.sdr.execute(MT_SDR_args)
