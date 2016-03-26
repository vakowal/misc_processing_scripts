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
    result_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results_1.16.15"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
            
    ## SDR Iowa                        
    IA_scenario_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\scenarios_Oct5\IA_watershed_extent"
    IA_workspace = os.path.join(result_dir, "IA_SDR")
    if not os.path.exists(IA_workspace):
        os.makedirs(IA_workspace)
        
    inputs_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Iowa_watershed_extent"

    IA_SDR_args = {
                u'biophysical_table_uri': '',
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
                u'workspace_dir': IA_workspace,
        }
    files = [f for f in os.listdir(IA_scenario_dir) if
                              os.path.isfile(os.path.join(IA_scenario_dir, f))]
    tifs = [f for f in files if re.search('.tif$', f)]
    IA_scenarios = [os.path.join(IA_scenario_dir, f) for f in tifs]

    for scenario in IA_scenarios:
        IA_SDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        
        # run low intensity for this scenario
        # suffix = lulc_name + '_low'
        # IA_SDR_args[u'biophysical_table_uri'] = low_biophys
        # IA_SDR_args[u'results_suffix'] = suffix
        # natcap.invest.sdr.sdr.execute(IA_SDR_args)
        
        # run high intensity for this scenario
        # suffix = lulc_name + '_high'
        # IA_SDR_args[u'biophysical_table_uri'] = high_biophys
        # IA_SDR_args[u'results_suffix'] = suffix
        # if not os.path.isfile(os.path.join(IA_SDR_args[u'workspace_dir'], 'output',
                                           # 'watershed_results_sdr_%s.shp' % 
                                           # suffix)):
            # print "******* executing SDR scenario " + lulc_name
            # natcap.invest.sdr.sdr.execute(IA_SDR_args)
    
    # NDR Iowa
    IA_workspace = os.path.join(result_dir, "IA_NDR")
    if not os.path.exists(IA_workspace):
        os.makedirs(IA_workspace)
        
    inputs_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Iowa_watershed_extent"
    biophys = r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Final per scenario\biophysical_coeffs_Iowa_Unilever_Corn_ext-int.csv"
    
    IA_NDR_args = {
            u'biophysical_table_uri': biophys,
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
            u'workspace_dir': IA_workspace,
        }
    files = [f for f in os.listdir(IA_scenario_dir) if
                              os.path.isfile(os.path.join(IA_scenario_dir, f))]
    tifs = [f for f in files if re.search('.tif$', f)]
    IA_scenarios = [os.path.join(IA_scenario_dir, f) for f in tifs]
    for scenario in IA_scenarios:
        IA_NDR_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        
        suffix = lulc_name
        IA_NDR_args[u'results_suffix'] = suffix
        if not os.path.isfile(os.path.join(IA_NDR_args[u'workspace_dir'], 'output',
                                           'watershed_results_ndr_%s.shp' % 
                                           suffix)):
            print "******* executing NDR Iowa scenario " + lulc_name
            # natcap.invest.ndr.ndr.execute(IA_NDR_args)

    ## NDR Mato Grosso
    MT_scenario_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\scenarios_Oct5\MT_watershed_extent"
    MT_workspace = os.path.join(result_dir, "MT_NDR")
    if not os.path.exists(MT_workspace):
        os.makedirs(MT_workspace)

    inputs_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\MatoGrosso_watershed_extent"
    biophys_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Final per scenario"
    MT_args = {
            u'biophysical_table_uri': '',
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
            u'workspace_dir': MT_workspace,
        }
    files = [f for f in os.listdir(MT_scenario_dir) if
                              os.path.isfile(os.path.join(MT_scenario_dir, f))]
    tifs = [f for f in files if re.search('.tif$', f)]
    MT_scenarios = [os.path.join(MT_scenario_dir, f) for f in tifs]

    for scenario in MT_scenarios:
        MT_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        suffix = lulc_name
        scen_name = re.search('_(.+?).tif',
                              os.path.basename(scenario)).group(1)
        if scen_name == "2007_baseline":
            biophys_name = "biophysical_coeffs_Brazil_Unilever_sugarcane_Sce1b.csv"
        else:
            biophys_name = "biophysical_coeffs_Brazil_Unilever_sugarcane_Sce" + \
                            scen_name + ".csv"
            continue
        biophys = os.path.join(biophys_dir, biophys_name)
        MT_args[u'biophysical_table_uri'] = biophys
        MT_args[u'results_suffix'] = suffix
        if not os.path.isfile(os.path.join(MT_args[u'workspace_dir'], 'output',
                                           'watershed_results_ndr_%s.shp' % 
                                           suffix)):
            print "******* executing NDR Mato Grosso scenario " + lulc_name
            natcap.invest.ndr.ndr.execute(MT_args)

    ## SDR Mato Grosso
    MT_workspace = os.path.join(result_dir, "MT_SDR")
    if not os.path.exists(MT_workspace):
        os.makedirs(MT_workspace)

    inputs_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\MatoGrosso_watershed_extent"

    MT_args = {
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
                u'workspace_dir': MT_workspace,
        }
    high_biophys = os.path.join(inputs_dir,
                       'biophysical_coeffs_Brazil_Unilever_sugarcane_High.csv')
    low_biophys = os.path.join(inputs_dir, 
                        'biophysical_coeffs_Brazil_Unilever_sugarcane_Low.csv')

    files = [f for f in os.listdir(MT_scenario_dir) if
                              os.path.isfile(os.path.join(MT_scenario_dir, f))]
    tifs = [f for f in files if re.search('.tif$', f)]
    MT_scenarios = [os.path.join(MT_scenario_dir, f) for f in tifs]

    for scenario in MT_scenarios:
        MT_args[u'lulc_uri'] = scenario
        lulc_name = os.path.basename(scenario)[3:-4]
        
        # run low intensity for this scenario
        # suffix = lulc_name + '_low'
        # MT_args[u'biophysical_table_uri'] = low_biophys
        # MT_args[u'results_suffix'] = suffix
        # if not os.path.isfile(os.path.join(MT_args[u'workspace_dir'], 'output',
                                           # 'watershed_results_sdr_%s.shp' % 
                                           # suffix)):
            # natcap.invest.sdr.sdr.execute(MT_args)
        
        # run high intensity for this scenario
        suffix = lulc_name + '_high'
        MT_args[u'biophysical_table_uri'] = high_biophys
        MT_args[u'results_suffix'] = suffix
        if not os.path.isfile(os.path.join(MT_args[u'workspace_dir'], 'output',
                                           'watershed_results_sdr_%s.shp' % 
                                           suffix)):
            print "******* executing SDR scenario " + lulc_name
            # natcap.invest.sdr.sdr.execute(MT_args)
