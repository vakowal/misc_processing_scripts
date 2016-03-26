"""
Run NDR for MT scenario 3a with new biophysical tables, 12.10.15
"""

import os
import sys
import re
import natcap.invest.ndr.ndr

if __name__ == '__main__':
    result_dir = "C:/Users/Ginger/Documents/NatCap/GIS_local/Unilever/Batched_runs_results"
    MT_workspace = os.path.join(result_dir, "MT_NDR")
    if not os.path.exists(MT_workspace):
        os.makedirs(MT_workspace)

    inputs_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/MatoGrosso_watershed_extent"

    MT_args = {
            u'biophysical_table_uri': '',
            u'calc_n': True,
            u'calc_p': False,
            u'dem_uri': os.path.join(inputs_dir, 'dem_180_fill.tif'),
            u'k_param': u'2',
            u'lulc_uri': '',
            u'results_suffix': '',
            u'subsurface_critical_length_n': u'150',
            u'subsurface_eff_n': u'0.8',
            u'threshold_flow_accumulation': u'600',
            u'watersheds_uri': os.path.join(inputs_dir,
                                            'hydro_1k_dissolve.shp'),
            u'workspace_dir': MT_workspace,
        }
    biophys_norm = os.path.join(inputs_dir,
                   'biophysical_coeffs_Brazil_Unilever_sugarcane_Int_orig.csv')
    biophys_18 = os.path.join(inputs_dir, 
                'biophysical_coeffs_Brazil_Unilever_sugarcane_Int_class18.csv')
    s3a_norm = "C:/Users/Ginger/Dropbox/NatCap_backup/Unilever/Unilever_model_inputs_10.6.15/scenarios_Oct5/MT_watershed_extent/MT_3a.tif"
    s3a_18 = "C:/Users/Ginger/Documents/NatCap/GIS_local/Unilever/Reclassify_MT_3a_NDR/MT_3a_ndr_18.tif"

    MT_args[u'lulc_uri'] = s3a_norm
    suffix = '3a_12.10.15'
    MT_args[u'biophysical_table_uri'] = biophys_norm
    MT_args[u'results_suffix'] = suffix
    print "******* executing NDR scenario " + suffix
    natcap.invest.ndr.ndr.execute(MT_args)
    
    MT_args[u'lulc_uri'] = s3a_18
    suffix = '3a_class18_12.10.15'
    MT_args[u'biophysical_table_uri'] = biophys_18
    MT_args[u'results_suffix'] = suffix
    print "******* executing NDR scenario " + suffix
    natcap.invest.ndr.ndr.execute(MT_args)
