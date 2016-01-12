"""
Run NDR for the Upper Mississippi River, 1.7.16
"""

import os
import natcap.invest.ndr.ndr

if __name__ == '__main__':
    result_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Floodplain\MS_project\NDR"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    biophys = r"C:\Users\Ginger\Dropbox\NatCap_backup\Floodplain\MS_Project\NDR\NASS_CDL_rmp_biophys.csv"
    
    UMR_NDR_args = {
            u'biophysical_table_uri': biophys,
            u'calc_n': True,
            u'calc_p': False,
            u'dem_uri': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Floodplain/MS_project/data/UMR_DEM_90m_fill.tif',
            u'k_param': u'2',
            u'lulc_uri': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Floodplain/MS_project/data/LULC/NASS_CDL_DEM_AOI_ext_rmp_prj.tif',
            u'results_suffix': '',
            u'subsurface_critical_length_n': u'1500',
            u'subsurface_eff_n': u'0.5',
            u'threshold_flow_accumulation': u'3000',
            u'watersheds_uri': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Floodplain/MS_project/data/StudyArea_HUC08_diss_proj.shp',
            u'workspace_dir': result_dir,
        }
    natcap.invest.ndr.ndr.execute(UMR_NDR_args)
