### SDR calibration for Joanna

from tempfile import mkstemp
import os
import pandas
import shutil
from arcpy import *
import natcap.invest.sdr.sdr

if __name__ == '__main__':
    exp_design = "C:/Users/Ginger/Desktop/SDR_calibration_exp_design.csv"
    exp_df = pandas.read_csv(exp_design)
    
    workspace = "F:/GIS/Joanna/SDR_calibration"
    summary_table = "C:/Users/Ginger/Desktop/Joanna_SDR_results.csv"
    summary_dict = pandas.read_csv(summary_table).to_dict(orient='list')
    ws_id_list = summary_dict['Watershed_ID']

    before_lulc = u'C:/Users/Ginger/Documents/NatCap/GIS_local/Joanna/LULC_Salinas_CalcStats2015May13/LULC_GK/ais05nlcd06.tif'
    after_lulc = u'C:/Users/Ginger/Documents/NatCap/GIS_local/Joanna/LULC_Salinas_CalcStats2015May13/LULC_GK/ais12nlcd11.tif'
    
    for row in xrange(len(exp_df)):
        run = str(int(exp_df.iloc[row].run))
        k_param = exp_df.iloc[row].k_param
        ic_0 = exp_df.iloc[row].ic_0
        
        # run the model
        SDR_args = {
            u'biophysical_table_uri': u'C:\\Users\\Ginger\\Documents\\NatCap\\GIS_local\\Joanna\\Salinas_erosivity_erodibility_coeffs\\Salinas_JLN\\BiophysicalTable_AISmergeNLCD_JLN_2015feb5.csv',
            u'dem_uri': u'C:/Users/Ginger/Documents/NatCap/GIS_local/Joanna/dem/dem_prepSalinas.tif',
            u'drainage_uri': u'',
            u'erodibility_uri': u'C:\\Users\\Ginger\\Documents\\NatCap\\GIS_local\\Joanna\\Salinas_erosivity_erodibility_coeffs\\Salinas_JLN\\erod_mask_ws.tif',
            u'erosivity_uri': u'C:\\Users\\Ginger\\Documents\\NatCap\\GIS_local\\Joanna\\Salinas_erosivity_erodibility_coeffs\\erosivity\\eros_buff_ws.tif',
            u'ic_0_param': ic_0,
            u'k_param': k_param,
            u'lulc_uri': after_lulc,
            u'results_suffix': '%s_after' % run,
            u'sdr_max': u'0.8',
            u'threshold_flow_accumulation': u'3000',
            u'watersheds_uri': u'C:\\Users\\Ginger\\Documents\\NatCap\\GIS_local\\Joanna\\Salinas_erosivity_erodibility_coeffs\\SalinasHUC12prj_shp\\SalHUC12prj.shp',
            u'workspace_dir': workspace,
        }
        natcap.invest.sdr.sdr.execute(SDR_args)
    
        result_dict = {}
        output_shp = os.path.join(workspace, 'output',
                                  'watershed_results_sdr_%s_after.shp' % run)
        fields = ['ws_id', 'sed_export']
        with da.SearchCursor(output_shp, fields) as cursor:
            for row in cursor:
                ws_id = row[0]
                sed_export = row[1]
                if ws_id in ws_id_list:
                    result_dict[ws_id] = sed_export
        current_results = [result_dict[ws_id] for ws_id in
                           summary_dict['Watershed_ID']]
        # post-hoc analysis: divide export by catchment area
        # write to summary table
        summary_dict['run_%s_11-12' % run] = current_results
        df = pandas.DataFrame(summary_dict)
        indexed = df.set_index(['Watershed_ID'])
        indexed.to_csv(summary_table)
