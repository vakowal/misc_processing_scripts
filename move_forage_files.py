# collect summary results for Perrine

import os
import shutil
import pandas

def move_summary_files():
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\raw_5.2.16"
    newdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\biomass_time_series"

    for folder in os.listdir(outerdir):
        summary_file = os.path.join(outerdir, folder, "summary_results.csv")
        new_path = os.path.join(newdir, folder + ".csv")
        if os.path.isfile(new_path):
            continue
        else:
            try:
                shutil.copyfile(summary_file, new_path)
            except IOError:
                continue
            
def erase_intermediate_files():
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\raw_5.2.16"
    for folder in os.listdir(outerdir):
        for file in os.listdir(os.path.join(outerdir, folder)):
            if file.endswith("summary_results.csv"):
                continue
            else:
                try:
                    object = os.path.join(outerdir, folder, file)
                    if os.path.isfile(object):
                        os.remove(object)
                    else:
                        shutil.rmtree(object)
                except OSError:
                    continue
 
def id_failed_simulations():
    failed = []
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/raw_5.2.16"
    for subbasin in range(1, 16):
        for anim_type in ['cow', 'sheep', 'camelid']:
            for sd_level in ['low', 'med', 'high']:
                out_dir = os.path.join(outer_dir, 's%d_%s_%s' %
                                       (subbasin, anim_type, sd_level))
                summary_csv = os.path.join(out_dir, "summary_results.csv")
                if os.path.isfile(summary_csv):
                    df = pandas.read_csv(summary_csv)
                    if df.shape[0] < 204:
                        failed.append('s%d_%s_%s' % (subbasin, anim_type,
                                                     sd_level))
    print "failed simulations: "
    for sim in failed:
        print sim
                    
                    
if __name__ == "__main__":
    move_summary_files()