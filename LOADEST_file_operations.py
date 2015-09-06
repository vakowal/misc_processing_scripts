#### file operations to run LOADEST on empirical sediment and nutrient data
# for calibration of SDR and NDR for Joanna's Salinas valley project
# Ginger Kowal 9.6.15

from tempfile import mkstemp
from subprocess import Popen
import os
import re
import shutil

def write_control_files(calib_dir, const_name):
    """Write the necessary files to run LOADEST, given a folder that contains
    a bunch of LOADEST calibration files."""
    
    hdir = os.path.join(os.path.split(calib_dir)[0], 'loadest_header_files')
    if not os.path.exists(hdir):
        os.makedirs(hdir)
    cdir = os.path.join(os.path.split(calib_dir)[0], 'loadest_control_files')
    if not os.path.exists(cdir):
        os.makedirs(cdir)
        
    c_files = os.listdir(calib_dir)
    for file in c_files:
        site = file[:-4]
                
        # write header file
        fh, abs_path = mkstemp()
        with open(abs_path, 'wb') as new_file:
            title_line = 'site %s, constituent %s\n' % (site, const_name)
            new_file.write(title_line)
            new_file.write('# PRTOPT\n')
            new_file.write('1\n')  # estimated values print option
            new_file.write("# SEOPT\n")
            new_file.write('1\n')  # standard error option
            new_file.write("# LDOPT\n")
            new_file.write('2\n')  # monthly load option
            new_file.write('# MODNO\n')
            new_file.write('0\n')
            new_file.write('# NCONST\n')
            new_file.write('1\n')
            new_file.write('# CNAME, UCFLAG, ULFLAG\n')
            new_file.write('%-45s' % const_name)
            # both sdr and ndr are in mg/l
            new_file.write('1    ')  # UCFLAG
            new_file.write('4')    # ULFLAG
        hfile = os.path.join(hdir, site + '_h.inp')
        shutil.copyfile(abs_path, hfile)

        est_file = site + '_e.inp'
        
        fh, newpath = mkstemp()
        with open(newpath, 'wb') as c_file:
            c_file.write(site + '_h.inp\n')  # header file
            c_file.write(site + '.inp\n')    # calibration file
            c_file.write(site + '_e.inp')
        cfile = os.path.join(cdir, site + '_c.inp')
        shutil.copyfile(newpath, cfile)

def process_est_files(est_file_dir):
    """Add a single '1' to the top line of estimate files generated in R."""
    
    for file in os.listdir(est_file_dir):
        fh, newpath = mkstemp()
        filepath = os.path.join(est_file_dir, file)
        with open(newpath, 'wb') as newfile:
            with open(filepath, 'rb') as oldfile:
                newfile.write('1\n')
                for line in oldfile:
                    newfile.write(line)
        save_as = os.path.join(est_file_dir, file[:-4] + '_e.inp')
        shutil.copyfile(newpath, save_as)
        os.remove(filepath)
    
def run_LOADEST(input_dir, loadest_dir, const_name):
    """Move the relevant files into the local LOADEST directory and launch
    LOADEST."""
    
    result_dir = os.path.join(input_dir, 'loadest_results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
    # find the files
    est_file_dir = os.path.join(input_dir, 'loadest_est_files')
    h_file_dir = os.path.join(input_dir, 'loadest_header_files')
    calib_file_dir = os.path.join(input_dir, 'loadest_calib_files')
    control_file_dir = os.path.join(input_dir, 'loadest_control_files')
    
    site_list = [file[:-4] for file in os.listdir(calib_file_dir)]
    for site in site_list:
        # site = site_list[0]
        est_file = os.path.join(est_file_dir, site + '_e.inp')
        h_file = os.path.join(h_file_dir, site + '_h.inp')
        calib_file = os.path.join(calib_file_dir, site + '.inp')
        control_file = os.path.join(control_file_dir, site + '_c.inp')
        
        # move all files to loadest_dir
        shutil.copyfile(est_file, os.path.join(loadest_dir, site + '_e.inp'))
        shutil.copyfile(h_file, os.path.join(loadest_dir, site + '_h.inp'))
        shutil.copyfile(calib_file, os.path.join(loadest_dir, site + '.inp'))
        shutil.copyfile(control_file, os.path.join(loadest_dir, 'control.inp'))
        
        # run LOADEST
        p = Popen(["cmd.exe", "/c launch_loadest.bat"], cwd=loadest_dir)
        stdout, stderr = p.communicate()
        
        # copy results to original directory
        results_list = ['echo.out', '%s.ind' % const_name, '%s.out' % const_name,
                        '%s.res' % const_name]
        for rfile in results_list:
            copyto = os.path.join(result_dir, site + '_' + rfile)
            shutil.copyfile(os.path.join(loadest_dir, rfile), copyto)
            
        # delete input files and results files
        os.remove(os.path.join(loadest_dir, site + '_e.inp'))
        os.remove(os.path.join(loadest_dir, site + '_h.inp'))
        os.remove(os.path.join(loadest_dir, site + '.inp'))
        os.remove(os.path.join(loadest_dir, 'control.inp'))
        for rfile in results_list:
            os.remove(os.path.join(loadest_dir, rfile))
    
ndr_calib_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Joanna/NDR_calibration/Data/loadest_calib_files'
sdr_calib_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Joanna/SDR_calibration/CCAMP_data/loadest_calib_files"
const_name = 'Nitrogen'

ndr_input_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Joanna/NDR_calibration/Data'
ndr_const_name = 'Nitrogen'

ndr_est_file_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Joanna/NDR_calibration/Data/loadest_est_files"
sdr_est_file_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Joanna/SDR_calibration/CCAMP_data/loadest_est_files"

sdr_input_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Joanna/SDR_calibration/CCAMP_data'
sdr_const_name = 'Sediment'

loadest_dir = 'C:/Users/Ginger/Documents/NatCap/GIS_local/Joanna/loadest'

if __name__ == "__main__":
    # write_control_files(sdr_calib_dir, 'Sediment')
    run_LOADEST(sdr_input_dir, loadest_dir, sdr_const_name)