## A script giving example of how to run CENTURY from a batch file
# Ginger Kowal, Natural Capital Project
# May 15 2015

import os
import sys
import shutil
from subprocess import Popen

def write_century_bat(century_dir, century_bat, schedule, output, fix_file,
    outvars, extend = None):
    """Write the batch file to run CENTURY"""
    
    if schedule[-4:] == '.sch':
        schedule = schedule[:-4]
    if output[-4:] == '.lis':
        output = output[:-4]
    
    with open(os.path.join(century_dir, century_bat), 'wb') as file:
        file.write('erase ' + output + '.bin\n')
        file.write('erase ' + output + '.lis\n\n')
        
        file.write('copy fix.100 fix_orig.100\n')
        file.write('erase fix.100\n\n')
        
        file.write('copy ' + fix_file + ' fix.100\n')
        file.write('erase ' + output + '_log.txt\n\n')
        
        if extend is not None:
            file.write('century_46 -s ' + schedule + ' -n ' + output + ' -e ' +
                extend + ' > ' + output + '_log.txt\n')
        else:
            file.write('century_46 -s ' + schedule + ' -n ' + output + ' > ' + 
                output + '_log.txt\n')
        file.write('list100_46 ' + output + ' ' + output + ' ' + outvars +
            '\n\n')
        
        file.write('erase fix.100\n')
        file.write('copy fix_orig.100 fix.100\n')
        file.write('erase fix_orig.100\n')

def read_CENTURY_outputs(cent_file, first_year, last_year):
    """Read biomass outputs from CENTURY for each month of CENTURY output within
    the specified range (between 'first_year' and 'last_year')."""
    
    cent_df = pandas.io.parsers.read_fwf(cent_file, skiprows = [1])
    df_subset = cent_df[(cent_df.time >= first_year) & (cent_df.time < last_year + 1)]
    biomass = df_subset[['time', 'aglivc', 'stdedc', 'aglive(1)', 'stdede(1)']]
    aglivc = biomass.aglivc * 2.5  # live biomass
    biomass.aglivc = aglivc
    stdedc = biomass.stdedc * 2.5  # standing dead biomass
    biomass.stdedc = stdedc
    biomass['total'] = biomass.aglivc + biomass.stdedc
    aglive1 = biomass[['aglive(1)']] * 6.25  # crude protein in  live material
    biomass['aglive1'] = aglive1
    stdede1 = biomass[['stdede(1)']] * 6.25  # crude protein in standing dead
    biomass['stdede1'] = stdede1
    biomass_indexed = biomass.set_index('time')
    return biomass_indexed
    
# inputs        
century_dir = 'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
graz_file = os.path.join(century_dir, 'graz.100')
fix_file = 'drytrpfi.100'
outvars = 'outvars.txt'

# A dictionary holding all inputs describing a grass type;
grass1 = {
    'label': 'grass1',
    'type': 'C4',  # 'C3' or 'C4'
    'percent_biomass': 0.5,  # initial percent biomass
    'DMD_green': 0.6,
    'DMD_dead': 0.4,
    'cprotein_green': .1,
    'cprotein_dead': 0.,
    'green_gm2': 0.,  # important that these are 0
    'dead_gm2': 0.,
    'prev_g_gm2': 0.,
    'prev_d_gm2': 0.,
    }

grass_list = [grass1]
schedule_list = []
for grass in grass_list:
    schedule = os.path.join(century_dir, (grass['label'] + '.sch'))
    if os.path.exists(schedule):
        schedule_list.append(schedule)
    else:
        er = "Error: schedule file not found"
        print er
        sys.exit(er)
    # write CENTURY bat
    hist_bat = os.path.join(century_dir, (grass['label'] + '_hist.bat'))
    hist_schedule = grass['label'] + '_hist.sch'
    hist_output = grass['label'] + '_hist'
    write_century_bat(century_dir, hist_bat, hist_schedule, hist_output,
        fix_file, outvars)
        
# make a copy of the original graz params and schedule file
shutil.copyfile(graz_file, os.path.join(century_dir, 'graz_orig.100'))
for schedule in schedule_list:
    label = os.path.basename(schedule)[:-4]
    copy_name = label + '_orig.100'
    shutil.copyfile(schedule, os.path.join(century_dir, copy_name))

# run CENTURY
for grass in grass_list:
    hist_bat = os.path.join(century_dir, (grass['label'] + '_hist.bat'))
    century_bat = os.path.join(century_dir, (grass['label'] + '.bat'))
    p = Popen(["cmd.exe", "/c " + hist_bat], cwd = century_dir)
    stdout, stderr = p.communicate()
    p = Popen(["cmd.exe", "/c " + century_bat], cwd = century_dir)
    stdout, stderr = p.communicate()

try:
    # get biomass and crude protein for each grass type from CENTURY
    for grass in grass_list:
        output_file = os.path.join(century_dir, (grass['label'] + '.lis'))
        outputs = read_CENTURY_outputs(output_file, start_year,
                                            start_year + 2)
        target_month = cent.find_prev_month(year, month)
        grass['prev_g_gm2'] = grass['green_gm2']
        grass['prev_d_gm2'] = grass['dead_gm2']
        grass['green_gm2'] = outputs.loc[target_month, 'aglivc']
        grass['dead_gm2'] = outputs.loc[target_month, 'stdedc']
        if not user_define_protein:
            grass['cprotein_green'] = (outputs.loc[target_month, 'aglive1']
                / outputs.loc[target_month, 'aglivc'])
            grass['cprotein_dead'] = (outputs.loc[target_month, 'stdede1']
                / outputs.loc[target_month, 'stdedc'])

        ## here is where you would modify inputs to CENTURY - I modify the 
        ## original schedule and parameter files so that you can call CENTURY
        ## from the same batch file
        
        # call CENTURY from the batch file
        century_bat = os.path.join(century_dir, (grass['label'] + '.bat'))
        p = Popen(["cmd.exe", "/c " + century_bat], cwd = century_dir)
        stdout, stderr = p.communicate()
        
finally:
    # replace graz params used by CENTURY with original file      
    os.remove(graz_file)
    shutil.copyfile(os.path.join(century_dir, 'graz_orig.100'), graz_file)
    os.remove(os.path.join(century_dir, 'graz_orig.100'))
    for schedule in schedule_list:
        # replace schedule files used by CENTURY with original files
        os.remove(schedule)
        label = os.path.basename(schedule)[:-4]
        copy_name = label + '_orig.100'
        shutil.copyfile(os.path.join(century_dir, copy_name), schedule)
        os.remove(os.path.join(century_dir, copy_name))
