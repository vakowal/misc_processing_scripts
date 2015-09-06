## Process meteorological data in text file from IDEAM (Colombia)

import os

directory = 'C:\\Users\\GingerBelle\\Dropbox\\NatCap_backup\\Forage_model\\Data\\Colombia'
name = 'IDEAM.txt'
data_path = os.path.join(directory, name)
outdir = 'C:\\Users\\GingerBelle\\Desktop\\Temp'

def get_data(data_path, outdir):
    variables = list()
    coords = list()
    years = list()
    index = 1
    with open(data_path, 'r') as data:
        for line in data:
            if 'I D E A M' in line:
                nextline = data.next()
                nextline = data.next()
                variable = nextline[:-19].strip()
                nextline = data.next()
                nextline = data.next()
                year = nextline[58:65].strip()
                nextline = data.next()
                nextline = data.next()
                if 'DEPTO      CAQUETA' in nextline:
                    latitude = nextline[14:21].strip()
                    nextline = data.next()
                    longitude = nextline[14:21].strip()
                    nextline = data.next()
                    nextline = data.next()
                    nextline = data.next()
                    variables.append(variable)
                    coords.append((latitude, longitude))
                    years.append(year)
                    name = str(index) + '.txt'
                    out_path = os.path.join(outdir, name)
                    index = index + 1
                    with open(out_path, 'w') as out:
                        out.write(variable + '   ' + year + '\n')
                        for c in xrange(35):
                            out.write(nextline)
                            nextline = data.next()
    results_dict = {'coords': coords, 'variables': variables, 'years': years}
    return results_dict

def write_results(results_dict, directory):
    out_name = os.path.join(directory, 'coordinates.txt')
    with open(out_name, 'w') as out:
        for index in xrange(len(results_dict['coords'])):
            out.write(str(index + 1) + '    ' + str(results_dict['coords'][index]) + '\n')
    out_name = os.path.join(directory, 'variables.txt')
    with open(out_name, 'w') as out:    
        for index in xrange(len(results_dict['variables'])):
            out.write(str(index + 1) + '    ' + str(results_dict['variables'][index]) + '\n')
    out_name = os.path.join(directory, 'years.txt')
    with open(out_name, 'w') as out:    
        for index in xrange(len(results_dict['years'])):
            out.write(str(index + 1) + '    ' + str(results_dict['years'][index]) + '\n')
            
def index_coords(file_path):
    lat_coords = list()
    long_coords = list()
    coords = list()
    with open(file_path, 'r') as data_file:
        for line in data_file:
            if 'DEPTO      CAQUETA' in line:
                lat_coords.append(line[14:21])
                nextline = data_file.next()
                long_coords.append(nextline[14:21])
    coords = {'lat': lat_coords, 'long': long_coords}
    return coords

def get_departments(file_path):
    departments = list()
    with open(file_path, 'r') as data_file:
        for num, line in enumerate(data_file, 1):
            if 'DEPTO' in line:
                departments.append(line[74:90].strip())
    return departments

