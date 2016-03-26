# process Peru household survey data

import os
import re
import pandas as pd

# label_file = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Household_survey_livestock_portion\Labels.txt"



# forage_quantity_cols = [f for f in df.columns.values if re.search('^l_13', f)]
# pasture_size_ha_cols = [f for f in df.columns.values if re.search('^l_10', f)]
# veg_type_cols = [f for f in df.columns.values if re.search('^l_8', f)]
# pasture_name_cols = ['l_8_1', 'l_8_2', 'l_8_3', 'l_8_4', 'l_8_5']

def retrieve_months_grazed(df):
    """Make a table containing the months grazed for each pasture."""
    
    combID_list = []
    months_list = []
    for row in xrange(len(df)):
        HH_ID = df.iloc[row].id_hogar
        for i in [1, 2, 3, 4, 5]:
            col = 'l_8_' + str(i)
            if df.iloc[row][col] != '':
                combID_list.append('%i_%s' % (HH_ID, df.iloc[row][col]))
                months = []
                for m in range(1, 14):
                    col = 'l_11_%d_%d' % (i, m)
                    try:
                        if df.iloc[row][col] == 1:
                            months.append(m)
                    except KeyError:
                        continue
                months_list.append(months)
    months_dict = {'comb_id': combID_list,
                   'months': months_list
                   }
    m_df = pd.DataFrame(months_dict)
    m_df.set_index(['comb_id'], inplace=True)
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Household_survey_livestock_portion\months_grazed_table.csv"
    m_df.to_csv(save_as)

def retrieve_stocking_density(df):
    """Make a table with total reported area across pastures and total number
    of animals of each type, for each respondent in the survey."""
    
    HH_ID_list = []
    total_area_list = []
    bull_list = []
    cow_list = []
    calf_list = []
    sheep_list = []
    camelid_list = []
    other_list = []
    for row in xrange(len(df)):
        HH_ID = df.iloc[row].id_hogar
        area_sum = 0.
        bull_sum = 0.
        cow_sum = 0.
        calf_sum = 0.
        sheep_sum = 0.
        camelid_sum = 0.
        other_sum = 0.
        for i in [1, 2, 3, 4, 5]:
            area_col = 'l_10_' + str(i)
            if pd.notnull(df.iloc[row][area_col]) and \
                          df.iloc[row][area_col] != 888:
                area_sum = area_sum + df.iloc[row][area_col]
            animal_col = 'm_1_' + str(i)
            if df.iloc[row][animal_col] != '':
                animal_name = df.iloc[row][animal_col]
                num_col = 'm_3_' + str(i)
                if animal_name == 'Ovinos':
                    sheep_sum = sheep_sum + df.iloc[row][num_col]
                elif animal_name == 'Toros':
                    bull_sum = bull_sum + df.iloc[row][num_col]
                elif animal_name == 'Vacas':
                    cow_sum = cow_sum + df.iloc[row][num_col]
                elif animal_name == 'Terneros':
                    calf_sum = calf_sum + df.iloc[row][num_col]
                elif animal_name == 'Otro':
                    other_sum = other_sum + df.iloc[row][num_col]    
                elif animal_name in ['Alpacas', 'Llamas']:
                    camelid_sum = camelid_sum + df.iloc[row][num_col]
                else:
                    raise Exception("Animal type not recognized: %s" %
                                    animal_name)
        HH_ID_list.append(HH_ID)
        total_area_list.append(area_sum)
        bull_list.append(bull_sum)
        cow_list.append(cow_sum)
        calf_list.append(calf_sum)
        sheep_list.append(sheep_sum)
        camelid_list.append(camelid_sum)
        other_list.append(other_sum)
   
    sd_dict = {'HH_ID': HH_ID_list,
               'total_area_ha': total_area_list,
               'bulls': bull_list,
               'cows': cow_list,
               'calves': calf_list,
               'sheep': sheep_list,
               'camelids': camelid_list,
               'others': other_list,
               }
    sd_df = pd.DataFrame(sd_dict)
    sd_df.set_index(['HH_ID'], inplace=True)
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Household_survey_livestock_portion\stocking_density_table.csv"
    sd_df.to_csv(save_as)

def retrieve_pasture_quantity(df):
    """Make a table of reported pasture quality for each month, for each
    pasture in the survey."""
    
    combID_list = []
    interviewee_name_list = []
    quantity_list = []
    for row in xrange(len(df)):
        HH_ID = df.iloc[row].id_hogar
        for i in [1, 2, 3, 4, 5]:
            col = 'l_8_' + str(i)
            if df.iloc[row][col] != '':
                combID_list.append('%i_%s' % (HH_ID, df.iloc[row][col]))
                interviewee_name_list.append(df.iloc[row]['entrevistado'])
                forage_quantity_cols = [f for f in df.columns.values if
                                        re.search('^l_13_%i' % i, f)]
                quantity_list.append(df.iloc[row][forage_quantity_cols].values)

    quantity_dict = {'HH_P_ID_survey': combID_list,
                     'interviewee_name': interviewee_name_list,
                     'Jan': [i[0] for i in quantity_list],
                     'Feb': [i[1] for i in quantity_list],
                     'Mar': [i[2] for i in quantity_list],
                     'Apr': [i[3] for i in quantity_list],
                     'May': [i[4] for i in quantity_list],
                     'Jun': [i[5] for i in quantity_list],
                     'Jul': [i[6] for i in quantity_list],
                     'Aug': [i[7] for i in quantity_list],
                     'Sep': [i[8] for i in quantity_list],
                     'Oct': [i[9] for i in quantity_list],
                     'Nov': [i[10] for i in quantity_list],
                     'Dec': [i[11] for i in quantity_list],
                     }
    quantity_df = pd.DataFrame(quantity_dict)
    quantity_df.set_index(['HH_P_ID_survey'], inplace=True)
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Household_survey_livestock_portion\quantity_table.csv"
    quantity_df.to_csv(save_as)

if __name__ == "__main__":    
    stata_file = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Household_survey_livestock_portion\Database livestock.dta"
    df = pd.read_stata(stata_file)
    retrieve_months_grazed(df)