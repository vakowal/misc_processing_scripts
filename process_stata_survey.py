# process Peru household survey data

import os
import re
import pandas as pd

# label_file = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Household_survey_livestock_portion\Labels.txt"

stata_file = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Household_survey_livestock_portion\Database livestock.dta"
df = pd.read_stata(stata_file)

forage_quantity_cols = [f for f in df.columns.values if re.search('^l_13', f)]
pasture_size_ha_cols = [f for f in df.columns.values if re.search('^l_10', f)]
veg_type_cols = [f for f in df.columns.values if re.search('^l_8', f)]
pasture_name_cols = ['l_8_1', 'l_8_2', 'l_8_3', 'l_8_4', 'l_8_5']

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
