#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import csv
#Simple code to reformat csv files from Sabine. 


def append_csv_files_as_polygons(folder_path, output_file):
    # List to store the polygons (one per CSV file)
    polygons = []
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # List to store ra and dec for the current file
            polygon_vertices = []
            
            # Open and read the CSV file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Read the header
                
                # Determine if there is an index column
                if header[0].lower() in ['index', '', '0']:
                    # If the first column is an index, RA is column 2, Dec is column 3
                    ra_col, dec_col = 1, 2
                else:
                    # If no index, RA is column 1, Dec is column 2
                    ra_col, dec_col = 0, 1
                
                # Extract only the ra and dec columns
                for row in reader:
                    ra, dec = row[ra_col], row[dec_col]
                    polygon_vertices.append(f"{ra} {dec}")
            
            # Join the ra and dec pairs with a space and treat them as one polygon
            polygons.append(' '.join(polygon_vertices))
    
    # Write the combined polygons to the output ASCII file, one per line
    with open(output_file, 'w', encoding='ascii') as output_f:
        output_f.write('\n'.join(polygons))

# Example usage:
folder_path = 'Masking/'  # Replace with the path to your folder
output_file = 'sabines_ngcs.dat'  # Output ASCII file

append_csv_files_as_polygons(folder_path, output_file)


# In[ ]:


#On these lines add an r infront, so it is the right way round for mangle: 
##rdmask: at line 57 of Sabines_NGCS/sabines_ngcs.dat: warning: polygon 0 may have its vertices ordered left- instead of right-handedly
#rdmask: at line 61 of Sabines_NGCS/sabines_ngcs.dat: warning: polygon 0 may have its vertices ordered left- instead of right-handedly
#rdmask: at line 71 of Sabines_NGCS/sabines_ngcs.dat: warning: polygon 0 may have its vertices ordered left- instead of right-handedly


# In[ ]:


#Also, polygon number 13 - at RA 12.76 DEC -34.048 has had the first point clipped from the original selection. The low separation between it and the final point seems to have been causing issues with MANGLE. Also have put an r infront for the same reason as above

