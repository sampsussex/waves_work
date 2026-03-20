#!/usr/bin/env python3
"""
A script to check the fibinfos file for calibration targets and their G band magnitudes.
Usage:
python3 check_fibinfos_calibrator_magnitudes.py path_to_fibinfos_file

Note, on Typhon, you need to specify python3 in CL.
"""

import numpy as np
import argparse
from astropy.table import Table

def check_fibinfos(file):

    fibinfos = Table.read(file, hdu=1) # Load fibinfos


    # Check if boolean array has at least 1 true value
    if not np.any(fibinfos['FIB_USE'] == 4):
        print("No targets given calibration designation in FIBINFO table (FIB_USE==4)")
        return

    calibs = fibinfos[fibinfos['FIB_USE'] == 4] # Extract CALIB column
    
    spectros = ['HR', 'LRA', 'LRB']
    spectro_fib_roots = [1, 2, 3]
    for i in range(len(spectros)):
        spectro=spectros[i]
        spectro_fib_root = spectro_fib_roots[i]
        spectro_calibs = calibs[calibs['FIB_ROOT'] == spectro_fib_root]

        print(f"Number of {spectro} calibrators: {len(spectro_calibs)}")
       
        if len(spectro_calibs) == 0:
            continue

        print(f"Brighest G band estimate for {spectro} calibrators: {np.min(spectro_calibs['OBJ_GR'])}")

        print(f"All G band magnitudes for {spectro} calibrators:")
        print(spectro_calibs['OBJ_GR'])


def main():
    parser = argparse.ArgumentParser(description='Check fibinfos file for calibration targets')
    parser.add_argument('file', type=str, help='Path to fibinfos file')
    args = parser.parse_args()
    check_fibinfos(args.file)
    

if __name__ == "__main__":
    main()