import numpy as np
import pandas as pd
import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq


class MagneticumLightConeReader:
    """
    Memory-efficient reader for Magneticum light cone galaxy data files
    """
    
    def __init__(self, data_directory):
        """
        Initialize reader with path to data directory
        """
        self.data_dir = data_directory
        self.galaxy_files = glob.glob(os.path.join(data_directory, "wmap.*.galaxies.dat"))
        
        # Column names from README
        self.column_names = [
            'isub', 'l', 'b', 'rr', 'vmax', 'z_true', 'z_obs', 'Mstar', 'sfr',
            'u', 'V', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'K', 'L', 'M', 
            'Age', 'Z', 'flag'
        ]
        
        print(f"Found {len(self.galaxy_files)} galaxy data files")
    
    def inspect_single_file(self, filename, n_rows=10):
        """
        Inspect a single data file
        """
        print(f"\n=== INSPECTING {os.path.basename(filename)} ===")
        
        # Read first few lines as text
        with open(filename, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= n_rows:
                    break
                lines.append(line.strip())
        
        print("First few lines:")
        for i, line in enumerate(lines):
            print(f"  {i+1:2d}: {line}")
        
        # Count columns in data lines (skip comments)
        data_lines = [line for line in lines if line and not line.startswith('#')]
        if data_lines:
            col_counts = [len(line.split()) for line in data_lines]
            print(f"\nColumn counts: {col_counts}")
            print(f"Most common: {max(set(col_counts), key=col_counts.count)} columns")
        
        # Try pandas read
        try:
            test_data = pd.read_csv(filename, sep='\s+', nrows=5, header=None, comment='#')
            print(f"\nPandas shape: {test_data.shape}")
            print("First few rows:")
            print(test_data)
        except Exception as e:
            print(f"Pandas read error: {e}")
    
    def read_single_file(self, filename):
        """
        Read a single galaxy data file with error handling
        """
        try:
            # Read with error handling
            data = pd.read_csv(filename, sep='\s+', names=self.column_names, 
                             comment='#', header=None, low_memory=False)
            
            # Convert numeric columns
            numeric_cols = ['isub', 'l', 'b', 'rr', 'vmax', 'z_true', 'z_obs', 'Mstar', 'sfr',
                           'u', 'V', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'K', 'L', 'M', 'Age', 'Z']
            
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove invalid rows
            initial_len = len(data)
            data = data.dropna(subset=['z_true', 'Mstar'])
            
            if len(data) < initial_len:
                print(f"  Removed {initial_len - len(data)} invalid rows")
            
            return data
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None
    
    def process_to_parquet(self, output_file, z_min=0.01, z_max=0.4, mass_min=1e9):
        """
        Process all files and save to parquet
        """
        print(f"Processing {len(self.galaxy_files)} files to {output_file}")
        
        all_data = []
        total_galaxies = 0
        saved_galaxies = 0
        
        for i, filename in enumerate(self.galaxy_files):
            print(f"\nProcessing file {i+1}/{len(self.galaxy_files)}: {os.path.basename(filename)}")
            
            # Read file
            data = self.read_single_file(filename)
            if data is None:
                continue
            
            total_galaxies += len(data)
            print(f"  Read {len(data)} galaxies")
            
            # Apply cuts
            if 'z_true' in data.columns:
                data = data[(data['z_true'] >= z_min) & (data['z_true'] <= z_max)]
            
            if 'Mstar' in data.columns:
                data = data[data['Mstar'] >= mass_min]
            
            if len(data) > 0:
                saved_galaxies += len(data)
                all_data.append(data)
                print(f"  Kept {len(data)} galaxies after cuts")
            else:
                print(f"  No galaxies survived cuts")
            
            # Write intermediate results every few files to save memory
            if len(all_data) >= 5:  # Write every 5 files
                self._write_chunk_to_parquet(all_data, output_file)
                all_data = []
        
        # Write remaining data
        if all_data:
            self._write_chunk_to_parquet(all_data, output_file)
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total galaxies read: {total_galaxies:,}")
        print(f"Galaxies saved: {saved_galaxies:,}")
        print(f"Output file: {output_file}")
        
        return {'total_read': total_galaxies, 'total_saved': saved_galaxies}
    
    def _write_chunk_to_parquet(self, data_list, output_file):
        """
        Write a chunk of data to parquet file
        """
        if not data_list:
            return
        
        # Combine all dataframes
        chunk_data = pd.concat(data_list, ignore_index=True)
        print(f"  Writing {len(chunk_data)} galaxies to parquet...")
        
        # Convert to arrow table
        table = pa.Table.from_pandas(chunk_data)
        
        # Append to existing file or create new one
        if os.path.exists(output_file):
            existing_table = pq.read_table(output_file)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, output_file)
        else:
            pq.write_table(table, output_file)
    
    def read_parquet_sample(self, parquet_file, n_rows=1000):
        """
        Read a sample from the parquet file
        """
        if not os.path.exists(parquet_file):
            print(f"File {parquet_file} not found!")
            return None
        
        # Read metadata
        parquet_file_obj = pq.ParquetFile(parquet_file)
        total_rows = parquet_file_obj.metadata.num_rows
        
        print(f"\nParquet file info:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Columns: {parquet_file_obj.schema.names}")
        
        # Read sample
        sample = pq.read_table(parquet_file).to_pandas().head(n_rows)
        print(f"\nSample data ({len(sample)} rows):")
        print(sample[['isub', 'l', 'b', 'z_true', 'z_obs', 'Mstar', 'z']].head())
        
        return sample


def diagnose_files(data_directory):
    """
    Diagnose the data files
    """
    reader = MagneticumLightConeReader(data_directory)
    
    if len(reader.galaxy_files) == 0:
        print("No galaxy files found!")
        return
    
    # Inspect first file
    reader.inspect_single_file(reader.galaxy_files[0])
    
    # Test reading first file
    print(f"\n=== TESTING READ OF FIRST FILE ===")
    test_data = reader.read_single_file(reader.galaxy_files[0])
    if test_data is not None:
        print(f"Successfully read {len(test_data)} rows")
        print(f"Columns: {list(test_data.columns)}")
        print(f"Sample data:")
        if 'z_true' in test_data.columns and 'Mstar' in test_data.columns:
            print(test_data[['z_true', 'Mstar', 'z']].head())


def main():
    """
    Main execution
    """
    data_dir = "/Users/sp624AA/Downloads/OneEightSky_z0.45_box2b_hr"
    output_file = "magneticum_galaxies.parquet"
    
    print("=== DIAGNOSING DATA FILES ===")
    diagnose_files(data_dir)
    
    response = input("\nProceed with processing? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return
    
    print("\n=== PROCESSING ALL FILES ===")
    reader = MagneticumLightConeReader(data_dir)
    stats = reader.process_to_parquet(output_file)
    
    print("\n=== CHECKING OUTPUT ===")
    reader.read_parquet_sample(output_file)


if __name__ == "__main__":
    main()