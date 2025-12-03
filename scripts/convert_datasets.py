#!/usr/bin/env python3
"""
Script to convert datasets from dataset/default to TFB format in dataset/research

TFB Expected Format:
- CSV file with columns: date, data, cols
- date: timestamp in format like "2015-07-01 12:00:00"
- data: the actual time series value
- cols: channel name (e.g., "channel_1")

Input Format:
- Plain text files with one value per line (no headers)
- Or Excel files that need to be converted
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_txt_to_tfb_format(input_path, output_path, dataset_name):
    """
    Convert a plain text file with one value per line to TFB CSV format.
    
    Args:
        input_path: Path to input .txt file
        output_path: Path to output .csv file
        dataset_name: Name of the dataset (used for channel naming)
    """
    try:
        # Read the data
        with open(input_path, 'r') as f:
            values = []
            for line in f:
                line = line.strip()
                if not line or line.lower() == 'data':
                    continue
                # Remove trailing commas and convert to float
                line = line.rstrip(',')
                try:
                    values.append(float(line))
                except ValueError:
                    continue
        
        # Create a DataFrame in TFB format
        # Generate timestamps - using daily frequency as default
        # Starting from 2000-01-01 as a reasonable default
        start_date = datetime(2000, 1, 1, 12, 0, 0)
        dates = [start_date + timedelta(days=i) for i in range(len(values))]
        
        df = pd.DataFrame({
            'date': dates,
            'data': values,
            'cols': 'channel_1'
        })
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Converted {dataset_name}: {len(values)} data points")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {dataset_name}: {e}")
        return False


def convert_tsv_to_tfb_format(input_path, output_path, dataset_name):
    """
    Convert a tab-separated file with dates as columns to TFB CSV format.
    
    Args:
        input_path: Path to input .xls/.txt file (tab-separated)
        output_path: Path to output .csv file
        dataset_name: Name of the dataset
    """
    try:
        # Read the tab-separated file
        df = pd.read_csv(input_path, sep='\t', encoding='utf-8')
        
        # Skip the first two columns (Brasil and description) and get the data
        # The columns from index 2 onwards are dates with values
        date_cols = df.columns[2:]
        values = df.iloc[0, 2:].values
        
        # Parse dates from column names (format: jan/00, fev/00, etc.)
        month_map = {
            'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
            'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
        }
        
        dates = []
        for col in date_cols:
            try:
                month_str, year_str = col.split('/')
                month = month_map.get(month_str.lower(), 1)
                year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
                dates.append(datetime(year, month, 1, 12, 0, 0))
            except:
                # If parsing fails, use sequential dates
                if dates:
                    last_date = dates[-1]
                    dates.append(last_date + timedelta(days=30))
                else:
                    dates.append(datetime(2000, 1, 1, 12, 0, 0))
        
        # Create DataFrame in TFB format
        df_tfb = pd.DataFrame({
            'date': dates,
            'data': values,
            'cols': 'channel_1'
        })
        
        # Save to CSV
        df_tfb.to_csv(output_path, index=False)
        logger.info(f"Converted {dataset_name}: {len(values)} data points")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {dataset_name}: {e}")
        return False


def convert_xls_to_tfb_format(input_path, output_path, dataset_name):
    """
    Convert an Excel file to TFB CSV format.
    
    Args:
        input_path: Path to input .xls/.xlsx file
        output_path: Path to output .csv file
        dataset_name: Name of the dataset
    """
    try:
        # Read the Excel file - try different engines for old Excel formats
        try:
            df_excel = pd.read_excel(input_path, engine='openpyxl')
        except:
            try:
                df_excel = pd.read_excel(input_path, engine='xlrd')
            except:
                df_excel = pd.read_excel(input_path)
        
        # Assume the first column contains dates and subsequent columns contain data
        # This is a common format, but may need adjustment based on actual file structure
        
        if df_excel.shape[1] == 1:
            # Single column of values, no dates
            values = df_excel.iloc[:, 0].values
            start_date = datetime(2000, 1, 1, 12, 0, 0)
            dates = [start_date + timedelta(days=i) for i in range(len(values))]
            
            df = pd.DataFrame({
                'date': dates,
                'data': values,
                'cols': 'channel_1'
            })
        else:
            # Multiple columns - assume first is date, rest are data
            # For simplicity, we'll take the second column as the main data
            dates = pd.to_datetime(df_excel.iloc[:, 0])
            values = df_excel.iloc[:, 1].values
            
            df = pd.DataFrame({
                'date': dates,
                'data': values,
                'cols': 'channel_1'
            })
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Converted {dataset_name}: {len(values)} data points")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {dataset_name}: {e}")
        return False


def main():
    """Main conversion function"""
    # Define paths
    input_dir = 'dataset/default'
    output_dir = 'dataset/research'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in the input directory
    files = os.listdir(input_dir)
    
    converted_count = 0
    failed_count = 0
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        
        # Skip if not a file
        if not os.path.isfile(input_path):
            continue
        
        # Get file extension
        name, ext = os.path.splitext(filename)
        
        # Determine output filename
        output_filename = f"{name}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert based on file type
        if ext == '.txt':
            success = convert_txt_to_tfb_format(input_path, output_path, name)
        elif ext in ['.xls', '.xlsx']:
            # Check if it's actually a tab-separated text file
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        # It's a tab-separated file, treat as text
                        success = convert_tsv_to_tfb_format(input_path, output_path, name)
                    else:
                        success = convert_xls_to_tfb_format(input_path, output_path, name)
            except:
                success = convert_xls_to_tfb_format(input_path, output_path, name)
        else:
            logger.warning(f"Skipping unsupported file type: {filename}")
            continue
        
        if success:
            converted_count += 1
        else:
            failed_count += 1
    
    logger.info(f"\nConversion complete!")
    logger.info(f"Successfully converted: {converted_count} files")
    logger.info(f"Failed: {failed_count} files")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()