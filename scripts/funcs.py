import pandas as pd
import numpy as np
import pandas as pd
import cfunits
import chardet
import os
import seaborn as sns
import matplotlib.pyplot as plt

def clean_values(values):
    """
    Cleans the given pandas Index (or Series) of string values by:
    - Stripping leading/trailing whitespace
    - Removing stray quotes
    - Normalizing whitespace and commas
    - Joining related terms
    - Replacing Greek mu (μ) with micro sign (µ)

    Parameters:
        values (pd.Index or pd.Series): The values to be cleaned.

    Returns:
        pd.Index: The cleaned values.
    """
    # Helper function to replace Greek mu with micro sign
    def replace_mu_with_micro(s):
        mu = '\u03BC'  # Greek small letter mu (μ)
        micro = '\u00B5'  # Micro sign (µ)
        if isinstance(s, str):
            return s.replace(mu, micro)
        return s

    cleaned_values = (
        values
        .str.strip()  # Remove leading/trailing whitespace
        .str.replace(r"(^['\"]|['\"]$)", '', regex=True)  # Remove leading/trailing quotes
        .str.replace(r"\s*,\s*", ', ', regex=True)  # Normalize commas with single space after them
        .str.replace(r"\s+", ' ', regex=True)  # Normalize whitespace within strings
        .str.replace(r"' '", '', regex=True)  # Remove isolated single quotes
        .str.replace(r"([a-zA-Z]),([a-zA-Z])", r"\1, \2", regex=True)  # Add space after commas if missing
    )

    # Handle cases like ' evergreen, deciduous '
    cleaned_values = cleaned_values.str.replace(r"['‘’]", '', regex=True)  # Remove any stray single quotes
    cleaned_values = cleaned_values.str.replace(r"\b ,\b", ',', regex=True)  # Fix any stray commas

    # Replace Greek mu with micro sign
    cleaned_values = cleaned_values.map(replace_mu_with_micro)

    return cleaned_values

# Function to normalize all text in a DataFrame (for headers and data)
def encode_dataframe_values(df):
    """
    Normalizes text encoding in a pandas DataFrame for both headers and data.
    - Converts lone '1' strings to integer 1
    - Normalizes Unicode characters to NFC form

    Parameters:
        df (pd.DataFrame): The DataFrame to be normalized.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    import unicodedata
    import pandas as pd

    # Helper function to normalize character encoding for a single text value
    def normalize_encoding(text):
        if isinstance(text, str):
            if text.strip() == '1':  # Check for lone '1' (ignoring surrounding whitespace)
                return 1  # Convert to integer 1
            return unicodedata.normalize('NFKC', text)  # Normalize Unicode to NFC form
        return text

    # Ensure the DataFrame's multi-index (if present) is sorted
    if isinstance(df.columns, pd.MultiIndex) and not df.columns.is_monotonic_increasing:
        df = df.sort_index(axis=1)

    # Normalize each level in the column headers if it is a multi-index
    if isinstance(df.columns, pd.MultiIndex):
        normalized_columns = [
            tuple(normalize_encoding(level) for level in col) for col in df.columns
        ]
        df.columns = pd.MultiIndex.from_tuples(normalized_columns, names=df.columns.names)
    else:
        df.columns = [normalize_encoding(col) for col in df.columns]

    # Normalize each cell in the DataFrame without overwriting during iteration
    result_df = df.copy()  # Work on a copy to avoid issues while modifying
    for col in df.columns:
        result_df[col] = df[col].apply(lambda x: normalize_encoding(x) if isinstance(x, str) else x)

    return result_df

# Function to use the headers lookup table to normalize header names across future additional datasets
def standardize_headers(df, lookup_dict):
    # First normalize the DataFrame for consistent encoding
    df = encode_dataframe_values(df)

    standardized_columns = []
    # Iterate over the multi-header columns in the dataset
    for col in df.columns:
        # Extract the variable, description, and unit for each column
        var, desc, unit = col[0], col[1], col[2]
        lookup_key = (var, desc, unit)
        
        # Check if we have a match in the lookup dictionary
        if lookup_key in lookup_dict:
            
            # Replace with standardized values
            std_var = lookup_dict[lookup_key]['standard_variable']
            std_desc = lookup_dict[lookup_key]['standard_description']
            std_unit = lookup_dict[lookup_key]['standard_unit']
            
            # If the standardized description is 1, force the unit to 1, otherwise keep original unit
            if std_unit == 1:
                unit = 1
            elif pd.isna(unit):  # Explicitly check for NaN
                unit = std_unit

            standardized_columns.append((std_var, std_desc, unit))
        else:
            # If no match found, keep the original multi-header
            standardized_columns.append((var, desc, unit))

    # Set the new standardized multi-level columns
    df.columns = pd.MultiIndex.from_tuples(standardized_columns, names=['standard_variable', 'standard_description', 'unit'])
    
    # Step 1: Identify duplicate standard_variables
    duplicated_cols = df.columns[df.columns.duplicated(keep=False)]

    # Step 2: Merge duplicate columns
    merged_data = {}
    for col in duplicated_cols.unique():
        # Select all duplicate columns for the same standard_variable
        duplicate_columns = [c for c in df.columns if c[0] == col[0]]

        # Merge values row-wise, ignoring NaNs, deduplicating, and stripping whitespace
        # Ensure all values are converted to strings before applying string operations
        merged_data[col] = df[duplicate_columns].apply(
            lambda x: ' '.join(pd.unique(x.dropna().astype(str).str.strip())).strip(), axis=1
        )

    # Convert merged data into DataFrame
    merged_df = pd.DataFrame(merged_data)

    # Step 3: Drop original duplicate columns and add merged ones
    df = df.drop(columns=duplicated_cols.unique(), axis=1)
    df = pd.concat([df, merged_df], axis=1)

    return df

def visualize_all_columns(data):
        
        # Extract the column data
        data = data.dropna()
        
        if data.empty:
            # Skip empty columns
            print(f"Column {col} is empty, skipping visualization.")
        
        # Check if the data is numeric
        elif pd.api.types.is_numeric_dtype(data):
            # print('Numeric', col)
            # Detect outliers using IQR
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Min and max values
            min_val = data.min()
            max_val = data.max()
            
            # Count repeated values
            repeated = data[data.duplicated()]
            
            # Plot numeric data
            sns.displot(data, kde=True)
            plt.show()

        else:
            # if object/categorical data
            sns.countplot(data)
            plt.show()

def clean_and_standardize_datetime_string(date_str, special_case=None):
    """Remove extra spaces and handle range of dates or special cases."""
    if not isinstance(date_str, str):
        return date_str
    # Remove extra spaces and convert to lowercase
    date_str = date_str.strip().lower()
    return date_str

def parse_total_minutes(total_minutes):
    """Convert total minutes since some reference into %H:%M:%S format."""
    try:
        
        total_minutes = float(total_minutes)
        minutes_in_day = total_minutes % 1440
        
        # Convert to hours, minutes, and seconds
        hours = int(minutes_in_day // 60)
        remaining_minutes = minutes_in_day % 60
        minutes = int(remaining_minutes)
        seconds = int((remaining_minutes - minutes) * 60)
        parsed_time = pd.Timestamp(year=2000, month=1, day=1, hour=hours, minute=minutes, second=seconds)
        
        return parsed_time
        
    except Exception as e:
        return None

def parse_decimal_hours(decimal_hours):
    """Convert decimal hours into %H:%M:%S format."""
    try:
        decimal_hours = float(decimal_hours)
        hours = int(decimal_hours)
        minutes = int((decimal_hours - hours) * 60)
        seconds = int(((decimal_hours - hours) * 60 - minutes) * 60)
        
        parsed_time = pd.Timestamp(year=2000, month=1, day=1, hour=hours, minute=minutes, second=seconds)
        return parsed_time
        
    except Exception:
        return None

def parse_date_with_time(date_str, fmt):
    """Extract time component from a datetime string."""
    try:
        parsed_datetime = pd.to_datetime(date_str, format=fmt, errors='coerce')
        if not pd.isna(parsed_datetime):
            return parsed_datetime
    except Exception:
        return None

def process_datetime(row, formats):
    """Infer date and time from a string using source-specific formats."""
    try:
        # Preserve original date and time for comparison
        original_date = row[('date', 'Date of observation', 1)]
        original_time = row[('time', 'Time of observation', 1)]

        # Clean and standardize the datetime string
        date_str = clean_and_standardize_datetime_string(original_date)
        time_str = clean_and_standardize_datetime_string(original_time)

        # Initialize year, month, day as None
        year_str, month_str, day_str = None, None, None

        # (1) Parse date_str
        date_parsed = False  # Track whether parsing was successful
        parsed_date = None   # Initialize parsed_date

        for fmt in formats.get("datetime_formats", []):
            # First, check for range_of_days special case
            if fmt == 'range_of_days' and "-" in date_str and any(char.isdigit() for char in date_str):
                for mon_fmt in ['%b', '%B']:
                    try:
                        month = date_str.split()[-1]
                        parsed_date = pd.to_datetime(date_str, format=mon_fmt, errors='coerce')
                        if not pd.isna(parsed_date):
                            date_str = parsed_date.strftime('%Y-%m-%d')
                            date_parsed = True
                            break
                    except Exception:
                        continue  # move to next mon_fmt
                if date_parsed:
                    break

            # Next, check for normal formats
            try:
                # Check presence of components based on format
                year_parsed = '%Y' in fmt or '%y' in fmt
                month_parsed = '%m' in fmt or '%b' in fmt or '%B' in fmt
                day_parsed = '%d' in fmt

                # Parse the date string using the format
                parsed_date = pd.to_datetime(date_str, format=fmt, errors='coerce')
                if not pd.isna(parsed_date):
                    # Extract only the available components
                    if year_parsed:
                        year_str = int(parsed_date.year)
                    if month_parsed:
                        month_str = int(parsed_date.month)
                    if day_parsed:
                        day_str = int(parsed_date.day)

                    date_parsed = True
                    break
            except Exception:
                continue

        # Set date_str to None if no formats succeed
        if not date_parsed:
            date_str = None

        # (2) Parse time_str
        time_parsed = False
        for fmt in formats.get("time_formats", []):
            if fmt == 'total_minutes':
                try:
                    parsed_time = parse_total_minutes(time_str)
                    if parsed_time:
                        time_str = parsed_time.strftime('%H:%M:%S')
                        time_parsed = True
                        break
                except Exception:
                    continue
            elif fmt == 'decimal_hours':
                try:
                    parsed_time = parse_decimal_hours(time_str)
                    if parsed_time:
                        time_str = parsed_time.strftime('%H:%M:%S')
                        time_parsed = True
                        break
                except Exception:
                    continue
            elif fmt == 'date_with_time':
                try:
                    ds = clean_and_standardize_datetime_string(original_date)
                    parsed_ds = pd.to_datetime(ds, errors='coerce')
                    if not pd.isna(parsed_ds):
                        time_str = parsed_ds.strftime('%H:%M:%S')
                        time_parsed = True
                        break
                except Exception:
                    continue
            else:
                try:
                    parsed_time = pd.to_datetime(time_str, format=fmt, errors='coerce')
                    if not pd.isna(parsed_time):
                        time_str = parsed_time.strftime('%H:%M:%S')
                        time_parsed = True
                        break
                except Exception:
                    continue

        # if time_str is NaT, set to None
        if not time_parsed:
            time_str = None

        return original_date, original_time, year_str, month_str, day_str, time_str
    except Exception:
        return row[('date', 'Date of observation', 1)], row[('time', 'Time of observation', 1)], None, None, None, None

# Sub-function for unit conversion
def convert_units(data, from_unit, to_unit):
    try:
        # Skip conversion if from_unit is dimensionless or to_unit is a datetime format
        if from_unit == "1" or any(x in to_unit for x in [":", "HH", "MM", "SS"]):
            return data

        # Ensure data is in the correct format for conversion
        if isinstance(data, pd.Series):
            data = data.to_numpy()

        # Handle NaN values by creating a mask
        nan_mask = np.isnan(data)
        
        # Replace NaN with a placeholder (0) for conversion, will restore NaN later
        data_no_nan = np.where(nan_mask, 0, data)

        # Create cfunits for conversion
        from_cf_unit = cfunits.Units(from_unit)
        to_cf_unit = cfunits.Units(to_unit)

        # Check if units are equivalent before converting
        if from_cf_unit.equivalent(to_cf_unit):
            # Perform unit conversion using cfunits
            converted_data = cfunits.Units.conform(data_no_nan, from_cf_unit, to_cf_unit)
            
            # Restore NaN values in the converted data
            converted_data[nan_mask] = np.nan
            
            print(f"Converted data from {from_unit} to {to_unit}: min={np.nanmin(converted_data)}, max={np.nanmax(converted_data)}")
            return converted_data
        else:
            print(f"Units {from_unit} and {to_unit} are not compatible.")
            return data

    except Exception as e:
        print(f"Error converting from {from_unit} to {to_unit}: {e}")
        return data  # Return the original data if conversion fails

def check_and_convert_units(column_standards, df):
    # Extract the multi-level headers from merged_df
    multi_headers = df.columns

    # Create lists to store successfully converted columns before and after conversion
    before_conversion_headers = []
    after_conversion_headers = []

    # Create a list to store updated headers
    updated_headers = []

    # Iterate through each column
    for col in multi_headers:
        # Extract the variable name, description, and unit from the multi-level header
        variable_name = col[0]
        description = col[1]
        header_unit = col[2]

        # Find matching rows in column_standards
        match = column_standards[
            (column_standards['standard_variable'] == variable_name) &
            (column_standards['standard_description'] == description)
        ]

        if not match.empty:
            # Get the standard unit
            standard_unit = match['standard_unit'].iloc[0]

            # Check if the units match
            if header_unit != standard_unit:
                print(f"Mismatch found: {variable_name} ({description}) - {header_unit} to {standard_unit}")

                # Try to convert the data and catch any errors
                try:
                    df[col] = convert_units(df[col], header_unit, standard_unit)

                    # Update the header unit to the standard unit
                    updated_col = (variable_name, description, standard_unit)
                    updated_headers.append(updated_col)

                    if not ':' in updated_col[2]: # ignore date/time unit formats

                        # Add to successfully converted columns lists
                        before_conversion_headers.append(col)
                        after_conversion_headers.append(updated_col)

                    print("Converted units and updated header.")
                except Exception as e:
                    print(f"Error converting {variable_name} ({description}): {e}")
                    updated_headers.append(col)
            else:
                # If units match, keep the original header
                updated_headers.append(col)
        else:
            # If no match found, keep the original header
            updated_headers.append(col)

    # Replace the existing headers with the updated headers
    df.columns = pd.MultiIndex.from_tuples(updated_headers)

    return df, before_conversion_headers, after_conversion_headers

def convert_to_utf8sig(file_path):
    # Define new file path with '_utf8' appended before the extension
    dir_name = os.path.dirname(file_path)
    base_name, extension = os.path.splitext(os.path.basename(file_path))
    new_file_path = os.path.join(dir_name, f"{base_name}_utf8sig.csv")

    # Read encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f'Encoding {encoding} was detected for {file_path}.')

    # If not UTF-8, re-write as utf-8 and save as new_file_path
    if encoding and encoding.lower() != 'utf-8-sig':
        
        # Read the file with the detected encoding
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()

        # Write the content in UTF-8 encoding
        with open(new_file_path, 'w', encoding='utf-8-sig') as f:
            f.write(content)
            
        print(f'Converted {file_path} to utf-8-sig and saved as {new_file_path}')
        return new_file_path

    elif not encoding:
        print(f'Chardet was unable to detect encoding for {file_path}')
        return file_path
        
    else:
        print(f"{file_path} is already in utf-8-sig encoding.")
        return file_path