import os
from collections import defaultdict

import cfunits
import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        mu = "\u03bc"  # Greek small letter mu (μ)
        micro = "\u00b5"  # Micro sign (µ)
        if isinstance(s, str):
            return s.replace(mu, micro)
        return s

    cleaned_values = (
        values.str.strip()  # Remove leading/trailing whitespace
        .str.replace(
            r"(^['\"]|['\"]$)", "", regex=True
        )  # Remove leading/trailing quotes
        .str.replace(
            r"\s*,\s*", ", ", regex=True
        )  # Normalize commas with single space after them
        .str.replace(r"\s+", " ", regex=True)  # Normalize whitespace within strings
        .str.replace(r"' '", "", regex=True)  # Remove isolated single quotes
        .str.replace(
            r"([a-zA-Z]),([a-zA-Z])", r"\1, \2", regex=True
        )  # Add space after commas if missing
    )

    # Handle cases like ' evergreen, deciduous '
    cleaned_values = cleaned_values.str.replace(
        r"['‘’]", "", regex=True
    )  # Remove any stray single quotes
    cleaned_values = cleaned_values.str.replace(
        r"\b ,\b", ",", regex=True
    )  # Fix any stray commas

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
            if (
                text.strip() == "1"
            ):  # Check for lone '1' (ignoring surrounding whitespace)
                return 1  # Convert to integer 1
            return unicodedata.normalize("NFKC", text)  # Normalize Unicode to NFC form
        return text

    # Ensure the DataFrame's multi-index (if present) is sorted
    if isinstance(df.columns, pd.MultiIndex) and not df.columns.is_monotonic_increasing:
        df = df.sort_index(axis=1)

    # Normalize each level in the column headers if it is a multi-index
    if isinstance(df.columns, pd.MultiIndex):
        normalized_columns = [
            tuple(normalize_encoding(level) for level in col) for col in df.columns
        ]
        df.columns = pd.MultiIndex.from_tuples(
            normalized_columns, names=df.columns.names
        )
    else:
        df.columns = [normalize_encoding(col) for col in df.columns]

    # Normalize each cell in the DataFrame without overwriting during iteration
    result_df = df.copy()  # Work on a copy to avoid issues while modifying
    for col in df.columns:
        result_df[col] = df[col].apply(
            lambda x: normalize_encoding(x) if isinstance(x, str) else x
        )

    return result_df


def standardize_headers(df, lookup_dict):
    """
    Standardizes multi-index column headers using lookup_dict and merges duplicate standard_variables.

    Parameters:
    df (pd.DataFrame): DataFrame with a 3-level multi-index column (variable, description, unit).
    lookup_dict (dict): Dictionary mapping (variable, description, unit) tuples to standardized names.

    Returns:
    pd.DataFrame: DataFrame with standardized and merged columns.
    """
    # Step 1: Standardize column headers
    new_columns = []
    original_columns_map = defaultdict(
        list
    )  # {std_variable : [(variable, description, unit), ...]}

    for col in df.columns:
        if col in lookup_dict:
            std_info = lookup_dict[col]
            std_variable = std_info["standard_variable"]
            std_description = std_info["standard_description"]
            std_unit = std_info["standard_unit"]

            # Handle special unit cases
            unit = 1 if std_unit == 1 else col[2] if not pd.isna(col[2]) else std_unit

            new_columns.append((std_variable, std_description, unit))
            original_columns_map[(std_variable, std_description, unit)].append(
                col
            )  # Map original column to standard_variable
        else:
            new_columns.append(col)  # Keep unchanged if no match
            original_columns_map[col].append(col)  # Default to original variable name

    # Step 2: Update DataFrame headers
    new_df = df.copy()
    new_df.columns = pd.MultiIndex.from_tuples(
        new_columns, names=["standard_variable", "standard_description", "unit"]
    )

    # Step 3: Handle duplicate multi-headers
    merged_data = {}
    for std_var, col_list in original_columns_map.items():
        if len(col_list) > 1:
            # Subselect the columns from df
            sub_df = df[list(col_list)]

            # Create the merged column
            def merge_row_values(row):
                merged_strings = []
                for col, value in zip(col_list, row):
                    var_name = col[0]  # Extract the variable name
                    value_str = (
                        str(value)
                        if pd.notna(value) and value not in ["", None]
                        else "None"
                    )
                    merged_strings.append(f"{var_name}: {value_str}")
                return ", ".join(merged_strings)

            merged_column = sub_df.apply(merge_row_values, axis=1)

            # Store the merged column
            merged_data[std_var] = merged_column

    # Convert merged columns into DataFrame
    merged_df = pd.DataFrame(merged_data, index=df.index)

    # Drop duplicate columns
    new_df = new_df.loc[:, ~new_df.columns.duplicated()]

    # Add merged columns to new_df (replace the duplicate cols we just dropped)
    for col, values in merged_df.items():
        new_df[col] = values

    return new_df


def visualize_all_columns(data):
    # Extract the column data
    data = data.dropna()

    if data.empty:
        # Skip empty columns
        print("Column is empty, skipping visualization.")

    # Check if the data is numeric
    elif pd.api.types.is_numeric_dtype(data):
        # Plot numeric data
        sns.displot(data, kde=True)
        plt.show()

    else:
        # if object/categorical data
        sns.countplot(data)
        plt.show()


def _clean_and_standardize_datetime_string(date_str, special_case=None):
    """Remove extra spaces and handle range of dates or special cases."""
    if not isinstance(date_str, str):
        return date_str
    # Remove extra spaces and convert to lowercase
    date_str = date_str.strip().lower()
    return date_str


def _parse_total_minutes(total_minutes):
    """Convert total minutes since some reference into %H:%M:%S format."""
    try:
        total_minutes = float(total_minutes)
        minutes_in_day = total_minutes % 1440

        # Convert to hours, minutes, and seconds
        hours = int(minutes_in_day // 60)
        remaining_minutes = minutes_in_day % 60
        minutes = int(remaining_minutes)
        seconds = int((remaining_minutes - minutes) * 60)
        parsed_time = pd.Timestamp(
            year=2000, month=1, day=1, hour=hours, minute=minutes, second=seconds
        )

        return parsed_time

    except Exception:
        return None


def _parse_decimal_hours(decimal_hours):
    """Convert decimal hours into %H:%M:%S format."""
    try:
        decimal_hours = float(decimal_hours)
        hours = int(decimal_hours)
        minutes = int((decimal_hours - hours) * 60)
        seconds = int(((decimal_hours - hours) * 60 - minutes) * 60)

        parsed_time = pd.Timestamp(
            year=2000, month=1, day=1, hour=hours, minute=minutes, second=seconds
        )
        return parsed_time

    except Exception:
        return None


def _parse_date_with_time(date_str, fmt):
    """Extract time component from a datetime string."""
    try:
        parsed_datetime = pd.to_datetime(date_str, format=fmt, errors="coerce")
        if not pd.isna(parsed_datetime):
            return parsed_datetime
    except Exception:
        return None


def create_datetime_temp(data, filename):
    # group-by data contributor and site and create date-related columns
    date_grouped = data.groupby(
        [
            (
                "dataContributor",
                "Name of the individual or organization that contributed to the data",
                1,
            ),
            ("siteIdentifier", "Location descriptor of where data was collected", 1),
        ]
    )[[("date", "Date of observation", 1)]].agg(lambda x: x.unique())
    date_grouped.columns = date_grouped.columns.get_level_values(0)
    date_grouped = date_grouped.reset_index()
    date_grouped["source"] = (
        date_grouped[
            (
                "dataContributor",
                "Name of the individual or organization that contributed to the data",
                1,
            )
        ]
        + " "
        + date_grouped[
            ("siteIdentifier", "Location descriptor of where data was collected", 1)
        ]
    )
    date_grouped = date_grouped.drop(
        columns=[
            (
                "dataContributor",
                "Name of the individual or organization that contributed to the data",
                1,
            ),
            ("siteIdentifier", "Location descriptor of where data was collected", 1),
        ]
    )
    date_grouped = date_grouped[["source", "date"]]

    # group-by data contribtor and site and create time-related columns
    time_grouped = data.groupby(
        [
            (
                "dataContributor",
                "Name of the individual or organization that contributed to the data",
                1,
            ),
            ("siteIdentifier", "Location descriptor of where data was collected", 1),
        ]
    )[[("time", "Time of observation", 1)]].agg(lambda x: x.unique())
    time_grouped.columns = time_grouped.columns.get_level_values(0)
    time_grouped = time_grouped.reset_index()
    time_grouped["source"] = (
        time_grouped[
            (
                "dataContributor",
                "Name of the individual or organization that contributed to the data",
                1,
            )
        ]
        + " "
        + time_grouped[
            ("siteIdentifier", "Location descriptor of where data was collected", 1)
        ]
    )
    time_grouped = time_grouped.drop(
        columns=[
            (
                "dataContributor",
                "Name of the individual or organization that contributed to the data",
                1,
            ),
            ("siteIdentifier", "Location descriptor of where data was collected", 1),
        ]
    )
    time_grouped = time_grouped[["source", "time"]]

    # combine time and date columns into one df
    grouped = date_grouped.merge(
        time_grouped, how="left", left_on="source", right_on="source"
    )

    # export
    grouped["datetime_formats"], grouped["time_formats"], grouped["dayfirst"] = (
        None,
        None,
        None,
    )
    grouped.index.name = ""
    outfile = f"../data/temp/datetimes/{filename}_datetime_formatting_temp.csv"
    grouped.to_csv(outfile)
    print(f"Exported datetime template to {outfile}")


def process_datetime(row, formats):
    """Infer date and time from a string using source-specific formats."""
    try:
        # Preserve original date and time for comparison
        original_date = row[("date", "Date of observation", 1)]
        original_time = row[("time", "Time of observation", 1)]

        # Clean and standardize the datetime string
        date_str = _clean_and_standardize_datetime_string(original_date)
        time_str = _clean_and_standardize_datetime_string(original_time)

        # Initialize year, month, day as None
        year_str, month_str, day_str = None, None, None

        # (1) Parse date_str
        date_parsed = False  # Track whether parsing was successful
        parsed_date = None  # Initialize parsed_date

        for fmt in formats.get("datetime_formats", []):
            # First, check for range_of_days special case
            if (
                fmt == "range_of_months"
                and "-" in date_str
                and any(char.isdigit() for char in date_str)
            ):
                for mon_fmt in ["%b", "%B"]:
                    try:
                        parsed_date = pd.to_datetime(
                            date_str, format=mon_fmt, errors="coerce"
                        )
                        if not pd.isna(parsed_date):
                            date_str = parsed_date.strftime("%Y-%m-%d")
                            date_parsed = True
                            break
                    except Exception:
                        continue  # move to next mon_fmt
                if date_parsed:
                    break

            # Next, check for normal formats
            try:
                # Check presence of components based on format
                year_parsed = "%Y" in fmt or "%y" in fmt
                month_parsed = "%m" in fmt or "%b" in fmt or "%B" in fmt
                day_parsed = "%d" in fmt

                # Parse the date string using the format
                parsed_date = pd.to_datetime(date_str, format=fmt, errors="coerce")
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
            if fmt == "total_minutes":
                try:
                    parsed_time = _parse_total_minutes(time_str)
                    if parsed_time:
                        time_str = parsed_time.strftime("%H:%M:%S")
                        time_parsed = True
                        break
                except Exception:
                    continue
            elif fmt == "decimal_hours":
                try:
                    parsed_time = _parse_decimal_hours(time_str)
                    if parsed_time:
                        time_str = parsed_time.strftime("%H:%M:%S")
                        time_parsed = True
                        break
                except Exception:
                    continue
            elif fmt == "date_with_time":
                try:
                    ds = _clean_and_standardize_datetime_string(original_date)
                    parsed_ds = pd.to_datetime(ds, errors="coerce")
                    if not pd.isna(parsed_ds):
                        time_str = parsed_ds.strftime("%H:%M:%S")
                        time_parsed = True
                        break
                except Exception:
                    continue
            else:
                try:
                    parsed_time = pd.to_datetime(time_str, format=fmt, errors="coerce")
                    if not pd.isna(parsed_time):
                        time_str = parsed_time.strftime("%H:%M:%S")
                        time_parsed = True
                        break
                except Exception:
                    continue

        # if time_str is NaT, set to None
        if not time_parsed:
            time_str = None

        return original_date, original_time, year_str, month_str, day_str, time_str
    except Exception:
        return (
            row[("date", "Date of observation", 1)],
            row[("time", "Time of observation", 1)],
            None,
            None,
            None,
            None,
        )


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
            converted_data = cfunits.Units.conform(
                data_no_nan, from_cf_unit, to_cf_unit
            )

            # Restore NaN values in the converted data
            converted_data[nan_mask] = np.nan

            print(
                f"Converted data from {from_unit} to {to_unit}: min={np.nanmin(converted_data)}, max={np.nanmax(converted_data)}"
            )
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
            (column_standards["standard_variable"] == variable_name)
            & (column_standards["standard_description"] == description)
        ]

        if not match.empty:
            # Get the standard unit
            standard_unit = match["standard_unit"].iloc[0]

            # Check if the units match
            if header_unit != standard_unit:
                print(
                    f"Mismatch found: {variable_name} ({description}) - {header_unit} to {standard_unit}"
                )

                # Try to convert the data and catch any errors
                try:
                    df[col] = convert_units(df[col], header_unit, standard_unit)

                    # Update the header unit to the standard unit
                    updated_col = (variable_name, description, standard_unit)
                    updated_headers.append(updated_col)

                    if ":" not in updated_col[2]:  # ignore date/time unit formats
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
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
        print(f"Encoding {encoding} was detected for {file_path}.")

    # If not UTF-8, re-write as utf-8 and save as new_file_path
    if encoding and encoding.lower() != "utf-8-sig":
        # Read the file with the detected encoding
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            content = f.read()

        # Write the content in UTF-8 encoding
        with open(new_file_path, "w", encoding="utf-8-sig") as f:
            f.write(content)

        print(f"Converted {file_path} to utf-8-sig and saved as {new_file_path}")
        return new_file_path

    elif not encoding:
        print(f"Chardet was unable to detect encoding for {file_path}")
        return file_path

    else:
        print(f"{file_path} is already in utf-8-sig encoding.")
        return file_path


def _hist_plot_comparison(selected_col_1, selected_col_2, col_name):
    """
    Creates a subplot with one row and two columns comparing data from merged_df and filtered_df.
    - Ensures consistent bin sizes across both histograms, while allowing different x-axis ranges.

    Parameters:
    selected_col_1 (Series): Data for the left plot (from merged_df).
    selected_col_2 (Series): Data for the right plot (from filtered_df).
    col_name (str): The variable name being compared.
    """

    # Drop NaN values only for plotting
    col_1_no_nan = selected_col_1.dropna()
    col_2_no_nan = selected_col_2.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Comparison of {col_name}", fontsize=16)

    # Check if the data is numeric
    if pd.api.types.is_numeric_dtype(col_1_no_nan):
        # Determine bin size based on the range of the first dataset
        min1, max1 = col_1_no_nan.min(), col_1_no_nan.max()
        min2, max2 = col_2_no_nan.min(), col_2_no_nan.max()

        # Define bin edges
        num_bins = 20
        bin_edges1 = np.linspace(min1, max1, num_bins + 1)
        bin_edges2 = np.linspace(min2, max2, num_bins + 1)

        # Plot numeric data with consistent bin sizes
        sns.histplot(col_1_no_nan, bins=bin_edges1, kde=True, ax=axes[0])
        axes[0].set_title("df1 (Numeric Data)")
        axes[0].axvline(
            col_1_no_nan.min(),
            color="green",
            linestyle="--",
            label=f"Min: {col_1_no_nan.min():.2f}",
        )
        axes[0].axvline(
            col_1_no_nan.max(),
            color="purple",
            linestyle="--",
            label=f"Max: {col_1_no_nan.max():.2f}",
        )
        axes[0].legend()

        sns.histplot(
            col_2_no_nan, bins=bin_edges2, kde=True, ax=axes[1], color="orange"
        )
        axes[1].set_title("df2 (Numeric Data)")
        axes[1].axvline(
            col_2_no_nan.min(),
            color="green",
            linestyle="--",
            label=f"Min: {col_2_no_nan.min():.2f}",
        )
        axes[1].axvline(
            col_2_no_nan.max(),
            color="purple",
            linestyle="--",
            label=f"Max: {col_2_no_nan.max():.2f}",
        )
        axes[1].legend()

    else:
        # Plot categorical data if not numeric
        sns.countplot(y=col_1_no_nan, ax=axes[0], color="blue", legend=False)
        axes[0].set_title("df1 (Categorical Data)")

        sns.countplot(y=col_2_no_nan, ax=axes[1], color="orange", legend=False)
        axes[1].set_title("df2 (Categorical Data)")

    # Remove Y-axis labels
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Now plot the comparison
def compare_col_hists(preconverted_df, preconverted_cols, converted_df, converted_cols):
    for preconv_col, conv_col in zip(preconverted_cols, converted_cols):
        print(f"Visualizing {preconv_col} and {conv_col}")

        try:
            # Select columns using multi-level tuples
            selected_col_1 = preconverted_df[preconv_col].squeeze()
            selected_col_2 = converted_df[conv_col].squeeze()

            # Call the comparison plot function
            _hist_plot_comparison(selected_col_1, selected_col_2, preconv_col)

        except Exception as e:
            print("Could not plot...", e)


def filter_out_of_range_data(out_of_range_info, df):
    """
    Filters out data in merged_df_2 that falls outside the expected value range specified in oor.
    Also handles potential unit errors for specific variables (e.g., Patm in hPa vs. kPa).
    Special case for RH: Values above 100 are capped at 100 instead of being removed.

    Parameters:
    oor (DataFrame): Contains columns variableName, variableUnit, variableDescription,
                     expectedValueRangeMin, expectedValueRangeMax.
    merged_df_2 (DataFrame): A DataFrame with a 3-row multi-header.

    Returns:
    filtered_df (DataFrame): A copy of merged_df_2 with out-of-range values removed or capped.
    filtered_columns (list): List of multi-header column names where out-of-range data was removed or capped.
    """

    # Create a copy of merged_df_2 to avoid modifying the original DataFrame
    filtered_df = df.copy()

    # List to store multi-header column names where out-of-range data was removed or capped
    filtered_columns = []

    # Iterate over each row in oor
    for _, row in out_of_range_info.iterrows():
        var_name = row["variableName"]
        min_val = row["expectedValueRangeMin"]
        max_val = row["expectedValueRangeMax"]

        # Check if var_name exists in row 0 of the multi-header
        if var_name in filtered_df.columns.get_level_values(0):
            print(f"Match found for variable: {var_name}")

            # Select the corresponding column(s) in merged_df_2
            selected_cols = filtered_df.loc[
                :, filtered_df.columns.get_level_values(0) == var_name
            ]

            try:
                # Attempt to convert to numeric (forcing errors to raise exceptions)
                selected_cols = selected_cols.apply(pd.to_numeric, errors="raise")

                # Store the count of non-NaN values before filtering
                count_before = selected_cols.count().sum()

                # Special handling for Patm: Check for potential hPa values mistakenly recorded as kPa
                if var_name == "Patm":
                    print(f"Original Patm count: {count_before}")
                    suspect_values = selected_cols[selected_cols < 10]
                    print(
                        f"Suspect Patm values count (less than 10 kPa): {suspect_values.count().sum()}"
                    )

                    if not suspect_values.empty:
                        print(
                            f"Suspect values detected for {var_name}. Converting from hPa to kPa..."
                        )
                        # Convert suspect values from hPa to kPa by multiplying by 10
                        selected_cols = selected_cols.where(
                            selected_cols >= 10, selected_cols * 10
                        )
                    print(f"Patm count after conversion: {selected_cols.count().sum()}")

                # Special case for RH: Cap values above 100 at 100
                if "RH" in var_name:
                    over_100_count = (selected_cols > 100).sum().sum()
                    if over_100_count > 0:
                        print(
                            f"Special case for RH: Capping {over_100_count} values over 100 at 100."
                        )
                        selected_cols = selected_cols.where(selected_cols <= 100, 100)

                # Apply min and max filtering using .where()
                if not pd.isna(min_val):
                    selected_cols = selected_cols.where(selected_cols >= min_val)
                if not pd.isna(max_val):
                    selected_cols = selected_cols.where(selected_cols <= max_val)

                # Store the count of non-NaN values after filtering
                count_after = selected_cols.count().sum()

                print(
                    f"{var_name} count before filtering: {count_before}, count after filtering: {count_after}"
                )

                # Check if any data was actually removed or capped
                if count_before > count_after or (
                    "RH" in var_name and over_100_count > 0
                ):
                    # Assign the filtered column back to filtered_df
                    filtered_df.loc[
                        :, filtered_df.columns.get_level_values(0) == var_name
                    ] = selected_cols

                    # Add the multi-header column names to filtered_columns list
                    filtered_columns.extend(selected_cols.columns.tolist())

            except ValueError:
                # Skip columns that cannot be converted to numeric
                print(f"Skipping non-numeric variable: {var_name}")
        else:
            print(f"No match found for variable: {var_name}, skipping.")

    return filtered_df, filtered_columns
