import pandas as pd

def clean_coverage_data(input_path, output_path=None):
    """
    Load CSV data and remove rows where 'Coverage Status' is null.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the cleaned CSV (optional, if None returns DataFrame only)
    
    Returns:
        Cleaned DataFrame
    """
    # Load the data (try multiple encodings for compatibility)
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_path, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='latin-1')
    
    # Get initial row count
    initial_count = len(df)
    
    # Remove rows where 'Coverage Status' is null
    df_cleaned = df.dropna(subset=['Coverage Status'])
    
    # Get final row count
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count
    
    print(f"Initial rows: {initial_count}")
    print(f"Rows removed (null Coverage Status): {removed_count}")
    print(f"Final rows: {final_count}")
    
    # Save to output path if provided
    if output_path:
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Cleaned data saved to: {output_path}")
    
    return df_cleaned


if __name__ == "__main__":
    # Input and output file paths
    input_file = r"c:\Users\VH0000812\Desktop\Coverage\data\Acronym.csv"
    output_file = r"c:\Users\VH0000812\Desktop\Coverage\data\Acronym_cleaned.csv"
    
    # Clean the data
    cleaned_df = clean_coverage_data(input_file, output_file)
