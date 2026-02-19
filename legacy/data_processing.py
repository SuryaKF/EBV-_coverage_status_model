"""
DEPRECATED LEGACY SCRIPT
This file is retained for historical reference only.
Active ETL/ML/API implementation lives under `coverage_pipeline/`.
Use: etl_ingest.py -> etl_validate.py -> etl_transform.py -> etl_snapshot.py -> train_model.py -> approve_model.py
"""
import pandas as pd
import os
import re

# Comprehensive PA keywords
PA_KEYWORDS = [
    'prior authorization', 'prior auth', 'pa required', 'pa needed',
    'authorization required',
    'requires prior authorization',
    'must be approved', 'approval needed', 'prior approval needed'
]

# For standalone 'PA' - we need special handling to avoid PA2, PA3, etc.
PA_STANDALONE = 'pa'

# Comprehensive ST keywords
ST_KEYWORDS = [
    'step therapy', 'step-therapy'
]

# For standalone 'ST' - we need special handling to avoid ST1, ST2, etc.
ST_STANDALONE = 'st'

# Build regex patterns with word boundaries
# For multi-word keywords, use the standard pattern
PA_MULTIWORD_PATTERN = '|'.join(map(re.escape, PA_KEYWORDS))
ST_MULTIWORD_PATTERN = '|'.join(map(re.escape, ST_KEYWORDS))

# For standalone PA/ST, we need to ensure they're NOT followed by digits
# Pattern explanation:
# (?:^|[^a-zA-Z0-9]) - Start of string OR non-alphanumeric character (word boundary)
# (?:keyword) - Match the keyword
# (?:$|(?![0-9])[^a-zA-Z]) - End of string OR (not followed by digit AND followed by non-letter)

# Combined PA pattern: matches multi-word phrases OR standalone 'pa' (not PA2, PA3)
PA_PATTERN = (
    r'(?:^|[^a-zA-Z0-9])(?:' + PA_MULTIWORD_PATTERN + r')(?:$|[^a-zA-Z0-9])|'  # Multi-word
    r'(?:^|[^a-zA-Z0-9])' + re.escape(PA_STANDALONE) + r'(?=(?:$|[^a-zA-Z0-9](?![0-9])))'  # Standalone PA
)
RE_PA = re.compile(PA_PATTERN, re.IGNORECASE)

# Combined ST pattern: matches multi-word phrases OR standalone 'st' (not ST1, ST2)
ST_PATTERN = (
    r'(?:^|[^a-zA-Z0-9])(?:' + ST_MULTIWORD_PATTERN + r')(?:$|[^a-zA-Z0-9])|'  # Multi-word
    r'(?:^|[^a-zA-Z0-9])' + re.escape(ST_STANDALONE) + r'(?=(?:$|[^a-zA-Z0-9](?![0-9])))'  # Standalone ST
)
RE_ST = re.compile(ST_PATTERN, re.IGNORECASE)

# Simpler, more robust approach using word boundaries
# This matches PA or ST as complete words, not followed by digits
RE_PA_SIMPLE = re.compile(
    r'\b(?:prior authorization|prior auth|pa required|pa needed|authorization required|'
    r'requires prior authorization|must be approved|approval needed|prior approval needed)\b|'
    r'\bpa\b(?!\d)',  # PA as word boundary, not followed by digit
    re.IGNORECASE
)

RE_ST_SIMPLE = re.compile(
    r'\b(?:step therapy|step-therapy)\b|'
    r'\bst\b(?!\d)',  # ST as word boundary, not followed by digit
    re.IGNORECASE
)

def clean_coverage_data(input_path, output_path=None):
    """
    Load CSV data, remove rows where 'Coverage Status' is null,
    and standardize PA/ST coverage status formats using comprehensive keyword matching.
    Handles edge cases like PA2, PA3, ST1, ST2 (which should NOT match).
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the cleaned CSV (optional)
    
    Returns:
        Cleaned DataFrame
    """
    print("=" * 60)
    print("DATA PREPROCESSING STARTED")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the data
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_path, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='latin-1')
    
    print(f"\nLoaded file: {input_path}")
    print(f"Columns found: {list(df.columns)}")
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Map column names to standard format
    column_mapping = {
        'COVERAGE STATUS': 'Coverage Status',
        'Coverage status': 'Coverage Status',
        'coverage status': 'Coverage Status',
        'EXPLANATION': 'Explanation',
        'explanation': 'Explanation',
        'ACRONYM': 'Acronym',
        'acronym': 'Acronym',
        'Acronyms': 'Acronym',
        'ACRONYMS': 'Acronym'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Check if Coverage Status column exists
    if 'Coverage Status' not in df.columns:
        raise ValueError(f"'Coverage Status' column not found. Available columns: {list(df.columns)}")
    
    # Check if Acronym column exists
    has_acronym_column = 'Acronym' in df.columns
    if has_acronym_column:
        print(f"✓ Acronym column found - will process PA/ST patterns")
    else:
        print(f"⚠ Acronym column NOT found - will only standardize Coverage Status")
    
    # Drop unnamed/empty columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"Dropping {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)
    
    # Print initial row count
    initial_count = len(df)
    print(f"\nRows before processing: {initial_count}")
    
    # Print null values per column
    print("-" * 60)
    print("NULL VALUES PER COLUMN:")
    print("-" * 60)
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"  {col}: {count} null values")
    
    # Remove rows where 'Coverage Status' is null
    print("-" * 60)
    print("REMOVING NULL COVERAGE STATUS ROWS")
    print("-" * 60)
    coverage_null_count = df['Coverage Status'].isnull().sum()
    print(f"Null values in 'Coverage Status' column: {coverage_null_count}")
    
    df_cleaned = df.dropna(subset=['Coverage Status']).copy()
    
    # Normalize Coverage Status - convert to string first
    df_cleaned['Coverage Status'] = df_cleaned['Coverage Status'].astype(str).str.strip()
    
    # Show unique values before standardization
    print("-" * 60)
    print("COVERAGE STATUS VALUES BEFORE STANDARDIZATION:")
    print("-" * 60)
    unique_before = df_cleaned['Coverage Status'].unique()
    for i, val in enumerate(unique_before[:20], 1):
        print(f"  {i}. {val}")
    if len(unique_before) > 20:
        print(f"  ... and {len(unique_before) - 20} more")
    
    # Standardize Coverage Status values
    print("-" * 60)
    print("STANDARDIZING COVERAGE STATUS VALUES")
    print("-" * 60)
    
    def standardize_coverage_status(status):
        """Standardize coverage status strings to canonical forms using regex patterns"""
        if pd.isna(status) or status == 'nan':
            return None
            
        status_str = str(status).strip()
        
        # Check using regex patterns for PA and ST (using the SIMPLE version for better accuracy)
        has_pa = bool(RE_PA_SIMPLE.search(status_str))
        has_st = bool(RE_ST_SIMPLE.search(status_str))
        
        # If both PA and ST are mentioned
        if has_pa and has_st:
            return 'Covered with Condition (PA & ST Required)'
        
        # If only PA is mentioned
        if has_pa:
            return 'Covered with Condition (PA Required)'
        
        # If only ST is mentioned
        if has_st:
            return 'Covered with Condition (ST Required)'
        
        # Check for generic "Covered with Condition" (without PA/ST)
        status_lower = status_str.lower()
        if 'covered with condition' in status_lower or 'coverage with condition' in status_lower:
            return 'Covered with Condition'
        
        # Check for "Covered" (without conditions)
        if status_lower in ['covered', 'part b covered', 'part b\ncovered', 'yes', 'y'] or \
           (status_lower.startswith('covered') and 'condition' not in status_lower and 'not' not in status_lower):
            return 'Covered'
        
        # Check for "Not Covered"
        if 'not covered' in status_lower or status_lower in ['no', 'n', 'not applicable', 'n/a', 'non-covered']:
            return 'Not Covered'
        
        # Return original if no pattern matches
        return status_str
    
    # Apply standardization
    df_cleaned['Coverage Status'] = df_cleaned['Coverage Status'].apply(standardize_coverage_status)
    
    print("\nStandardization applied based on Coverage Status text")
    
    # Process Acronym column if it exists
    if has_acronym_column:
        print("-" * 60)
        print("PROCESSING ACRONYM COLUMN FOR PA/ST PATTERNS")
        print("-" * 60)
        print("Note: PA2, PA3, ST1, ST2 etc. will NOT be matched (digits excluded)")
        
        # Fill NaN values with empty string
        df_cleaned['Acronym'] = df_cleaned['Acronym'].fillna('').astype(str)
        
        # Apply regex patterns to Acronym column (using SIMPLE version)
        df_cleaned['has_PA'] = df_cleaned['Acronym'].apply(lambda x: bool(RE_PA_SIMPLE.search(str(x))))
        df_cleaned['has_ST'] = df_cleaned['Acronym'].apply(lambda x: bool(RE_ST_SIMPLE.search(str(x))))
        
        # Count matches
        pa_count = df_cleaned['has_PA'].sum()
        st_count = df_cleaned['has_ST'].sum()
        both_count = (df_cleaned['has_PA'] & df_cleaned['has_ST']).sum()
        
        print(f"\nAcronym column analysis:")
        print(f"  Rows with PA pattern: {pa_count}")
        print(f"  Rows with ST pattern: {st_count}")
        print(f"  Rows with BOTH PA and ST: {both_count}")
        
        # Show sample matches
        if pa_count > 0:
            print("\nSample PA matches from Acronym column:")
            pa_samples = df_cleaned[df_cleaned['has_PA']][['Acronym', 'Coverage Status']].head(5)
            for idx, row in pa_samples.iterrows():
                acronym_preview = str(row['Acronym'])[:60] + "..." if len(str(row['Acronym'])) > 60 else str(row['Acronym'])
                print(f"  ✓ Acronym: {acronym_preview}")
                print(f"    Status: {row['Coverage Status']}")
        
        if st_count > 0:
            print("\nSample ST matches from Acronym column:")
            st_samples = df_cleaned[df_cleaned['has_ST']][['Acronym', 'Coverage Status']].head(5)
            for idx, row in st_samples.iterrows():
                acronym_preview = str(row['Acronym'])[:60] + "..." if len(str(row['Acronym'])) > 60 else str(row['Acronym'])
                print(f"  ✓ Acronym: {acronym_preview}")
                print(f"    Status: {row['Coverage Status']}")
        
        # Test for false positives (PA2, PA3, etc.)
        test_patterns = ['PA2', 'PA3', 'ST1', 'ST2', 'PAST', 'PART', 'STEP','DST']
        false_positives = df_cleaned[df_cleaned['Acronym'].str.upper().str.contains('|'.join(test_patterns), na=False)]
        if len(false_positives) > 0:
            false_pa = false_positives[false_positives['has_PA']]
            false_st = false_positives[false_positives['has_ST']]
            if len(false_pa) > 0:
                print(f"\n⚠ WARNING: {len(false_pa)} potential false positive PA matches detected")
                print("Sample acronyms that matched PA pattern:")
                for idx, row in false_pa[['Acronym']].head(3).iterrows():
                    print(f"  - {row['Acronym']}")
            if len(false_st) > 0:
                print(f"\n⚠ WARNING: {len(false_st)} potential false positive ST matches detected")
                print("Sample acronyms that matched ST pattern:")
                for idx, row in false_st[['Acronym']].head(3).iterrows():
                    print(f"  - {row['Acronym']}")
        
        # Now apply PA/ST logic to Coverage Status for rows that are still generic "Covered with Condition"
        print("-" * 60)
        print("UPDATING COVERAGE STATUS BASED ON ACRONYMS")
        print("-" * 60)
        
        # Only update "Covered with Condition" rows based on acronyms
        condition_mask = df_cleaned['Coverage Status'] == 'Covered with Condition'
        condition_count = condition_mask.sum()
        print(f"\nRows with generic 'Covered with Condition' status: {condition_count}")
        
        if condition_count > 0:
            # Update based on PA/ST in Acronym column
            # Priority: Both > PA > ST > Generic Condition
            
            # Rows with BOTH PA and ST
            both_mask = condition_mask & df_cleaned['has_PA'] & df_cleaned['has_ST']
            both_update_count = both_mask.sum()
            if both_update_count > 0:
                df_cleaned.loc[both_mask, 'Coverage Status'] = 'Covered with Condition (PA & ST Required)'
                print(f"  ✓ Updated {both_update_count} rows to 'PA & ST Required'")
            
            # Rows with PA only (not already updated)
            pa_only_mask = condition_mask & df_cleaned['has_PA'] & ~df_cleaned['has_ST']
            pa_update_count = pa_only_mask.sum()
            if pa_update_count > 0:
                df_cleaned.loc[pa_only_mask, 'Coverage Status'] = 'Covered with Condition (PA Required)'
                print(f"  ✓ Updated {pa_update_count} rows to 'PA Required'")
            
            # Rows with ST only (not already updated)
            st_only_mask = condition_mask & ~df_cleaned['has_PA'] & df_cleaned['has_ST']
            st_update_count = st_only_mask.sum()
            if st_only_mask.sum() > 0:
                df_cleaned.loc[st_only_mask, 'Coverage Status'] = 'Covered with Condition (ST Required)'
                print(f"  ✓ Updated {st_update_count} rows to 'ST Required'")
            
            # Count remaining generic conditions
            remaining_generic = (df_cleaned['Coverage Status'] == 'Covered with Condition').sum()
            if remaining_generic > 0:
                print(f"  ℹ {remaining_generic} rows remain as generic 'Covered with Condition' (no PA/ST in Acronym)")
            
            # Show sample of updated rows
            if pa_update_count > 0 or st_update_count > 0 or both_update_count > 0:
                print("\nSample of rows updated from Acronym column:")
                updated_mask = df_cleaned['Coverage Status'].str.contains('Required', na=False)
                sample_df = df_cleaned[updated_mask][['Coverage Status', 'Acronym']].head(10)
                if len(sample_df) > 0:
                    for idx, row in sample_df.iterrows():
                        acronym_preview = str(row['Acronym'])[:50] + "..." if len(str(row['Acronym'])) > 50 else str(row['Acronym'])
                        print(f"  - {row['Coverage Status']}")
                        print(f"    Acronym: {acronym_preview}")
        
        # Drop temporary columns
        df_cleaned.drop(columns=['has_PA', 'has_ST'], inplace=True)
    
    # Define valid Coverage Status values
    valid_statuses = [
        'Covered', 
        'Covered with Condition',
        'Covered with Condition (PA Required)',
        'Covered with Condition (ST Required)',
        'Covered with Condition (PA & ST Required)',
        'Not Covered'
    ]
    
    # Find invalid rows (rows that didn't match any pattern)
    invalid_rows = df_cleaned[~df_cleaned['Coverage Status'].isin(valid_statuses)]
    if len(invalid_rows) > 0:
        print("\n" + "-" * 60)
        print(f"WARNING: {len(invalid_rows)} ROWS WITH NON-STANDARD COVERAGE STATUS")
        print("-" * 60)
        print("These rows have coverage status values that don't match standard patterns:")
        for idx, row in invalid_rows.head(10).iterrows():
            status_preview = str(row['Coverage Status'])[:70]
            print(f"  - {status_preview}")
        if len(invalid_rows) > 10:
            print(f"  ... and {len(invalid_rows) - 10} more")
        
        print("\n⚠ These rows will be REMOVED from the final output.")
        print("If you want to keep them, update the standardize_coverage_status function.")
    
    df_cleaned = df_cleaned[df_cleaned['Coverage Status'].isin(valid_statuses)]
    
    # Count Coverage Status categories
    print("-" * 60)
    print("FINAL COVERAGE STATUS DISTRIBUTION:")
    print("-" * 60)
    status_counts = df_cleaned['Coverage Status'].value_counts().sort_index()
    for status, count in status_counts.items():
        percentage = (count / len(df_cleaned)) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    print(f"\nTotal records: {len(df_cleaned)}")
    
    # Print final row count
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count
    print("-" * 60)
    print("PREPROCESSING SUMMARY")
    print("-" * 60)
    print(f"Rows before processing: {initial_count}")
    print(f"Rows removed (null or invalid): {removed_count}")
    print(f"Rows after processing: {final_count}")
    print(f"Data retention rate: {(final_count/initial_count)*100:.1f}%")
    
    # Save to output path if provided
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\nCreated output directory: {output_dir}")
        
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Cleaned data saved to: {output_path}")
    
    print("=" * 60)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 60)
    
    return df_cleaned

if __name__ == "__main__":
    input_file = r"data\January_Acronym_2026(Consolidate) (1).csv"
    output_file = r"data\Cleaned Output\Cleaned_January_Acronym_2026_v1.csv"
    
    try:
        df = clean_coverage_data(input_file, output_file)
        print(f"\n✅ SUCCESS! Processed {len(df)} rows.")
        
        # Show sample of cleaned data
        print("\nSample of final cleaned data:")
        display_cols = ['Coverage Status']
        if 'Acronym' in df.columns:
            display_cols.append('Acronym')
        sample_data = df[display_cols].head(10)
        for idx, row in sample_data.iterrows():
            print(f"\n  Row {idx + 1}:")
            print(f"    Coverage Status: {row['Coverage Status']}")
            if 'Acronym' in display_cols:
                acronym_preview = str(row['Acronym'])[:60] + "..." if len(str(row['Acronym'])) > 60 else str(row['Acronym'])
                print(f"    Acronym: {acronym_preview}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
