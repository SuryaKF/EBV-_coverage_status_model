import pandas as pd
import os
import re

# ---------------------------------------------------------------------------
# Regex patterns for detecting PA / ST / DST in the Acronym column
# ---------------------------------------------------------------------------
# Match standalone "PA" (not PA2, PA3, PART, PAST, etc.)
RE_PA = re.compile(
    r'(?:^|(?<=[^a-zA-Z0-9]))pa(?=(?:[^a-zA-Z0-9]|$))(?!\d)',
    re.IGNORECASE
)

# Match standalone "ST" or "DST" (not ST1, ST2, STEP, etc.)
RE_ST = re.compile(
    r'(?:^|(?<=[^a-zA-Z0-9]))(?:st|dst)(?=(?:[^a-zA-Z0-9]|$))(?!\d)',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# The 5 target classes after augmentation
# ---------------------------------------------------------------------------
VALID_CLASSES = [
    'Covered',                                  # 1
    'Not Covered',                              # 2
    'Coverage with Conditions',                 # 3
    'Coverage with Conditions(PA Required)',     # 4
    'Coverage with Conditions(ST Required)',     # 5
]


def clean_coverage_data(input_path, output_path=None):
    """
    Load CSV data, drop null Coverage Status rows, and augment the
    Coverage Status column so that the final dataset has exactly 5 classes:

        1. Covered
        2. Not Covered
        3. Coverage with Conditions
        4. Coverage with Conditions(PA Required)
        5. Coverage with Conditions(ST Required)

    Logic:
        - If the ACRONYM column value matches "PA"
              → Coverage with Conditions(PA Required)
        - If the ACRONYM column value matches "ST" or "DST"
              → Coverage with Conditions(ST Required)
        - Otherwise the existing Coverage Status is standardised into one of
          Covered / Not Covered / Coverage with Conditions.

    Args:
        input_path:  Path to the input CSV file.
        output_path: Path to save the augmented CSV (optional).

    Returns:
        Augmented DataFrame.
    """
    print("=" * 60)
    print("DATA AUGMENTATION STARTED")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            df = pd.read_csv(input_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Could not decode {input_path}")

    print(f"\nLoaded file: {input_path}")
    print(f"Columns found: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Standardise column names
    # ------------------------------------------------------------------
    df.columns = df.columns.str.strip()

    column_mapping = {
        'COVERAGE STATUS': 'Coverage Status',
        'Coverage status': 'Coverage Status',
        'coverage status': 'Coverage Status',
        'EXPLANATION': 'Explanation',
        'explanation': 'Explanation',
        'ACRONYM': 'Acronym',
        'acronym': 'Acronym',
        'Acronyms': 'Acronym',
        'ACRONYMS': 'Acronym',
    }
    df.rename(columns=column_mapping, inplace=True)

    if 'Coverage Status' not in df.columns:
        raise ValueError(f"'Coverage Status' column not found. Available: {list(df.columns)}")

    has_acronym = 'Acronym' in df.columns
    if has_acronym:
        print("✓ Acronym column found – will derive PA/ST classes")
    else:
        print("⚠ Acronym column NOT found – will only standardise Coverage Status")

    # Drop unnamed columns
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        print(f"Dropping {len(unnamed_cols)} unnamed columns")
        df.drop(columns=unnamed_cols, inplace=True)

    initial_count = len(df)
    print(f"\nRows before processing: {initial_count}")

    # ------------------------------------------------------------------
    # 3. Null handling
    # ------------------------------------------------------------------
    print("-" * 60)
    print("NULL VALUES PER COLUMN:")
    print("-" * 60)
    for col, cnt in df.isnull().sum().items():
        print(f"  {col}: {cnt}")

    coverage_null = df['Coverage Status'].isnull().sum()
    print(f"\nRemoving {coverage_null} rows with null Coverage Status")
    df = df.dropna(subset=['Coverage Status']).copy()
    df['Coverage Status'] = df['Coverage Status'].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 4. Show original distribution
    # ------------------------------------------------------------------
    print("-" * 60)
    print("COVERAGE STATUS VALUES BEFORE AUGMENTATION:")
    print("-" * 60)
    for i, val in enumerate(df['Coverage Status'].unique()[:25], 1):
        print(f"  {i}. {val}")

    # ------------------------------------------------------------------
    # 5. Standardise base Coverage Status (before acronym logic)
    # ------------------------------------------------------------------
    def _base_status(status):
        """Map raw status text → one of the 3 base classes."""
        if pd.isna(status) or str(status).strip().lower() == 'nan':
            return None
        s = str(status).strip().lower()

        # Not Covered
        if 'not covered' in s or s in ('no', 'n', 'not applicable', 'n/a', 'non-covered'):
            return 'Not Covered'

        # Covered with Condition (any variant)
        if 'covered with condition' in s or 'coverage with condition' in s:
            return 'Coverage with Conditions'

        # Covered
        if s in ('covered', 'part b covered', 'part b\ncovered', 'yes', 'y') or \
           (s.startswith('covered') and 'condition' not in s and 'not' not in s):
            return 'Covered'

        # Fallback – treat as generic condition if it looks conditional
        if 'condition' in s:
            return 'Coverage with Conditions'

        return status  # keep original for later filtering

    df['Coverage Status'] = df['Coverage Status'].apply(_base_status)
    # Drop rows that couldn't be mapped
    df = df[df['Coverage Status'].isin(['Covered', 'Not Covered', 'Coverage with Conditions'])].copy()

    print(f"\nAfter base standardisation: {len(df)} rows")

    # ------------------------------------------------------------------
    # 6. Augment based on Acronym column → PA / ST classes
    # ------------------------------------------------------------------
    if has_acronym:
        print("-" * 60)
        print("AUGMENTING CLASSES BASED ON ACRONYM COLUMN")
        print("-" * 60)

        df['Acronym'] = df['Acronym'].fillna('').astype(str).str.strip()

        is_pa = df['Acronym'].apply(lambda x: bool(RE_PA.search(x)))
        is_st = df['Acronym'].apply(lambda x: bool(RE_ST.search(x)))

        pa_count = is_pa.sum()
        st_count = is_st.sum()
        print(f"  Rows where Acronym matches PA : {pa_count}")
        print(f"  Rows where Acronym matches ST/DST: {st_count}")

        # Overwrite Coverage Status for PA rows
        df.loc[is_pa, 'Coverage Status'] = 'Coverage with Conditions(PA Required)'
        # Overwrite Coverage Status for ST / DST rows
        df.loc[is_st, 'Coverage Status'] = 'Coverage with Conditions(ST Required)'

        print(f"  ✓ Updated {pa_count} rows → Coverage with Conditions(PA Required)")
        print(f"  ✓ Updated {st_count} rows → Coverage with Conditions(ST Required)")

    # ------------------------------------------------------------------
    # 7. Final validation – keep only the 5 target classes
    # ------------------------------------------------------------------
    invalid = df[~df['Coverage Status'].isin(VALID_CLASSES)]
    if len(invalid) > 0:
        print(f"\n⚠ Removing {len(invalid)} rows with non-standard status values:")
        for val in invalid['Coverage Status'].unique()[:10]:
            print(f"    - {val}")
        df = df[df['Coverage Status'].isin(VALID_CLASSES)].copy()

    # ------------------------------------------------------------------
    # 8. Report final distribution
    # ------------------------------------------------------------------
    print("-" * 60)
    print("FINAL COVERAGE STATUS DISTRIBUTION (5 CLASSES):")
    print("-" * 60)
    status_counts = df['Coverage Status'].value_counts().sort_index()
    num_classes = len(status_counts)
    if num_classes == 5:
        print(f"✅ Confirmed: exactly 5 classes present\n")
    else:
        print(f"⚠  Found {num_classes} classes (some classes may be missing from this dataset)\n")

    for i, (status, count) in enumerate(status_counts.items(), 1):
        pct = count / len(df) * 100
        print(f"  {i}. {status:50s}  {count:>6}  ({pct:.1f}%)")
    print(f"\n  Total records: {len(df)}")

    final_count = len(df)
    removed = initial_count - final_count
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Rows before : {initial_count}")
    print(f"  Rows removed: {removed}")
    print(f"  Rows after  : {final_count}")
    print(f"  Retention   : {final_count / initial_count * 100:.1f}%")

    # ------------------------------------------------------------------
    # 9. Save output
    # ------------------------------------------------------------------
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ Augmented data saved to: {output_path}")

    print("=" * 60)
    print("DATA AUGMENTATION COMPLETED")
    print("=" * 60)
    return df


if __name__ == "__main__":
    input_file = r"data\Cleaned Output\Cleaned_January_Acronym_2026.csv"
    output_file = r"data\Cleaned Output\Cleaned_January_Acronym_2026_augmented.csv"

    try:
        df = clean_coverage_data(input_file, output_file)
        print(f"\n✅ SUCCESS – {len(df)} rows with 5 classes.")

        print("\nSample rows:")
        cols = ['Acronym', 'Coverage Status'] if 'Acronym' in df.columns else ['Coverage Status']
        for idx, row in df[cols].head(12).iterrows():
            print(f"  {row.to_dict()}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()