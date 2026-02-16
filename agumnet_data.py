import pandas as pd

df = pd.read_csv(r'c:\Users\VH0000812\Desktop\Coverage\data\Cleaned Output\Cleaned_January_Acronym_2026.csv')

# Standardize columns for robust matching
df['ACRONYM'] = df['ACRONYM'].astype(str).str.strip().str.upper()
df['Coverage Status'] = df['Coverage Status'].astype(str).str.strip().str.upper()

# Exact acronym matching for PA and ST
pa_acronyms = ['PA', 'MNPA']  # Add more PA-related acronyms if needed
st_acronyms = ['ST']          # Add more ST-related acronyms if needed

mask_pa = df['ACRONYM'].isin(pa_acronyms)
df.loc[mask_pa, 'Coverage Status'] = 'COVERED WITH CONDITIONS (PA REQUIRED)'

mask_st = df['ACRONYM'].isin(st_acronyms)
df.loc[mask_st, 'Coverage Status'] = 'COVERED WITH CONDITIONS (ST REQUIRED)'

# Standardize remaining 'COVERED WITH CONDITION' to 'COVERED WITH CONDITIONS'
df.loc[df['Coverage Status'] == 'COVERED WITH CONDITION', 'Coverage Status'] = 'COVERED WITH CONDITIONS'

# Save to a new file
new_path = r'c:\Users\VH0000812\Desktop\Coverage\data\Cleaned Output\Cleaned_January_Acronym_2026_augmented.csv'
df.to_csv(new_path, index=False)
print(f'Saved to {new_path}')