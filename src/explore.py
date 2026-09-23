import pandas as pd
import glob

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

files = sorted(glob.glob("data/tfwp_*_pos_en.xlsx"))
print(f"Found {len(files)} files:")
for f in files:
    print(" ", f)
print()

all_dfs = []
for f in files:
    df = pd.read_excel(f, sheet_name=0, header=1)
    df["source_file"] = f  # track which quarter each row came from
    all_dfs.append(df)
    print(f"{f}: {df.shape[0]} rows, columns: {df.columns.tolist()}")

combined = pd.concat(all_dfs, ignore_index=True)
print()
print("Combined shape:", combined.shape)
print()
print("Missing values per column:")
print(combined.isna().sum())

# Look at the actual rows with missing data - are they footer notes, not real data?
print()
print("Rows with missing Employer:")
print(combined[combined["Employer"].isna()])