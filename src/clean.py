
import pandas as pd
import glob
import re

FILES = sorted(glob.glob("data/tfwp_*_pos_en.xlsx"))

all_dfs = []
for f in FILES:
    df = pd.read_excel(f, sheet_name=0, header=1)
    df["source_file"] = f
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
print(f"Raw combined rows: {len(combined)}")

# Drop footer/note rows - these have no Employer value
combined = combined.dropna(subset=["Employer"]).copy()
print(f"After dropping footer rows: {len(combined)}")

# Extract NOC code (e.g. "2173" from "2173-Software engineers and designers")
combined["NOC_Code"] = combined["Occupation"].str.extract(r"^(\d{4,5})-")
combined["Occupation_Title"] = combined["Occupation"].str.replace(r"^\d{4,5}-", "", regex=True)

# Extract fiscal quarter from the source filename for time-based analysis
combined["Quarter"] = combined["source_file"].str.extract(r"tfwp_(\d{4}q\d)_pos_en")

# Clean whitespace in text columns
text_cols = ["Employer", "Address", "Occupation_Title", "Program Stream", "Province/Territory"]
for col in text_cols:
    combined[col] = combined[col].str.strip()

# Make sure numeric columns are actually numeric
combined["Approved LMIAs"] = pd.to_numeric(combined["Approved LMIAs"], errors="coerce")
combined["Approved Positions"] = pd.to_numeric(combined["Approved Positions"], errors="coerce")

print()
print("Sample of cleaned data:")
print(combined[["Quarter", "Province/Territory", "Employer", "NOC_Code", "Occupation_Title", "Approved Positions"]].head(10))

print()
print("Rows missing a valid NOC code (worth checking):", combined["NOC_Code"].isna().sum())

# Save the cleaned combined dataset for use in the notebook
combined.to_csv("data/lmia_combined_clean.csv", index=False)
print()
print("Saved cleaned dataset to data/lmia_combined_clean.csv")
print()
print("Rows with missing NOC code:")
print(combined[combined["NOC_Code"].isna()][["Occupation", "Quarter"]])
# Drop the handful of rows with no occupation data at all
combined = combined.dropna(subset=["NOC_Code"]).copy()
print(f"After dropping missing-NOC rows: {len(combined)}")