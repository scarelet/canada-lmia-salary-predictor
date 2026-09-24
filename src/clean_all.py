"""
Combines positive AND negative LMIA files into one labeled dataset
for a classification model: will this LMIA be approved or denied?
"""

import pandas as pd
import glob

def load_and_label(pattern, decision_label, count_prefix):
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        df = pd.read_excel(f, sheet_name=0, header=1)
        df["source_file"] = f
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # Standardize column names (positive uses "Approved", negative uses "Requested")
    combined = combined.rename(columns={
        f"{count_prefix} LMIAs": "LMIAs",
        f"{count_prefix} Positions": "Positions"
    })

    combined["Decision"] = decision_label
    return combined

positive = load_and_label("data/tfwp_*_pos_en.xlsx", "Approved", "Approved")
negative = load_and_label("data/tfwp_*_neg_en.xlsx", "Denied", "Requested")

print(f"Positive rows: {len(positive)}")
print(f"Negative rows: {len(negative)}")

all_data = pd.concat([positive, negative], ignore_index=True)

# Drop footer/note rows (no Employer value)
all_data = all_data.dropna(subset=["Employer"]).copy()

# Extract NOC code and clean occupation title
all_data["NOC_Code"] = all_data["Occupation"].str.extract(r"^(\d{4,5})-")
all_data["Occupation_Title"] = all_data["Occupation"].str.replace(r"^\d{4,5}-", "", regex=True)
all_data = all_data.dropna(subset=["NOC_Code"]).copy()

# Extract quarter
all_data["Quarter"] = all_data["source_file"].str.extract(r"tfwp_(\d{4}q\d)_(?:pos|neg)_en")

# Clean whitespace
for col in ["Employer", "Address", "Occupation_Title", "Program Stream", "Province/Territory"]:
    all_data[col] = all_data[col].str.strip()

all_data["LMIAs"] = pd.to_numeric(all_data["LMIAs"], errors="coerce")
all_data["Positions"] = pd.to_numeric(all_data["Positions"], errors="coerce")

print()
print(f"Total labeled rows: {len(all_data)}")
print()
print("Decision breakdown:")
print(all_data["Decision"].value_counts())
print()
print("Decision breakdown as percentage:")
print(all_data["Decision"].value_counts(normalize=True) * 100)

all_data.to_csv("data/lmia_all_labeled.csv", index=False)
print()
print("Saved to data/lmia_all_labeled.csv")