"""
Inspect the structure of a negative LMIA file - may differ from positive.
"""

import pandas as pd

FILE = "data/tfwp_2024q1_neg_en.xlsx"

xls = pd.ExcelFile(FILE)
print("Sheet names:", xls.sheet_names)
print()

raw = pd.read_excel(FILE, sheet_name=0, header=None, nrows=5)
print("First 5 raw rows:")
print(raw)
print()

# Try header=1 same as positive files
df = pd.read_excel(FILE, sheet_name=0, header=1)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()
print(df.head())