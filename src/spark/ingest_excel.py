import glob
import os
import pandas as pd

files = sorted(glob.glob("data/tfwp_*_en.xlsx"))
print(f"Found {len(files)} Excel files", flush=True)

os.makedirs("data/raw_csv", exist_ok=True)

for f in files:
    print(f"Reading {f} ...", flush=True)
    df = pd.read_excel(f, sheet_name=0, header=1)
    out = os.path.join("data/raw_csv", os.path.basename(f).replace(".xlsx", ".csv"))
    df.to_csv(out, index=False)
    print(f"  -> {out} ({len(df)} rows)", flush=True)

print("Done", flush=True)
