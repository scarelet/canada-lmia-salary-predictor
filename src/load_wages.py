"""
Loads and cleans the Job Bank wage data, filtering to province-level rows,
normalizing annual wages to an hourly-equivalent rate (some occupations are
reported annually, others hourly - mixing them would corrupt the model),
and aggregating to a 4-digit NOC key to join against the LMIA dataset.
"""

import pandas as pd

PROVINCE_MAP = {
    "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island",
    "NS": "Nova Scotia",
    "NB": "New Brunswick",
    "QC": "Quebec",
    "ON": "Ontario",
    "MB": "Manitoba",
    "SK": "Saskatchewan",
    "AB": "Alberta",
    "BC": "British Columbia",
    "YT": "Yukon",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
}

HOURS_PER_YEAR = 2080  # standard full-time approximation (40 hrs/week x 52 weeks)

def load_wage_file(path, year):
    df = pd.read_csv(path)

    # Keep only provincial-level rows (4-char ER code, e.g. "ER10")
    df = df[df["ER_Code_Code_RE"].str.len() == 4].copy()

    df["Province"] = df["prov"].map(PROVINCE_MAP)
    df = df.dropna(subset=["Province"])

    df["NOC_4digit"] = df["NOC_CNP"].str.replace("NOC_", "", regex=False).str[:4]

    # Normalize: if Annual_Wage_Flag == 1, this wage is annual - convert to hourly
    is_annual = df["Annual_Wage_Flag_Salaire_annuel"] == 1
    df["Median_Wage_Hourly"] = df["Median_Wage_Salaire_Median"]
    df.loc[is_annual, "Median_Wage_Hourly"] = df.loc[is_annual, "Median_Wage_Salaire_Median"] / HOURS_PER_YEAR

    df["Wage_Year"] = year

    result = df[["NOC_4digit", "Province", "Median_Wage_Hourly", "Wage_Year"]].dropna(subset=["Median_Wage_Hourly"])

    # Multiple 5-digit NOC codes collapse into one 4-digit key - average them
    result = result.groupby(["NOC_4digit", "Province", "Wage_Year"], as_index=False)["Median_Wage_Hourly"].mean()

    return result

wages_2024 = load_wage_file("data/wages_2024.csv", 2024)
wages_2025 = load_wage_file("data/wages_2025.csv", 2025)

all_wages = pd.concat([wages_2024, wages_2025], ignore_index=True)

print(f"Total wage rows (provincial, normalized to hourly): {len(all_wages)}")
print()
print("Sample - check for consistent hourly-scale values now:")
print(all_wages.head(10))
print()
print("Wage range check (should all look like reasonable hourly rates now):")
print(all_wages["Median_Wage_Hourly"].describe())

all_wages.to_csv("data/wages_cleaned.csv", index=False)
print()
print("Saved to data/wages_cleaned.csv")