"""
Analysis 2: Who, what, where - top occupations, provinces, and employers
by approved positions across the full 2024-2026 dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/lmia_combined_clean.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# --- Top 15 occupations ---
top_occupations = (
    df.groupby("Occupation_Title")["Approved Positions"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)
print("Top 15 occupations by approved positions:")
print(top_occupations)
print()

# --- Top 10 provinces ---
top_provinces = (
    df.groupby("Province/Territory")["Approved Positions"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print("Top 10 provinces by approved positions:")
print(top_provinces)
print()

# --- Top 15 employers ---
top_employers = (
    df.groupby("Employer")["Approved Positions"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)
print("Top 15 employers by approved positions:")
print(top_employers)
print()

# --- Program stream breakdown ---
stream_breakdown = (
    df.groupby("Program Stream")["Approved Positions"]
    .sum()
    .sort_values(ascending=False)
)
print("Approved positions by program stream:")
print(stream_breakdown)
print()

# --- Charts ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_occupations.sort_values().plot(kind="barh", ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Top 15 Occupations by Approved Positions")
axes[0, 0].set_xlabel("Approved Positions")

top_provinces.sort_values().plot(kind="barh", ax=axes[0, 1], color="seagreen")
axes[0, 1].set_title("Top 10 Provinces by Approved Positions")
axes[0, 1].set_xlabel("Approved Positions")

top_employers.sort_values().plot(kind="barh", ax=axes[1, 0], color="darkorange")
axes[1, 0].set_title("Top 15 Employers by Approved Positions")
axes[1, 0].set_xlabel("Approved Positions")

stream_breakdown.plot(kind="bar", ax=axes[1, 1], color="mediumpurple")
axes[1, 1].set_title("Approved Positions by Program Stream")
axes[1, 1].set_ylabel("Approved Positions")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("notebooks/top_breakdowns.png", dpi=150)
print("Saved chart to notebooks/top_breakdowns.png")
print()
print("=" * 60)
print("FILTERED: High Wage + Global Talent Stream only")
print("(the segment relevant to skilled/tech positions)")
print("=" * 60)

skilled = df[df["Program Stream"].isin(["High Wage", "Global Talent Stream"])]

top_skilled_occupations = (
    skilled.groupby("Occupation_Title")["Approved Positions"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)
print()
print("Top 15 occupations WITHIN High Wage / Global Talent Stream:")
print(top_skilled_occupations)

print()
print(f"Total approved positions in this segment: {skilled['Approved Positions'].sum():.0f}")
print(f"That's {skilled['Approved Positions'].sum() / df['Approved Positions'].sum() * 100:.1f}% of all approved positions")
print()
print("=" * 60)
print("FILTERED: Tech/Science sector (NOC codes starting with 21)")
print("=" * 60)

tech = df[df["NOC_Code"].astype(str).str.startswith("21")]

top_tech_occupations = (
    tech.groupby("Occupation_Title")["Approved Positions"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)
print()
print("Top occupations in NOC 21xxx (tech/science) sector:")
print(top_tech_occupations)

print()
print(f"Total approved positions in tech/science sector: {tech['Approved Positions'].sum():.0f}")
print(f"That's {tech['Approved Positions'].sum() / df['Approved Positions'].sum() * 100:.2f}% of all approved positions")

print()
top_tech_provinces = (
    tech.groupby("Province/Territory")["Approved Positions"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print("Top provinces for tech/science sector:")
print(top_tech_provinces)
