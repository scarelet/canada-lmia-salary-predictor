import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/lmia_combined_clean.csv")

# Aggregate by quarter
quarterly = df.groupby("Quarter").agg(
    total_approved_positions=("Approved Positions", "sum"),
    total_approved_lmias=("Approved LMIAs", "sum"),
    unique_employers=("Employer", "nunique")
).reset_index()

# Quarter strings sort correctly alphabetically here (2024q1, 2024q2... 2026q1)
quarterly = quarterly.sort_values("Quarter")

print(quarterly)

# Calculate the actual percentage drop for the README
first = quarterly.iloc[0]["total_approved_positions"]
last = quarterly.iloc[-1]["total_approved_positions"]
pct_change = ((last - first) / first) * 100
print(f"\nChange from {quarterly.iloc[0]['Quarter']} to {quarterly.iloc[-1]['Quarter']}: {pct_change:.1f}%")

# Plot it
plt.figure(figsize=(10, 6))
plt.plot(quarterly["Quarter"], quarterly["total_approved_positions"], marker="o", linewidth=2)
plt.title("Approved LMIA Positions by Quarter (2024 Q1 - 2026 Q1)")
plt.xlabel("Quarter")
plt.ylabel("Total Approved Positions")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/quarterly_trend.png", dpi=150)
print("\nSaved chart to notebooks/quarterly_trend.png")