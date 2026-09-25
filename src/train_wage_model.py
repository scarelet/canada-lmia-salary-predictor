"""
Predicts median hourly wage using broader features (province, program stream,
2-digit occupation category) rather than the exact occupation code - this
makes it a genuine regression/generalization task instead of a lookup,
since multiple specific occupations share the same broad category input
but have different real wages.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

lmia = pd.read_csv("data/lmia_all_labeled.csv")
wages = pd.read_csv("data/wages_cleaned.csv", dtype={"NOC_4digit": str})
wages["NOC_4digit"] = wages["NOC_4digit"].str.zfill(4)

# Only look at Approved positions - wage relevance is clearest there
lmia = lmia[lmia["Decision"] == "Approved"].copy()

# Match LMIA's NOC code to the wage table's 4-digit key
# (zfill(5) first to restore any leading zero lost when NOC_Code was
# read back from CSV, matching the original NOC 2021 5-digit format)
lmia["NOC_4digit"] = lmia["NOC_Code"].astype(str).str.zfill(5).str[:4]

# Map LMIA quarter to the wage data year (2026q1 uses 2025 wages - most recent available)
def quarter_to_wage_year(q):
    year = int(q[:4])
    return 2025 if year >= 2025 else 2024

lmia["Wage_Year"] = lmia["Quarter"].apply(quarter_to_wage_year)

merged = lmia.merge(
    wages.rename(columns={"Province": "Province/Territory"}),
    on=["NOC_4digit", "Province/Territory", "Wage_Year"],
    how="left"
)

print(f"Total approved LMIA rows: {len(lmia)}")
print(f"Rows with a matched wage: {merged['Median_Wage_Hourly'].notna().sum()}")
print(f"Match rate: {merged['Median_Wage_Hourly'].notna().mean()*100:.1f}%")

merged = merged.dropna(subset=["Median_Wage_Hourly"])

# Broad occupation category = first 2 digits (NOT the full code - see docstring)
merged["NOC_Broad"] = merged["NOC_4digit"].str[:2]

features = ["Province/Territory", "Program Stream", "NOC_Broad"]
target = "Median_Wage_Hourly"

merged = merged.dropna(subset=features + [target])

X = merged[features]
y = merged[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), features)]
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = (r2, mae)

    print()
    print(f"{name}: R\u00b2 = {r2:.3f}, MAE = ${mae:.2f}/hr")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Median Wage ($/hr)")
    plt.ylabel("Predicted Median Wage ($/hr)")
    plt.title(f"{name}: Predicted vs Actual Wage")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"notebooks/wage_prediction_{safe_name}.png", dpi=150)
    plt.close()

print()
print("Saved prediction charts to notebooks/")