"""
Predicts LMIA approval vs denial using scikit-learn.
Uses broad occupation category (first 2 digits of NOC), province,
program stream, incorporate status, AND the employer's historical
denial rate (calculated only from earlier quarters, to avoid leakage)
as features - NOT the raw Employer name (too high-cardinality).

Uses a TIME-BASED train/test split (train on earlier quarters, test
on later ones) rather than a random split, since we're testing whether
past employer behavior predicts future outcomes - a random split would
leak future information into training.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/lmia_all_labeled.csv")

# Broad occupation category = first 2 digits of NOC code (e.g. "21" = tech/science)
df["NOC_Broad"] = df["NOC_Code"].astype(str).str[:2]

target = "Decision"

# Sort by quarter so we can do a proper time-based split (avoid leakage)
quarter_order = sorted(df["Quarter"].unique())
print("Quarters available:", quarter_order)

# Use first 6 quarters to CALCULATE employer history, last 3 to TEST on
# (this ensures we never use future information to predict the past)
history_quarters = quarter_order[:6]
test_quarters = quarter_order[6:]

history_df = df[df["Quarter"].isin(history_quarters)]

# Employer's historical denial rate, calculated ONLY from earlier quarters
employer_history = (
    history_df.groupby("Employer")[target]
    .apply(lambda x: (x == "Denied").mean())
    .rename("Employer_Historical_Denial_Rate")
)

df = df.merge(employer_history, on="Employer", how="left")
# Employers with no prior history get the overall average denial rate
overall_history_denial_rate = history_df[target].eq("Denied").mean()
df["Employer_Historical_Denial_Rate"] = df["Employer_Historical_Denial_Rate"].fillna(
    overall_history_denial_rate
)

print(f"Employers with historical data: {employer_history.notna().sum()}")

categorical_features = ["Province/Territory", "Program Stream", "NOC_Broad", "Incorporate Status"]
numeric_features = ["Employer_Historical_Denial_Rate"]
features = categorical_features + numeric_features

df = df.dropna(subset=features + [target])

train_df = df[df["Quarter"].isin(history_quarters)]
test_df = df[df["Quarter"].isin(test_quarters)]

X_train, y_train = train_df[features], (train_df[target] == "Denied").astype(int)
X_test, y_test = test_df[features], (test_df[target] == "Denied").astype(int)

print(f"Training rows: {len(X_train)} (quarters {history_quarters})")
print(f"Test rows: {len(X_test)} (quarters {test_quarters})")
print(f"Denial rate in training set: {y_train.mean()*100:.1f}%")
print(f"Denial rate in test set: {y_test.mean()*100:.1f}%")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
}

for name, model in models.items():
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Approved", "Denied"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Approved", "Denied"], yticklabels=["Approved", "Denied"])
    plt.title(f"Confusion Matrix - {name} (with employer history)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"notebooks/confusion_matrix_{safe_name}_v2.png", dpi=150)
    plt.close()

print()
print("Saved confusion matrix charts to notebooks/ (v2 = with employer history feature)")