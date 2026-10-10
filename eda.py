
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/Churn_Modelling.csv")

# Create folder for charts
Path("reports/figures").mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("DATASET INFORMATION")
print(df.info())

print("\nNUMERICAL SUMMARY")
print(df.describe().round(2))

print("\nCHURN BY GEOGRAPHY")
print(
    df.groupby("Geography")["Exited"]
    .agg(["count", "mean"])
    .assign(churn_percent=lambda x: x["mean"] * 100)
    .round(2)
)

print("\nCHURN BY GENDER")
print(
    df.groupby("Gender")["Exited"]
    .agg(["count", "mean"])
    .assign(churn_percent=lambda x: x["mean"] * 100)
    .round(2)
)

print("\nCHURN BY ACTIVE MEMBERSHIP")
print(
    df.groupby("IsActiveMember")["Exited"]
    .agg(["count", "mean"])
    .assign(churn_percent=lambda x: x["mean"] * 100)
    .round(2)
)

# 1. Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Exited", hue="Exited", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("reports/figures/churn_distribution.png", dpi=150)
plt.close()
# 2. Churn by age
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df, x="Age", hue="Exited",
    bins=30, multiple="layer", element="step"
)
plt.title("Customer Age Distribution by Churn")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_age.png", dpi=150)
plt.close()
# 3. Churn by geography
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Geography", y="Exited", errorbar=None)
plt.title("Churn Rate by Geography")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_geography.png", dpi=150)
plt.close()

# 4. Churn by active membership
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="IsActiveMember", y="Exited", errorbar=None)
plt.title("Churn Rate by Active Membership")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_activity.png", dpi=150)
plt.close()

print("\nEDA completed. Charts saved in reports/figures/")
