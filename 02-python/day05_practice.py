import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

D = r"E:\DataAnalytics\Datasets"
customers = pd.read_csv(D + r"\customers.csv")
sales = pd.read_csv(D + r"\sales.csv", parse_dates=["OrderDate"])

rev = sales.groupby("CustomerKey")["SalesAmount"].sum().rename("Revenue").reset_index()
m = customers.merge(rev, on="CustomerKey", how="left")
m["Revenue"] = m["Revenue"].fillna(0)

print("\n=== 1 Top 5 occupations by average income ===")
print(m.groupby("EnglishOccupation")["YearlyIncome"].mean().round(0).sort_values(ascending=False).head(5))

print("\n=== 2 Age bands: customers and average revenue ===")
m["Age"] = (pd.Timestamp("2014-01-01") - pd.to_datetime(m["BirthDate"])).dt.days // 365
m["AgeBand"] = pd.cut(m["Age"], bins=[17, 30, 45, 60, 150], labels=["18-30", "31-45", "46-60", "61+"])
print(m.groupby("AgeBand", observed=True).agg(customers=("CustomerKey", "count"), avg_revenue=("Revenue", "mean")).round(0))

print("\n=== 3 YearlyIncome outliers (IQR rule) ===")
q1, q3 = m["YearlyIncome"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n = int(((m["YearlyIncome"] < lo) | (m["YearlyIncome"] > hi)).sum())
print("lower bound:", lo, " upper bound:", hi)
print("outliers:", n, " share of customers:", round(100 * n / len(m), 1), "%")

print("\n=== 4 Missing Title ===")
print(round(100 * customers["Title"].isna().mean(), 1), "% of customers have no Title")

print("\n=== 5 Pivot: monthly revenue by year ===")
sales["Year"] = sales["OrderDate"].dt.year
sales["Month"] = sales["OrderDate"].dt.month
print(sales.pivot_table(index="Month", columns="Year", values="SalesAmount", aggfunc="sum").round(0))