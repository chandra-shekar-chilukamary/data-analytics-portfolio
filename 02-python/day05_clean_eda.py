import numpy as np
import pandas as pd

D = r"E:\DataAnalytics\Datasets"
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

cu = pd.read_csv(D + r"\customers.csv")
sa = pd.read_csv(D + r"\sales.csv", parse_dates=["OrderDate"])


def head(t):
    print("\n=== " + t + " ===")


head("1 Shape and data types")
print("customers:", cu.shape, " sales:", sa.shape)
print(cu.dtypes.value_counts())
print(sa.dtypes)

head("2 Missing values (customers)")
miss = cu.isna().sum()
print(pd.DataFrame({"missing": miss, "pct": (100 * miss / len(cu)).round(1)}).query("missing > 0").sort_values("missing", ascending=False))

head("3 Duplicates")
print("duplicate customer rows:", cu.duplicated().sum())
print("duplicate CustomerKey:", cu["CustomerKey"].duplicated().sum())
print("duplicate sales rows:", sa.duplicated().sum())

head("4 Fix types and fill missing values")
cu["BirthDate"] = pd.to_datetime(cu["BirthDate"])
cu["DateFirstPurchase"] = pd.to_datetime(cu["DateFirstPurchase"])
cu["Age"] = (pd.Timestamp("2014-01-01") - cu["BirthDate"]).dt.days // 365
cu["MiddleName"] = cu["MiddleName"].fillna("Unknown")
rng = np.random.default_rng(1)
holes = rng.choice(cu.index, size=int(0.05 * len(cu)), replace=False)
cu["YearlyIncomeTest"] = cu["YearlyIncome"]
cu.loc[holes, "YearlyIncomeTest"] = np.nan
print("holes added:", cu["YearlyIncomeTest"].isna().sum())
print("true mean income:", round(cu["YearlyIncome"].mean()))
print("mean after holes:", round(cu["YearlyIncomeTest"].mean()))
cu["IncomeFilled"] = cu["YearlyIncomeTest"].fillna(cu["YearlyIncomeTest"].median())
grp_median = cu.groupby("EnglishOccupation")["YearlyIncomeTest"].transform("median")
cu["IncomeByOccupation"] = cu["YearlyIncomeTest"].fillna(grp_median)
print("error with overall median:", round(abs(cu["IncomeFilled"].mean() - cu["YearlyIncome"].mean())))
print("error with group median:", round(abs(cu["IncomeByOccupation"].mean() - cu["YearlyIncome"].mean())))

head("5 Outliers in SalesAmount")
q1, q3 = sa["SalesAmount"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = sa[(sa["SalesAmount"] < lo) | (sa["SalesAmount"] > hi)]
print("IQR bounds:", round(lo, 2), round(hi, 2))
print("IQR outliers:", len(out), "=", round(100 * len(out) / len(sa), 1), "% of rows")
z = (sa["SalesAmount"] - sa["SalesAmount"].mean()) / sa["SalesAmount"].std()
print("z-score above 3:", int((z.abs() > 3).sum()))
print(sa.nlargest(5, "SalesAmount")[["SalesOrderNumber", "ProductKey", "SalesAmount"]])

head("6 Revenue and income by occupation")
rev = sa.groupby("CustomerKey")["SalesAmount"].sum().rename("Revenue").reset_index()
m = cu.merge(rev, on="CustomerKey", how="left")
print("customers with no sales:", int(m["Revenue"].isna().sum()))
m["Revenue"] = m["Revenue"].fillna(0)
print(m.groupby("EnglishOccupation").agg(customers=("CustomerKey", "count"), avg_income=("YearlyIncome", "mean"), revenue=("Revenue", "sum")).round(0).sort_values("revenue", ascending=False))

head("7 Education by gender (row percentages)")
print(pd.crosstab(m["EnglishEducation"], m["Gender"], normalize="index").round(2))

head("8 Correlation with revenue")
print(m[["YearlyIncome", "TotalChildren", "NumberCarsOwned", "Age", "Revenue"]].corr().round(2)["Revenue"])

head("9 Monthly revenue trend")
sa["Month"] = sa["OrderDate"].dt.to_period("M")
mt = sa.groupby("Month")["SalesAmount"].sum()
print(mt.tail(8).round(0))
print("best month:", mt.idxmax(), round(mt.max()))

head("10 Save cleaned data")
m.drop(columns=["YearlyIncomeTest", "IncomeFilled", "IncomeByOccupation"]).to_csv(D + r"\customers_clean.csv", index=False)
print("saved customers_clean.csv", m.shape)