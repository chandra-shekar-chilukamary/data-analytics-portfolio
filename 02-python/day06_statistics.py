import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 30)

D = r"E:\DataAnalytics\Datasets"
cu = pd.read_csv(D + r"\customers_clean.csv")
sa = pd.read_csv(D + r"\sales.csv", parse_dates=["OrderDate"])
rng = np.random.default_rng(42)


def head(t):
    print("\n=== " + t + " ===")


head("1 Descriptive statistics")
rows = []
for c in ["YearlyIncome", "Revenue", "Age", "TotalChildren"]:
    x = cu[c]
    rows.append([c, x.mean(), x.median(), x.mode().iloc[0], x.std(), 100 * x.std() / x.mean(), stats.skew(x), stats.kurtosis(x)])
t = pd.DataFrame(rows, columns=["column", "mean", "median", "mode", "std", "cv_pct", "skew", "kurtosis"]).set_index("column")
print(t.round(2))
print("Mean above median and skew above 0 means a right-skewed column.")

head("2 Shape of customer revenue (text histogram)")
edges = [0, 500, 1000, 1500, 2000, 3000, 5000, 1e9]
labels = ["0-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", "3000-5000", "5000+"]
cnt, _ = np.histogram(cu["Revenue"], bins=edges)
for lab, n in zip(labels, cnt):
    print(f"{lab:>10} {n:6d} {100 * n / len(cu):5.1f}%  " + "#" * int(60 * n / cnt.max()))
print("skew before log:", round(stats.skew(cu["Revenue"]), 2), " after log1p:", round(stats.skew(np.log1p(cu["Revenue"])), 2))

head("3 Probability, conditional probability, Bayes")
male = cu["Gender"] == "M"
prof = cu["EnglishOccupation"] == "Professional"
high = cu["YearlyIncome"] >= 70000
p_m, p_prof, p_high = male.mean(), prof.mean(), high.mean()
print("P(male):", round(p_m, 3), " P(Professional):", round(p_prof, 3), " P(income>=70000):", round(p_high, 3))
print("P(male and Professional):", round((male & prof).mean(), 3), " if independent:", round(p_m * p_prof, 3))
p_high_g_prof = high[prof].mean()
print("P(high income | Professional):", round(p_high_g_prof, 3), " lift vs overall:", round(p_high_g_prof / p_high, 2))
print("Bayes P(Professional | high income):", round(p_high_g_prof * p_prof / p_high, 3), " direct check:", round(prof[high].mean(), 3))

head("4 Normal distribution on Age")
mu, sd = cu["Age"].mean(), cu["Age"].std()
print("mean age:", round(mu, 1), " sd:", round(sd, 1))
for k, expect in [(1, 68.3), (2, 95.4), (3, 99.7)]:
    share = 100 * ((cu["Age"] - mu).abs() <= k * sd).mean()
    print(f"within {k} sd: {share:.1f}% (normal says {expect}%)")
print("z-score of age 60:", round((60 - mu) / sd, 2), " percentile (normal):", round(100 * stats.norm.cdf(60, mu, sd), 1))
print("P(30 <= age <= 45) normal:", round(stats.norm.cdf(45, mu, sd) - stats.norm.cdf(30, mu, sd), 3), " actual:", round(cu["Age"].between(30, 45).mean(), 3))

head("5 Binomial: customers owning 2 or more cars")
p = (cu["NumberCarsOwned"] >= 2).mean()
n = 20
print("p =", round(p, 3), " n =", n, " expected count:", round(n * p, 1), " sd:", round(np.sqrt(n * p * (1 - p)), 2))
print("P(exactly 8):", round(stats.binom.pmf(8, n, p), 4))
print("P(at least 10):", round(1 - stats.binom.cdf(9, n, p), 4))
sim = rng.binomial(n, p, 100000)
print("simulation P(exactly 8):", round((sim == 8).mean(), 4), " P(at least 10):", round((sim >= 10).mean(), 4))

head("6 Poisson: orders per day in 2013")
d13 = sa[sa["OrderDate"].dt.year == 2013].groupby("OrderDate")["SalesOrderNumber"].nunique()
daily = d13.reindex(pd.date_range("2013-01-01", "2013-12-31"), fill_value=0)
lam, var = daily.mean(), daily.var()
print("mean orders/day:", round(lam, 1), " variance:", round(var, 1), " variance/mean:", round(var / lam, 1))
k = int(round(lam))
print(f"share of days with <= {k} orders: actual {(daily <= k).mean():.3f}, Poisson {stats.poisson.cdf(k, lam):.3f}")
print("days with zero orders:", int((daily == 0).sum()), " Poisson P(0):", round(stats.poisson.pmf(0, lam), 6))
print("A variance far above the mean means demand is trending or clustered, so Poisson is too tight.")

head("7 Central Limit Theorem on customer revenue")
pop = cu["Revenue"].to_numpy()
print("population mean:", round(pop.mean(), 1), " sd:", round(pop.std(), 1), " skew:", round(stats.skew(pop), 2))
print("sample_size  mean_of_means  sd_of_means  theory_sd  skew_of_means")
for s in [1, 5, 30, 100]:
    means = rng.choice(pop, size=(5000, s)).mean(axis=1)
    print(f"{s:10d} {means.mean():14.1f} {means.std():12.1f} {pop.std() / np.sqrt(s):10.1f} {stats.skew(means):13.2f}")

head("8 Covariance and correlation: income vs revenue")
r, pv = stats.pearsonr(cu["YearlyIncome"], cu["Revenue"])
rho, ps = stats.spearmanr(cu["YearlyIncome"], cu["Revenue"])
print("Pearson r:", round(r, 3), " Spearman rho:", round(rho, 3), " covariance:", round(np.cov(cu["YearlyIncome"], cu["Revenue"])[0, 1], 0))
print("R squared:", round(r ** 2, 4), "(share of revenue variation explained by income alone)")