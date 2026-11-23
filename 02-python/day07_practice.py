import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

D = r"E:\DataAnalytics\Datasets"
cu = pd.read_csv(D + r"\customers_clean.csv")


def head(t):
    print("\n=== " + t + " ===")


head("1 95 percent CI for mean age and for the share of homeowners")
ag = cu["Age"]
lo, hi = stats.t.interval(0.95, len(ag) - 1, loc=ag.mean(), scale=ag.std(ddof=1) / np.sqrt(len(ag)))
print("mean age:", round(ag.mean(), 2), " 95% CI:", round(lo, 2), "to", round(hi, 2))
own = int((cu["HouseOwnerFlag"] == 1).sum())
lo, hi = proportion_confint(own, len(cu), alpha=0.05, method="wilson")
print("homeowner share:", round(own / len(cu), 3), " 95% CI:", round(lo, 3), "to", round(hi, 3))

head("2 Revenue: married vs single")
a = cu.loc[cu["MaritalStatus"] == "M", "Revenue"]
b = cu.loc[cu["MaritalStatus"] == "S", "Revenue"]
t, p = stats.ttest_ind(a, b, equal_var=False)
u, pu = stats.mannwhitneyu(a, b)
d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
print("married mean:", round(a.mean(), 1), " single mean:", round(b.mean(), 1))
print("Welch p:", f"{p:.4g}", " Mann-Whitney p:", f"{pu:.4g}", " Cohen d:", round(d, 3))
print("Significant" if p < 0.05 else "Not significant", "- look at d for the size:", "small" if abs(d) < 0.3 else "meaningful")

head("3 Chi-square: marital status vs home ownership")
ct = pd.crosstab(cu["MaritalStatus"], cu["HouseOwnerFlag"])
chi2, p, dof, exp = stats.chi2_contingency(ct)
v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
print(ct)
print("chi2:", round(chi2, 1), " p:", f"{p:.4g}", " Cramers V:", round(v, 3))

head("4 ANOVA: revenue by commute distance")
groups = [g["Revenue"].to_numpy() for _, g in cu.groupby("CommuteDistance")]
f, p = stats.f_oneway(*groups)
gm = cu["Revenue"].mean()
eta = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / ((cu["Revenue"] - gm) ** 2).sum()
print(cu.groupby("CommuteDistance")["Revenue"].agg(["count", "mean"]).round(1))
print("F:", round(f, 1), " p:", f"{p:.4g}", " eta squared:", round(eta, 4))

head("5 Sample size planning (80 percent power, alpha 0.05)")
for base, new in [(0.05, 0.055), (0.05, 0.06), (0.05, 0.07)]:
    es = proportion_effectsize(new, base)
    n = NormalIndPower().solve_power(es, power=0.8, alpha=0.05, ratio=1.0)
    print(f"{100 * base:.1f}% -> {100 * new:.1f}%: need {int(np.ceil(n)):,} per group")
print("Smaller lifts need far bigger tests.")