import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

pd.set_option("display.width", 170)
D = r"E:\DataAnalytics\Datasets"
cu = pd.read_csv(D + r"\customers_clean.csv")
rng = np.random.default_rng(7)


def head(t):
    print("\n=== " + t + " ===")


def cramers(ct):
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return chi2, p, dof, v


head("1 Confidence interval and one-sample t-test (revenue per customer)")
x = cu["Revenue"]
n = len(x)
mean = x.mean()
se = x.std(ddof=1) / np.sqrt(n)
lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
print("n:", n, " mean:", round(mean, 1), " 95% CI:", round(lo, 1), "to", round(hi, 1))
t, p = stats.ttest_1samp(x, 1500)
print("H0: true mean = 1500   t:", round(t, 2), " p-value:", f"{p:.4g}")
print("Decision at 0.05:", "reject H0" if p < 0.05 else "fail to reject H0")

head("2 Two-sample test: revenue, men vs women")
a = cu.loc[cu["Gender"] == "M", "Revenue"]
b = cu.loc[cu["Gender"] == "F", "Revenue"]
t, p = stats.ttest_ind(a, b, equal_var=False)
diff = a.mean() - b.mean()
sed = np.sqrt(a.var() / len(a) + b.var() / len(b))
d = diff / np.sqrt((a.var() + b.var()) / 2)
u, pu = stats.mannwhitneyu(a, b)
print("mean M:", round(a.mean(), 1), " mean F:", round(b.mean(), 1), " difference:", round(diff, 1))
print("95% CI of difference:", round(diff - 1.96 * sed, 1), "to", round(diff + 1.96 * sed, 1))
print("Welch t:", round(t, 2), " p:", f"{p:.4g}", " Mann-Whitney p:", f"{pu:.4g}", " Cohen d:", round(d, 3))

head("3 Two-sample test: homeowners vs non-owners")
a = cu.loc[cu["HouseOwnerFlag"] == 1, "Revenue"]
b = cu.loc[cu["HouseOwnerFlag"] == 0, "Revenue"]
t, p = stats.ttest_ind(a, b, equal_var=False)
d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
print("mean owners:", round(a.mean(), 1), " non-owners:", round(b.mean(), 1), " Welch p:", f"{p:.4g}", " Cohen d:", round(d, 3))

head("4 Chi-square: are two categories linked?")
for c1, c2 in [("Gender", "EnglishOccupation"), ("EnglishEducation", "EnglishOccupation")]:
    ct = pd.crosstab(cu[c1], cu[c2])
    chi2, p, dof, v = cramers(ct)
    print(c1, "vs", c2, " chi2:", round(chi2, 1), " dof:", dof, " p:", f"{p:.4g}", " Cramers V:", round(v, 3))
print("V near 0.1 is weak, near 0.3 is moderate, 0.5 or more is strong.")

head("5 ANOVA: revenue across occupations")
groups = [g["Revenue"].to_numpy() for _, g in cu.groupby("EnglishOccupation")]
f, p = stats.f_oneway(*groups)
h, pk = stats.kruskal(*groups)
gm = x.mean()
ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
sst = ((x - gm) ** 2).sum()
print("ANOVA F:", round(f, 1), " p:", f"{p:.4g}", " Kruskal p:", f"{pk:.4g}", " eta squared:", round(ssb / sst, 4))
print("Eta squared is the share of revenue variation explained by occupation.")

head("6 Correlation test: income vs revenue")
r, p = stats.pearsonr(cu["YearlyIncome"], cu["Revenue"])
rho, ps = stats.spearmanr(cu["YearlyIncome"], cu["Revenue"])
print("Pearson r:", round(r, 3), " p:", f"{p:.4g}", " Spearman rho:", round(rho, 3), " R squared:", round(r ** 2, 4))

head("7 A/B test on conversion (simulated: A 5.0 percent, B 5.6 percent)")
n_ab = 4000
ca = int(rng.binomial(n_ab, 0.050))
cb = int(rng.binomial(n_ab, 0.056))
pa, pb = ca / n_ab, cb / n_ab
z, p = proportions_ztest([cb, ca], [n_ab, n_ab])
sed = np.sqrt(pa * (1 - pa) / n_ab + pb * (1 - pb) / n_ab)
print("A:", ca, "/", n_ab, "=", round(100 * pa, 2), "%   B:", cb, "/", n_ab, "=", round(100 * pb, 2), "%")
print("lift:", round(100 * (pb - pa), 2), "points   95% CI:", round(100 * (pb - pa - 1.96 * sed), 2), "to", round(100 * (pb - pa + 1.96 * sed), 2))
print("z:", round(z, 2), " p:", f"{p:.4g}", " ->", "significant" if p < 0.05 else "not significant")
es = proportion_effectsize(0.056, 0.05)
need = NormalIndPower().solve_power(es, power=0.8, alpha=0.05, ratio=1.0)
pw = NormalIndPower().power(es, nobs1=n_ab, alpha=0.05, ratio=1.0)
print("sample size per group for 80 percent power:", int(np.ceil(need)), " power with 4000 per group:", round(pw, 2))

head("8 Peeking: A/A tests with no real difference, stopping at the first p below 0.05")
sims, N = 2000, 2000
A = rng.binomial(1, 0.05, (sims, N)).cumsum(axis=1)
B = rng.binomial(1, 0.05, (sims, N)).cumsum(axis=1)
checks = [400, 800, 1200, 1600, 2000]


def zp(c1, c2, k):
    pp = (c1 + c2) / (2 * k)
    se2 = np.sqrt(pp * (1 - pp) * 2 / k)
    zz = (c1 / k - c2 / k) / np.where(se2 == 0, np.nan, se2)
    return np.nan_to_num(2 * (1 - stats.norm.cdf(np.abs(zz))), nan=1.0)


pv = np.column_stack([zp(A[:, k - 1], B[:, k - 1], k) for k in checks])
print("false positive rate, test once at the end:", round((pv[:, -1] < 0.05).mean(), 3))
print("false positive rate, peek 5 times and stop early:", round((pv < 0.05).any(axis=1).mean(), 3))

head("9 Multiple testing: 20 tests on pure noise")
fam, k, m = 1000, 20, 100
a = rng.normal(size=(fam, k, m))
b = rng.normal(size=(fam, k, m))
pv = stats.ttest_ind(a, b, axis=2).pvalue
print("average false positives per set of 20 tests:", round((pv < 0.05).sum(axis=1).mean(), 2))
print("sets with at least one false positive:", round((pv < 0.05).any(axis=1).mean(), 3))
print("after Bonferroni (alpha / 20):", round((pv < 0.05 / k).any(axis=1).mean(), 3))

head("10 Bootstrap 95 percent CI for the median revenue")
arr = x.to_numpy()
boots = np.array([np.median(rng.choice(arr, size=len(arr))) for _ in range(1000)])
print("median:", round(np.median(arr), 1), " 95% CI:", round(np.percentile(boots, 2.5), 1), "to", round(np.percentile(boots, 97.5), 1))