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


head("1 UnitPrice: mean, median, skew")
up = sa["UnitPrice"]
print("mean:", round(up.mean(), 2), " median:", round(up.median(), 2), " skew:", round(stats.skew(up), 2))
if up.mean() > up.median():
    print("Mean is above median: right-skewed. Most sales are cheap items, with a few expensive products pulling the mean up.")
else:
    print("Mean is not above median: no strong right skew.")
print("share of rows priced under 100:", round(100 * (up < 100).mean(), 1), "%")
print("share of rows priced over 1000:", round(100 * (up > 1000).mean(), 1), "%")

head("2 P(no car | Manual) vs P(no car) overall")
manual = cu["EnglishOccupation"] == "Manual"
nocar = cu["NumberCarsOwned"] == 0
p_all = nocar.mean()
p_man = nocar[manual].mean()
print("P(no car) overall:", round(p_all, 3))
print("P(no car | Manual):", round(p_man, 3))
print("lift:", round(p_man / p_all, 2), "(above 1 means Manual workers are more likely to own no car)")

head("3 Share within 1 SD of mean YearlyIncome")
inc = cu["YearlyIncome"]
mu, sd = inc.mean(), inc.std()
share = 100 * ((inc - mu).abs() <= sd).mean()
print("mean:", round(mu), " sd:", round(sd), " skew:", round(stats.skew(inc), 2))
print("share within 1 sd:", round(share, 1), "% (normal says 68.3%)  gap:", round(share - 68.3, 1), "points")
print("Income comes in set bands and is right-skewed, so it is not a bell curve.")

head("4 Binomial: at least 10 of 15 customers are female")
p_f = (cu["Gender"] == "F").mean()
print("female share p:", round(p_f, 3))
print("P(at least 10 of 15):", round(stats.binom.sf(9, 15, p_f), 4))
print("P(exactly 8 of 15):", round(stats.binom.pmf(8, 15, p_f), 4))
print("expected females in 15:", round(15 * p_f, 2))

head("5 Central Limit Theorem on SalesAmount")
pop = sa["SalesAmount"].to_numpy()
print("population mean:", round(pop.mean(), 1), " sd:", round(pop.std(), 1), " skew:", round(stats.skew(pop), 2))
print("sample_size  mean_of_means  sd_of_means  theory_sd  skew_of_means")
for s in [1, 10, 50, 200]:
    means = rng.choice(pop, size=(5000, s)).mean(axis=1)
    print(f"{s:10d} {means.mean():14.1f} {means.std():12.1f} {pop.std() / np.sqrt(s):10.1f} {stats.skew(means):13.2f}")
found = None
for s in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200]:
    means = rng.choice(pop, size=(5000, s)).mean(axis=1)
    if abs(stats.skew(means)) < 0.3:
        found = s
        break
print("smallest sample size tested where skew of the means is under 0.3:", found)
print("The sd of the means follows the theory column, which is sd divided by the square root of n.")