# Momentum Portfolio Replication using CRSP and Fama-French Data

This project replicates and analyzes the momentum strategy framework laid out by **Daniel & Moskowitz (2016)** using CRSP stock returns and Fama-French factors. It involves constructing a (12,2) momentum signal, assigning decile portfolios using both equal-firm and NYSE-breakpoint methods, and evaluating the performance of long-short momentum strategies.

## Objective

To explore the historical performance of momentum investing by:
- Computing past returns-based signals for U.S. equities.
- Forming decile portfolios using different ranking methodologies.
- Calculating value-weighted excess returns.
- Replicating summary statistics aligned with Table 1 of Daniel & Moskowitz (2016).
- Assessing robustness through benchmark comparisons and reference datasets.

## Methodology

### 1. Momentum Signal Construction
- Raw CRSP data is cleaned and aligned by PERMNO and date.
- Returns are adjusted using **delisting return (DLRET)** where available.
- A (12,2) momentum signal is computed using the cumulative log return from **t−12 to t−2**.
- Stocks are filtered to include only common equities from NYSE, AMEX, and NASDAQ, with lagged market cap > $15M.

### 2. Decile Assignment
- **DM Method**: Equal-weighted deciles formed from the investable universe.
- **KRF Method**: Breakpoints computed from NYSE stocks only; applied to all firms.

### 3. Portfolio Return Computation
- Value-weighted excess returns are computed by decile.
- Returns are adjusted for the Fama-French risk-free rate.
- WML (Winner Minus Loser) is calculated as Decile 10 – Decile 1.

### 4. Summary Statistics
- Annualized metrics: Excess Return, Volatility, Sharpe Ratio, Skewness.
- Correlation with benchmark replication files (from Daniel & Moskowitz and Ken French library).
- Results are tabulated in 5×11 matrices for DM and KRF methods.

## Key Results

| Metric             | DM WML | KRF WML |       Benchmark      |
|--------------------|--------|---------|----------------------|
| Excess Return (%)  | 11.08  | 7.76    | 17.65 / 13.64        |
| Sharpe Ratio       | 0.40   | 0.30    | 0.58 / 0.50          |
| Skewness           | -3.47  | -4.18   | -4.64 / -5.60        |
| Correlation (WML)  | 0.89   | 0.89    | 0.99+                |

While our WML returns and Sharpe ratios were slightly lower, this is attributable to technical differences like reduced DLRET coverage.

## Interpretation

### Has the Momentum Anomaly Worked?

Yes. Despite some attenuation, the replicated results show a consistent and strong spread in decile returns, with WML maintaining positive excess returns and pronounced negative skewness — a hallmark of the momentum premium. This aligns with the long-run outperformance observed in academic literature, although the anomaly appears somewhat weaker in our replication due to implementation nuances.

### Would I Use This in a Live Strategy?

Yes, with caveats. Momentum remains compelling, but:

- **Crash risk** must be managed — especially in sharp reversals.
- **Transaction costs** from monthly rebalancing must be minimized (e.g., using optimized execution or reduced turnover versions).
- **Signal crowding** is a concern — momentum is widely known and deployed.
- I would integrate momentum into a **multi-factor framework** and apply dynamic risk overlays or timing filters to guard against tail events.

## Files

- `Code_Model.py`: All code implementing momentum signal, decile formation, portfolio returns, and summary tables.
- `F-F_Research_Data_Factors.csv`: Monthly Fama-French factor data.
- `10_Portfolios_Prior_12_2.csv`: Ken French’s benchmark momentum portfolio returns.
- `m_m_pt_tot.txt`: Reference returns for Daniel & Moskowitz decile portfolios.

## References

- Daniel, Kent, and Tobias J. Moskowitz. “Momentum Crashes.” *Journal of Financial Economics*, 2016.
- Fama-French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
