# =============================================
# Name: Tanvi Johri
# Student ID: 806555315
# Assignment: Problem Set 3 (QAM MFE)
# File: PS3_806555315.py
# =============================================


#Load libraries
import pandas as pd
import numpy as np

# Load CRSP data downloaded from WRDS
CRSP_Stocks = pd.read_csv("h01nlruknvekazyg.csv")
CRSP_Stocks["date"] = pd.to_datetime(CRSP_Stocks["date"], errors="coerce")
CRSP_Stocks

# Question 1
def PS3_Q1(CRSP_Stocks):
    
    df = CRSP_Stocks.copy()

    # Step 1: Parse all relevant numeric columns
    for col in ["RET", "DLRET", "PRC", "SHROUT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 2: Apply total return formula with DLRET adjustment
    df["Ret"] = np.where(
        df["DLRET"].notna(),
        (1 + df["RET"].fillna(0)) * (1 + df["DLRET"]) - 1,
        df["RET"]
    )

    # Step 3: Keep only common stocks from major exchanges
    df = df[df["SHRCD"].isin([10, 11]) & df["EXCHCD"].isin([1, 2, 3])]

    # Step 4: Calculate market cap in millions
    df["MktCap"] = df["PRC"].abs() * df["SHROUT"] / 1000

    # Step 5: Add time columns
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["PERMNO", "date"])

    # Step 6: Lagged values
    df["lag_Mkt_Cap"] = df.groupby("PERMNO")["MktCap"].shift(1)
    df["PRC_lag13"] = df.groupby("PERMNO")["PRC"].shift(13)

    # Step 7: Construct momentum signal (12,2) using log returns
    with np.errstate(divide='ignore'):
        df["logret"] = np.where((1 + df["Ret"]) > 0, np.log1p(df["Ret"]), np.nan)

    rolling_logret = (
        df.groupby("PERMNO")["logret"]
        .rolling(window=11, min_periods=11)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["Ranking_Ret"] = rolling_logret
    df["Ranking_Ret"] = df.groupby("PERMNO")["Ranking_Ret"].shift(1)  # exclude t−1

    # Step 8: Final filters
    df = df[df["PRC_lag13"].notna()]
    df = df[df["lag_Mkt_Cap"] > 15]
    df = df[(df["Year"] >= 1927) & (df["Year"] <= 2024)]

    # Step 9: Prepare output
    output_1 = df[["Year", "Month", "PERMNO", "EXCHCD", "lag_Mkt_Cap", "Ret", "Ranking_Ret"]].dropna().reset_index(drop=True)

    print("-----------PS3 - Output 1------------")
    print(output_1)

    return output_1

CRSP_Stocks_Momentum = PS3_Q1(CRSP_Stocks)

# Question 2
def PS3_Q2(CRSP_Stocks_Momentum):
    df = CRSP_Stocks_Momentum.copy()
    df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    df = df.sort_values(by=["Year", "Month", "Ranking_Ret"]).reset_index(drop=True)

    decile_dm, decile_krf = [], []

    for date, group in df.groupby("date"):
        group = group.dropna(subset=["Ranking_Ret"]).copy()

        # Use lag_Mkt_Cap > 15 ONLY for forming breakpoints (not for assignment)
        sort_universe = group[group["lag_Mkt_Cap"] > 15]

        if len(sort_universe) >= 10:
            # DM deciles based on Ranking_Ret of investable universe
            try:
                group["DM_decile"] = pd.qcut(sort_universe["Ranking_Ret"], 10, labels=False, duplicates="drop").reindex(group.index) + 1
            except:
                group["DM_decile"] = np.nan

            # KRF deciles using NYSE breakpoints from investable NYSE stocks
            nyse = sort_universe[sort_universe["EXCHCD"] == 1]
            if len(nyse) >= 100:
                try:
                    breakpoints = pd.qcut(nyse["Ranking_Ret"], 10, retbins=True, duplicates="drop")[1]
                    group["KRF_decile"] = pd.cut(group["Ranking_Ret"], bins=breakpoints, labels=False, include_lowest=True) + 1
                except:
                    group["KRF_decile"] = np.nan
            else:
                group["KRF_decile"] = np.nan
        else:
            group["DM_decile"] = np.nan
            group["KRF_decile"] = np.nan

        decile_dm.append(group[["PERMNO", "date", "DM_decile"]])
        decile_krf.append(group[["PERMNO", "date", "KRF_decile"]])

    # Merge decile assignments back into full df
    df = df.merge(pd.concat(decile_dm), on=["PERMNO", "date"], how="left")
    df = df.merge(pd.concat(decile_krf), on=["PERMNO", "date"], how="left")
    output_2 = df[["Year", "Month", "PERMNO", "lag_Mkt_Cap", "Ret", "DM_decile", "KRF_decile", "EXCHCD"]]

    print("-----------PS3 - Output 2------------")
    print(output_2)

    # Final output
    return output_2

CRSP_Stocks_Momentum_decile = PS3_Q2(CRSP_Stocks_Momentum)


#Question 3 
ff = pd.read_csv("F-F_Research_Data_Factors.CSV", skiprows=3)
ff_df = ff.rename(columns={ff.columns[0]: 'Date_String'})
ff_df = ff_df.dropna(subset=['Date_String'])
ff_df = ff_df[ff_df['Date_String'].astype(str).str.match(r'^\d{6}$')].copy()

# Format columns and dates
ff_df['Date'] = pd.to_datetime(ff_df['Date_String'], format='%Y%m')
ff_df['Year'] = ff_df['Date'].dt.year
ff_df['Month'] = ff_df['Date'].dt.month
ff_df['Mkt-RF'] = ff_df['Mkt-RF'].astype(float) / 100
ff_df['RF'] = ff_df['RF'].astype(float) / 100

ff_df = ff_df[['Year', 'Month', 'Mkt-RF', 'SMB', 'HML', 'RF']]

def PS3_Q3(CRSP_Stocks_Momentum_decile, ff_df):
    df = CRSP_Stocks_Momentum_decile.copy()
    ff = ff_df.copy()

    # Drop rows with missing required fields
    df = df.dropna(subset=["Ret", "lag_Mkt_Cap", "DM_decile", "KRF_decile"])

    # Enforce integer type for deciles
    df["DM_decile"] = df["DM_decile"].astype(int)
    df["KRF_decile"] = df["KRF_decile"].astype(int)

    # (Optional) Confirm lagged market cap cutoff used in Q2
    df = df[df["lag_Mkt_Cap"] > 15]

    # Merge in risk-free rate
    df = df.merge(ff[["Year", "Month", "RF"]].rename(columns={"RF": "Rf"}), on=["Year", "Month"], how="left")


    # Compute excess return
    df["ExcessRet"] = df["Ret"] - df["Rf"]

    # DM method: value-weighted excess return by decile
    dm_ret = (
        df.groupby(["Year", "Month", "DM_decile"], group_keys=False)
        .apply(lambda x: np.average(x["ExcessRet"], weights=x["lag_Mkt_Cap"]))
        .reset_index(name="DM_Ret")
        .rename(columns={"DM_decile": "decile"})
    )

    # KRF method: value-weighted excess return by decile
    krf_ret = (
        df.groupby(["Year", "Month", "KRF_decile"], group_keys=False)
        .apply(lambda x: np.average(x["ExcessRet"], weights=x["lag_Mkt_Cap"]))
        .reset_index(name="KRF_Ret")
        .rename(columns={"KRF_decile": "decile"})
    )

    # Merge decile return tables
    merged_ret = pd.merge(dm_ret, krf_ret, on=["Year", "Month", "decile"], how="outer")

    # Merge RF again for completeness in output
    merged_ret = pd.merge(merged_ret, ff[["Year", "Month", "RF"]].rename(columns={"RF": "Rf"}), on=["Year", "Month"], how="left")

    # Clip sample range
    merged_ret = merged_ret[(merged_ret["Year"] >= 1927) & (merged_ret["Year"] <= 2024)]
    output_3 = merged_ret.sort_values(by=["Year", "Month", "decile"]).reset_index(drop=True)

    print("-----------PS3 - Output 3------------")
    print(output_3)

    # Final sort
    return output_3


CRSP_Stocks_Momentum_returns = PS3_Q3(CRSP_Stocks_Momentum_decile, ff_df)

# Question 4
from scipy.stats import skew

def PS3_Q4(CRSP_Stocks_Momentum_returns, dm_ref_df):
    q3_df = CRSP_Stocks_Momentum_returns
   
    # Pivot DM_Ret to wide format
    q3_df["date"] = pd.to_datetime(q3_df["Year"].astype(str) + "-" + q3_df["Month"].astype(str))
    dm_pivot = q3_df.pivot(index="date", columns="decile", values="DM_Ret")
    dm_pivot["WML"] = dm_pivot[10] - dm_pivot[1]

    # Reorder columns
    ordered_cols = [i for i in range(1, 11)] + ["WML"]
    dm_pivot = dm_pivot[[*ordered_cols]]
    dm_pivot.columns = [str(c) for c in dm_pivot.columns]

    # Summary statistics
    metrics = {}
    for col in dm_pivot.columns:
        r = dm_pivot[col].dropna()
        mean = r.mean()
        std = r.std(ddof=1)
        excess_ret = round(mean * 12 * 100, 2)
        vol = round(std * np.sqrt(12) * 100, 2)
        sharpe = round((mean / std) * np.sqrt(12), 2) if std > 0 else np.nan
        log_r = np.log1p(r[r > -1])
        skewness = round(skew(log_r), 2)
        metrics[col] = [excess_ret, vol, sharpe, skewness]

    summary_matrix = pd.DataFrame(metrics, index=["Excess Return", "Volatility", "Sharpe Ratio", "Skewness"])

    # Reference matrix for correlation (use full sample)
    ref = dm_ref_df[[0, 1, 2]].copy()
    ref.columns = ["date", "decile", "ret"]
    ref["date"] = pd.to_datetime(ref["date"], format="%Y%m%d")
    ref["date"] = ref["date"].values.astype("datetime64[M]")
    dm_wide = ref.pivot(index="date", columns="decile", values="ret")
    dm_wide["WML"] = dm_wide[10] - dm_wide[1]
    dm_wide = dm_wide[[*range(1, 11), "WML"]]
    dm_wide.columns = [str(c) for c in dm_wide.columns]

    # Correlation (use full common date range)
    common = dm_pivot.index.intersection(dm_wide.index)
    corr_row = dm_pivot.loc[common].corrwith(dm_wide.loc[common])
    summary_matrix.loc["Correlation"] = corr_row.round(2)
    output_4 = summary_matrix.reset_index(names="Metric")

    print("-----------PS3 - Output 4------------")
    print(output_4)

    return output_4


dm_ref_df = pd.read_csv("m_m_pt_tot.txt", delim_whitespace=True, header=None)
PS3_Q4(CRSP_Stocks_Momentum_returns, dm_ref_df)


#Question 5
column_names = ["Date"] + [str(i) for i in range(1, 11)]

# Read from line 11, rename columns, and clean missing values
krf_df = pd.read_csv('10_Portfolios_Prior_12_2.csv', skiprows=11, names=column_names)

# Drop any footer or -99 values
krf_df = krf_df[krf_df["Date"].apply(lambda x: str(x).isdigit())].copy()
krf_df = krf_df.replace([-99.99, -999], np.nan).dropna()

# Convert date and format returns
krf_df["Year"] = krf_df["Date"].astype(str).str[:4].astype(int)
krf_df["Month"] = krf_df["Date"].astype(str).str[4:6].astype(int)

# Ensure all returns are numeric and in decimal form
for i in range(1, 11):
    krf_df[str(i)] = pd.to_numeric(krf_df[str(i)], errors="coerce") / 100

# Melt to long format
krf_long = krf_df.melt(
    id_vars=["Year", "Month"],
    value_vars=[str(i) for i in range(1, 11)],
    var_name="decile",
    value_name="KRF_Ret"
)
krf_long["decile"] = krf_long["decile"].astype(int)


def PS3_Q5(q3_df: pd.DataFrame, krf_ref_df: pd.DataFrame) -> pd.DataFrame:

    # Compute excess return
    q3_df = q3_df.copy()
    q3_df["Excess_KRF_Ret"] = q3_df["KRF_Ret"] - q3_df["Rf"]
    q3_df["date"] = pd.to_datetime(q3_df["Year"].astype(str) + "-" + q3_df["Month"].astype(str) + "-01")
    krf_pivot = q3_df.pivot(index="date", columns="decile", values="Excess_KRF_Ret")
    krf_pivot["WML"] = krf_pivot[10] - krf_pivot[1]
    krf_pivot.columns = krf_pivot.columns.astype(str)
    ordered_cols = [str(i) for i in range(1, 11)] + ["WML"]
    krf_pivot = krf_pivot[ordered_cols]

    # Compute summary metrics
    metrics = {}
    for col in ordered_cols:
        returns = krf_pivot[col].dropna()
        excess_ret = round(returns.mean() * 12 * 100, 2)
        vol = round(returns.std() * (12 ** 0.5) * 100, 2)
        sharpe = round(excess_ret / vol, 2) if vol > 0 else np.nan
        skewness = round(skew(np.log(1+returns)), 2)
        metrics[col] = [excess_ret, vol, sharpe, skewness]

    summary_matrix = pd.DataFrame(metrics, index=["Excess Return", "Volatility", "Sharpe Ratio", "Skewness"])

    # Prepare Ken French reference benchmark
    krf_ref_df = krf_ref_df.copy()
    krf_ref_df["date"] = pd.to_datetime(krf_ref_df["Year"].astype(str) + "-" + krf_ref_df["Month"].astype(str) + "-01")
    krf_ref_df = krf_ref_df.drop_duplicates(subset=["date", "decile"])
    krf_benchmark = krf_ref_df.pivot(index="date", columns="decile", values="KRF_Ret")
    krf_benchmark["WML"] = krf_benchmark[10] - krf_benchmark[1]
    krf_benchmark.columns = krf_benchmark.columns.astype(str)
    krf_benchmark = krf_benchmark[ordered_cols]

    # Correlation row
    common_dates = krf_pivot.index.intersection(krf_benchmark.index)
    correlation_row = krf_pivot.loc[common_dates].corrwith(krf_benchmark.loc[common_dates])
    correlation_row = correlation_row.round(2)
    summary_matrix.loc["Correlation"] = correlation_row

    output_5 = summary_matrix.reset_index(names="Metric")

    print("-----------PS3 - Output 5------------")
    print(output_5)

    return output_5


PS3_Q5(CRSP_Stocks_Momentum_returns, krf_long)





