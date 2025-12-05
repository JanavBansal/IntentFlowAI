# Survivorship Bias Audit Report

**Generated:** 2025-12-05 03:09
**Training Period:** 2015-01-01 to 2024-12-31

---

## Risk Assessment

**Overall Risk Score:** 5508.6/100 (HIGH)

| Metric | Value |
|--------|-------|
| Total Tickers | 462 |
| Incomplete Coverage | 100.0% |
| Late Additions | 100.0% |
| Early Dropouts | 0.0% |
| Universe Mismatch | 0.4% |

---

## Disappeared Tickers

Tickers that stopped trading before the end of training period:

| Ticker | Last Date | Days Covered |
|--------|-----------|--------------|

---

## Late Additions

Tickers added after significant portion of training period:

| Ticker | First Date | Days Late |
|--------|------------|-----------|
| 3MINDIA | 2020-12-07 | 2167 |
| AARTIDRUGS | 2020-12-07 | 2167 |
| AARTIIND | 2020-12-07 | 2167 |
| AAVAS | 2020-12-07 | 2167 |
| ABB | 2020-12-07 | 2167 |
| ABBOTINDIA | 2020-12-07 | 2167 |
| ABCAPITAL | 2020-12-07 | 2167 |
| ABFRL | 2020-12-07 | 2167 |
| ACC | 2020-12-07 | 2167 |
| ADANIENT | 2020-12-07 | 2167 |
| ADANIGREEN | 2020-12-07 | 2167 |
| ADANIPORTS | 2020-12-07 | 2167 |
| ADVENZYMES | 2020-12-07 | 2167 |
| AFFLE | 2020-12-07 | 2167 |
| AIAENG | 2020-12-07 | 2167 |
| AJANTPHARM | 2020-12-07 | 2167 |
| AKZOINDIA | 2020-12-07 | 2167 |
| ALEMBICLTD | 2020-12-07 | 2167 |
| ALKEM | 2020-12-07 | 2167 |
| ALKYLAMINE | 2020-12-07 | 2167 |
| ... | ... | (442 more) |

---

## Known Delisted Stocks

| Ticker | Reason | In Price Data | In Universe |
|--------|--------|---------------|-------------|
| HDFC | Merged with HDFC Bank | No | No |
| RCOM | Bankruptcy | No | No |
| DHFL | Resolution | No | No |
| YESBANK | Still trading but restructured | Yes | Yes |
| RELCAPITAL | Resolution | No | No |
| INFRATEL | Merged with Bharti Airtel | No | No |
| IBULHSGFIN | Restructured | No | No |

---

## Recommendations

⚠️ **HIGH RISK** - Significant survivorship bias potential

1. Consider using point-in-time constituent data
2. Include delisted stocks in training data
3. Be cautious with backtest results
4. Apply haircut to expected returns