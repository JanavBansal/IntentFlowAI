# Phase 3 Proposal: Data Maturity & Robustness

## The Problem
We have successfully fixed the critical data leakage, making the model "honest." However, the **"honest" performance is low (Test IC ~0.03)**.
*   **Root Cause:** Our data history is too short (starts late 2022 for many tickers).
*   **Requirement:** Quantitative models need at least one full market cycle (5-10 years) to learn robust patterns.
*   **Limitation:** Our current scraping approach (`Screener.in`) is fragile, lacks deep point-in-time history, and is prone to breaking.

## The Solution: "Data Lake" Architecture
We need to move from "scraping on the fly" to a professional **Local Data Lake**.
1.  **Ingest:** Fetch 10+ years of raw data once from a reliable API.
2.  **Store:** Save as high-performance Parquet files (e.g., `prices.parquet`, `fundamentals.parquet`).
3.  **Train:** Feed the model deep, clean history from this local lake.

## Recommended Data Provider
After researching multiple options (Tiingo, FMP, Alpha Vantage, OpenBB), the best candidate for **NSE (India) + Quant** is:

### **EOD Historical Data (EODHD)**
*   **Why:**
    *   Explicitly supports **NSE (India)** with 20+ years of history.
    *   Provides **Point-in-Time Fundamentals** (crucial to prevent leakage).
    *   Includes delisted tickers (survivorship bias free).
*   **Python Support:** Has an official `eodhd` Python library.
*   **Cost:**
    *   **~20 EUR/month:** "All World" plan (includes NSE prices).
    *   **~50 EUR/month:** "All-In-One" plan (includes Fundamentals).
    *   *Note: This is significantly cheaper than institutional feeds but robust enough for quant trading.*

### Alternative: Tiingo
*   **Pros:** Excellent price accuracy, cheaper (~$30/month).
*   **Cons:** Fundamental data depth for NSE is less explicitly detailed/guaranteed compared to EODHD.

## Action Plan
1.  **Decision:** We need to choose a provider. **EODHD** is the recommendation.
2.  **Implementation:**
    *   Sign up for API key.
    *   Develop `scripts/ingest_history.py` to build the Data Lake.
    *   Refactor `FundamentalDataProvider` to read from the lake.
    *   Retrain model on 2010-2024 data.

## Immediate Question
**Do you want to proceed with a paid subscription to EODHD (or similar), or stick to free/scraping methods?**
*   *Paid:* Professional, robust, allows us to actually improve the model.
*   *Free:* Will severely limit performance and reliability.
