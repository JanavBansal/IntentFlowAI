"""
EODHD Fundamental Data Provider

Parses EODHD JSON files containing comprehensive fundamental data:
- Income Statement (quarterly/yearly)
- Balance Sheet (quarterly/yearly)
- Cash Flow Statement (quarterly/yearly)
- Valuation metrics
- ESG scores
- Insider transactions
- Splits and dividends

Ensures point-in-time correctness using filing_date for available_date calculation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EODHDConfig:
    """Configuration for EODHD data parsing."""
    
    # Directory containing EODHD JSON files
    data_dir: Path = Path("data/raw/eodhd")
    
    # Output cache directory
    cache_dir: Path = Path("data/cache/fundamentals")
    
    # Reporting delay for point-in-time safety (days)
    # Companies typically file within 45 days of quarter end
    reporting_delay_days: int = 45
    
    # Use filing_date if available, else apply reporting_delay
    use_filing_date: bool = True


class EODHDProvider:
    """
    Parse and provide fundamental data from EODHD JSON files.
    
    Usage:
        provider = EODHDProvider()
        
        # Parse all files and create parquet
        df = provider.parse_all_tickers()
        provider.save_parquet(df)
        
        # Get fundamentals for a specific ticker
        fund = provider.get_fundamentals("RELIANCE", as_of_date="2024-01-15")
    """
    
    def __init__(self, config: Optional[EODHDConfig] = None):
        self.config = config or EODHDConfig()
        self._fundamentals_cache: Optional[pd.DataFrame] = None
    
    def parse_all_tickers(self) -> pd.DataFrame:
        """
        Parse all EODHD JSON files and return consolidated DataFrame.
        
        Returns:
            DataFrame with columns:
            - symbol, date, report_date, available_date
            - Income statement metrics
            - Balance sheet metrics
            - Cash flow metrics
            - Valuation metrics
            - ESG scores
        """
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"EODHD data directory not found: {data_dir}")
        
        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")
        
        logger.info(f"Parsing {len(json_files)} EODHD JSON files...")
        
        all_records = []
        failed_files = []
        
        for json_path in json_files:
            try:
                records = self._parse_single_file(json_path)
                all_records.extend(records)
            except Exception as e:
                failed_files.append((json_path.name, str(e)))
                logger.warning(f"Failed to parse {json_path.name}: {e}")
        
        if not all_records:
            raise ValueError("No records extracted from EODHD files")
        
        df = pd.DataFrame(all_records)
        
        # Ensure date columns are datetime
        for col in ["date", "report_date", "available_date", "filing_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Sort by symbol and date
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        
        logger.info(
            f"Parsed EODHD data",
            extra={
                "total_records": len(df),
                "unique_symbols": df["symbol"].nunique(),
                "date_range": f"{df['date'].min()} to {df['date'].max()}",
                "failed_files": len(failed_files),
            }
        )
        
        if failed_files:
            logger.warning(f"Failed to parse {len(failed_files)} files: {failed_files[:5]}")
        
        return df
    
    def _parse_single_file(self, json_path: Path) -> List[Dict[str, Any]]:
        """Parse a single EODHD JSON file."""
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract symbol from filename or General section
        symbol = json_path.stem
        if "General" in data:
            symbol = data["General"].get("Code", symbol)
        
        records = []
        
        # Parse quarterly financials (primary source)
        financials = data.get("Financials", {})
        
        income_quarterly = financials.get("Income_Statement", {}).get("quarterly", {})
        balance_quarterly = financials.get("Balance_Sheet", {}).get("quarterly", {})
        cashflow_quarterly = financials.get("Cash_Flow", {}).get("quarterly", {})
        
        # Get all unique dates from financials
        all_dates = set(income_quarterly.keys()) | set(balance_quarterly.keys()) | set(cashflow_quarterly.keys())
        
        for date_str in all_dates:
            record = self._create_record(
                symbol=symbol,
                date_str=date_str,
                income=income_quarterly.get(date_str, {}),
                balance=balance_quarterly.get(date_str, {}),
                cashflow=cashflow_quarterly.get(date_str, {}),
                highlights=data.get("Highlights", {}),
                valuation=data.get("Valuation", {}),
                esg=data.get("ESGScores", {}),
                shares_stats=data.get("SharesStats", {}),
            )
            if record:
                records.append(record)
        
        return records
    
    def _create_record(
        self,
        symbol: str,
        date_str: str,
        income: Dict,
        balance: Dict,
        cashflow: Dict,
        highlights: Dict,
        valuation: Dict,
        esg: Dict,
        shares_stats: Dict,
    ) -> Optional[Dict[str, Any]]:
        """Create a single fundamental record."""
        
        try:
            report_date = pd.to_datetime(date_str)
        except Exception:
            return None
        
        # Determine available_date (point-in-time safety)
        filing_date = income.get("filing_date") or balance.get("filing_date")
        if filing_date and self.config.use_filing_date:
            try:
                available_date = pd.to_datetime(filing_date)
            except Exception:
                available_date = report_date + timedelta(days=self.config.reporting_delay_days)
        else:
            available_date = report_date + timedelta(days=self.config.reporting_delay_days)
        
        record = {
            "symbol": symbol,
            "date": report_date,
            "report_date": report_date,
            "filing_date": filing_date,
            "available_date": available_date,
            "period_type": "quarterly",
        }
        
        # Income Statement metrics
        record.update(self._extract_income_metrics(income))
        
        # Balance Sheet metrics
        record.update(self._extract_balance_metrics(balance))
        
        # Cash Flow metrics
        record.update(self._extract_cashflow_metrics(cashflow))
        
        # Derived ratios
        record.update(self._compute_ratios(record, highlights, valuation))
        
        # ESG scores
        record.update(self._extract_esg(esg))
        
        # Shares stats
        record.update(self._extract_shares_stats(shares_stats))
        
        return record
    
    def _extract_income_metrics(self, income: Dict) -> Dict[str, Any]:
        """Extract income statement metrics."""
        return {
            "revenue": self._safe_float(income.get("totalRevenue")),
            "gross_profit": self._safe_float(income.get("grossProfit")),
            "operating_income": self._safe_float(income.get("operatingIncome")),
            "ebitda": self._safe_float(income.get("ebitda")),
            "net_income": self._safe_float(income.get("netIncome")),
            "eps": self._safe_float(income.get("eps")),
            "eps_diluted": self._safe_float(income.get("epsDiluted")),
            "cost_of_revenue": self._safe_float(income.get("costOfRevenue")),
            "rd_expense": self._safe_float(income.get("researchDevelopment")),
            "sga_expense": self._safe_float(income.get("sellingGeneralAdministrative")),
            "interest_expense": self._safe_float(income.get("interestExpense")),
            "income_tax_expense": self._safe_float(income.get("incomeTaxExpense")),
        }
    
    def _extract_balance_metrics(self, balance: Dict) -> Dict[str, Any]:
        """Extract balance sheet metrics."""
        return {
            "total_assets": self._safe_float(balance.get("totalAssets")),
            "total_liabilities": self._safe_float(balance.get("totalLiab")),
            "total_equity": self._safe_float(balance.get("totalStockholderEquity")),
            "total_debt": self._safe_float(balance.get("totalDebt") or balance.get("shortLongTermDebtTotal")),
            "cash": self._safe_float(balance.get("cash")),
            "cash_and_equivalents": self._safe_float(balance.get("cashAndShortTermInvestments")),
            "current_assets": self._safe_float(balance.get("totalCurrentAssets")),
            "current_liabilities": self._safe_float(balance.get("totalCurrentLiabilities")),
            "inventory": self._safe_float(balance.get("inventory")),
            "receivables": self._safe_float(balance.get("netReceivables")),
            "payables": self._safe_float(balance.get("accountsPayable")),
            "retained_earnings": self._safe_float(balance.get("retainedEarnings")),
            "goodwill": self._safe_float(balance.get("goodWill")),
            "intangibles": self._safe_float(balance.get("intangibleAssets")),
        }
    
    def _extract_cashflow_metrics(self, cashflow: Dict) -> Dict[str, Any]:
        """Extract cash flow metrics."""
        return {
            "operating_cashflow": self._safe_float(cashflow.get("totalCashFromOperatingActivities")),
            "investing_cashflow": self._safe_float(cashflow.get("totalCashflowsFromInvestingActivities")),
            "financing_cashflow": self._safe_float(cashflow.get("totalCashFromFinancingActivities")),
            "free_cashflow": self._safe_float(cashflow.get("freeCashFlow")),
            "capex": self._safe_float(cashflow.get("capitalExpenditures")),
            "dividends_paid": self._safe_float(cashflow.get("dividendsPaid")),
            "depreciation": self._safe_float(cashflow.get("depreciation")),
            "stock_repurchase": self._safe_float(cashflow.get("salePurchaseOfStock")),
        }
    
    def _compute_ratios(self, record: Dict, highlights: Dict, valuation: Dict) -> Dict[str, Any]:
        """Compute financial ratios."""
        ratios = {}
        
        # Profitability ratios
        revenue = record.get("revenue") or 0
        net_income = record.get("net_income") or 0
        gross_profit = record.get("gross_profit") or 0
        operating_income = record.get("operating_income") or 0
        total_equity = record.get("total_equity") or 0
        total_assets = record.get("total_assets") or 0
        
        if revenue > 0:
            ratios["gross_margin"] = gross_profit / revenue
            ratios["operating_margin"] = operating_income / revenue
            ratios["net_margin"] = net_income / revenue
        
        if total_equity > 0:
            ratios["roe"] = (net_income * 4) / total_equity  # Annualized quarterly
        
        if total_assets > 0:
            ratios["roa"] = (net_income * 4) / total_assets  # Annualized quarterly
        
        # Leverage ratios
        total_debt = record.get("total_debt") or 0
        if total_equity > 0:
            ratios["debt_to_equity"] = total_debt / total_equity
        
        if total_assets > 0:
            ratios["debt_to_assets"] = total_debt / total_assets
        
        # Liquidity ratios
        current_assets = record.get("current_assets") or 0
        current_liabilities = record.get("current_liabilities") or 0
        inventory = record.get("inventory") or 0
        
        if current_liabilities > 0:
            ratios["current_ratio"] = current_assets / current_liabilities
            ratios["quick_ratio"] = (current_assets - inventory) / current_liabilities
        
        # Cash flow quality
        operating_cf = record.get("operating_cashflow") or 0
        if net_income != 0:
            ratios["cf_to_net_income"] = operating_cf / net_income
        
        # Accruals (lower is better quality)
        if total_assets > 0:
            ratios["accruals_ratio"] = (net_income - operating_cf) / total_assets
        
        # Valuation from highlights/valuation section
        ratios["pe_ratio"] = self._safe_float(highlights.get("PERatio") or valuation.get("TrailingPE"))
        ratios["pb_ratio"] = self._safe_float(valuation.get("PriceBookMRQ"))
        ratios["ps_ratio"] = self._safe_float(valuation.get("PriceSalesTTM"))
        ratios["peg_ratio"] = self._safe_float(highlights.get("PEGRatio"))
        ratios["ev_ebitda"] = self._safe_float(valuation.get("EnterpriseValueEbitda"))
        ratios["ev_revenue"] = self._safe_float(valuation.get("EnterpriseValueRevenue"))
        ratios["dividend_yield"] = self._safe_float(highlights.get("DividendYield"))
        ratios["book_value"] = self._safe_float(highlights.get("BookValue"))
        ratios["market_cap"] = self._safe_float(highlights.get("MarketCapitalization"))
        
        return ratios
    
    def _extract_esg(self, esg: Dict) -> Dict[str, Any]:
        """Extract ESG scores."""
        return {
            "esg_total": self._safe_float(esg.get("TotalEsg")),
            "esg_percentile": self._safe_float(esg.get("TotalEsgPercentile")),
            "esg_environment": self._safe_float(esg.get("EnvironmentScore")),
            "esg_social": self._safe_float(esg.get("SocialScore")),
            "esg_governance": self._safe_float(esg.get("GovernanceScore")),
        }
    
    def _extract_shares_stats(self, shares: Dict) -> Dict[str, Any]:
        """Extract shares statistics."""
        return {
            "shares_outstanding": self._safe_float(shares.get("SharesOutstanding")),
            "shares_float": self._safe_float(shares.get("SharesFloat")),
            "percent_insiders": self._safe_float(shares.get("PercentInsiders")),
            "percent_institutions": self._safe_float(shares.get("PercentInstitutions")),
            "shares_short": self._safe_float(shares.get("SharesShort")),
            "short_ratio": self._safe_float(shares.get("ShortRatio")),
        }
    
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        except (ValueError, TypeError):
            return None
    
    def save_parquet(self, df: pd.DataFrame, filename: str = "eodhd_full.parquet") -> Path:
        """Save DataFrame to parquet file."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = cache_dir / filename
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Saved EODHD fundamentals to {output_path}")
        return output_path
    
    def load_parquet(self, filename: str = "eodhd_full.parquet") -> pd.DataFrame:
        """Load fundamentals from parquet file."""
        cache_path = Path(self.config.cache_dir) / filename
        
        if not cache_path.exists():
            raise FileNotFoundError(f"EODHD parquet not found at {cache_path}. Run parse_all_tickers() first.")
        
        df = pd.read_parquet(cache_path)
        
        # Cache for faster lookups
        self._fundamentals_cache = df
        
        return df
    
    def get_fundamentals(
        self,
        symbol: str,
        as_of_date: Optional[str | datetime | pd.Timestamp] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get fundamentals for a symbol as of a specific date (point-in-time safe).
        
        Args:
            symbol: Stock symbol
            as_of_date: Date to get fundamentals as of (uses latest if None)
            
        Returns:
            Dictionary of fundamental metrics or None if not found
        """
        if self._fundamentals_cache is None:
            try:
                self.load_parquet()
            except FileNotFoundError:
                logger.warning("Fundamentals parquet not found. Call parse_all_tickers() first.")
                return None
        
        df = self._fundamentals_cache
        symbol_df = df[df["symbol"] == symbol]
        
        if symbol_df.empty:
            return None
        
        if as_of_date is not None:
            as_of_date = pd.to_datetime(as_of_date)
            # Only include records that were available by as_of_date
            symbol_df = symbol_df[symbol_df["available_date"] <= as_of_date]
        
        if symbol_df.empty:
            return None
        
        # Get the most recent record
        latest = symbol_df.sort_values("available_date").iloc[-1]
        return latest.to_dict()
    
    def get_fundamentals_batch(
        self,
        symbols: List[str],
        as_of_date: Optional[str | datetime | pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get fundamentals for multiple symbols (point-in-time safe).
        
        Args:
            symbols: List of stock symbols
            as_of_date: Date to get fundamentals as of
            
        Returns:
            DataFrame with one row per symbol (most recent available data)
        """
        if self._fundamentals_cache is None:
            self.load_parquet()
        
        df = self._fundamentals_cache
        df = df[df["symbol"].isin(symbols)]
        
        if as_of_date is not None:
            as_of_date = pd.to_datetime(as_of_date)
            df = df[df["available_date"] <= as_of_date]
        
        if df.empty:
            return pd.DataFrame()
        
        # Get most recent record per symbol
        idx = df.groupby("symbol")["available_date"].idxmax()
        return df.loc[idx].reset_index(drop=True)


def get_eodhd_provider(data_dir: Optional[str] = None) -> EODHDProvider:
    """
    Get configured EODHD provider.
    
    Args:
        data_dir: Optional path to EODHD JSON files
        
    Returns:
        Configured EODHDProvider instance
    """
    config = EODHDConfig()
    if data_dir:
        config.data_dir = Path(data_dir)
    return EODHDProvider(config)


if __name__ == "__main__":
    # Test the provider
    provider = get_eodhd_provider()
    
    print("Parsing EODHD data...")
    df = provider.parse_all_tickers()
    
    print(f"\nTotal records: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Save to parquet
    output_path = provider.save_parquet(df)
    print(f"\nSaved to: {output_path}")
    
    # Test point-in-time lookup
    fund = provider.get_fundamentals("RELIANCE", as_of_date="2024-06-30")
    if fund:
        print(f"\nRELIANCE fundamentals as of 2024-06-30:")
        print(f"  Revenue: {fund.get('revenue')}")
        print(f"  Net Income: {fund.get('net_income')}")
        print(f"  ROE: {fund.get('roe')}")
        print(f"  P/E: {fund.get('pe_ratio')}")
