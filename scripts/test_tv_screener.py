"""
Test tradingview-screener library.
"""
import sys
import os

def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    # print(f"Sys path: {sys.path}")

    try:
        import tradingview_screener
        print(f"Successfully imported tradingview_screener from {tradingview_screener.__file__}")
        from tradingview_screener import Query, Column
    except ImportError as e:
        print(f"ImportError: {e}")
        return

    print("Testing tradingview-screener...")
    
    try:
        q = (Query()
             .select('name', 'close', 'price_earnings_ttm', 'return_on_equity_fq', 'debt_to_equity_fq', 'total_revenue_fq')
             .where(Column('exchange') == 'NSE')
             .where(Column('name') == 'RELIANCE')
             .limit(1))
             
        rows = q.get_scanner_data()
        
        if rows:
            print("Success!")
            print(rows[0])
        else:
            print("No data found for RELIANCE on NSE.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
