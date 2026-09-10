import duckdb

DB_PATH = "data/duckdb/market_data.duckdb"

def run_data_validation():
    con = duckdb.connect(DB_PATH)
    tables = ["nifty_spot_1m", "banknifty_spot_1m", "nifty_futures_daily", "nifty_futures_active_1m"]

    print("=" * 70)
    print("                MARKET DATA INTEGRITY VALIDATION")
    print("=" * 70)

    all_passed = True

    for table in tables:
        print(f"\n[+] Auditing table: '{table}'")

        # 1. Null & Zero/Negative Price Verification
        null_check = con.execute(f"""
            SELECT COUNT(*) FROM {table}
            WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
               OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
        """).fetchone()[0]

        # 2. OHLC Logical Invariant Violation Check
        ohlc_violations = con.execute(f"""
            SELECT COUNT(*) FROM {table}
            WHERE high < open OR high < close OR low > open OR low > close
        """).fetchone()[0]

        # 3. Duplicate Timestamp Check
        duplicates = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT timestamp, COUNT(*) as cnt
                FROM {table}
                GROUP BY timestamp
                HAVING cnt > 1
            )
        """).fetchone()[0]

        # 4. Strict Chronological Monotonicity Check
        out_of_order = con.execute(f"""
            WITH ranked AS (
                SELECT timestamp, LAG(timestamp) OVER (ORDER BY timestamp) as prev_ts
                FROM {table}
            )
            SELECT COUNT(*) FROM ranked WHERE timestamp <= prev_ts
        """).fetchone()[0]

        # 5. Extreme Returns / Data Artifact Check (|return| > 5% in 1 minute intraday)
        if "1m" in table:
            spikes = con.execute(f"""
                WITH returns AS (
                    SELECT timestamp, close,
                           ABS((close - LAG(close) OVER (PARTITION BY CAST(timestamp AS DATE) ORDER BY timestamp)) /
                               LAG(close) OVER (PARTITION BY CAST(timestamp AS DATE) ORDER BY timestamp)) as pct_change
                    FROM {table}
                )
                SELECT COUNT(*) FROM returns WHERE pct_change > 0.05
            """).fetchone()[0]
        else:
            spikes = 0

        # 6. Out-of-Session Timestamp Check (Outside 09:15 - 15:30 IST for intraday)
        if "1m" in table:
            out_of_hours = con.execute(f"""
                SELECT COUNT(*) FROM {table}
                WHERE strftime(timestamp, '%H:%M:%S') < '09:15:00'
                   OR strftime(timestamp, '%H:%M:%S') > '15:30:00'
            """).fetchone()[0]
        else:
            out_of_hours = 0

        # Reporting
        checks = [
            ("Null / Non-Positive Prices", null_check),
            ("OHLC Invariant Violations", ohlc_violations),
            ("Duplicate Timestamps", duplicates),
            ("Timestamp Order Violations", out_of_order),
            ("Extreme Return Spikes (>5% 1m)", spikes),
            ("Out-of-Market-Hours Records", out_of_hours)
        ]

        table_passed = True
        for check_name, count in checks:
            status = "PASS" if count == 0 else "FAIL"
            if count > 0:
                table_passed = False
                all_passed = False
            print(f"  ├─ [{status}] {check_name}: {count:,} errors")

        if table_passed:
            print(f"  └─ STATUS: TABLE CLEARED")
        else:
            print(f"  └─ STATUS: DATA CORRUPTION DETECTED")

    con.close()

    print("\n" + "=" * 70)
    if all_passed:
        print("[+] AUDIT COMPLETE: ALL DATASETS PASSED INTEGRITY CHECKS.")
    else:
        print("[!] AUDIT FAILED: FIX FLAG ERRORS BEFORE TRAINING.")
    print("=" * 70)

if __name__ == "__main__":
    run_data_validation()
