import argparse
import os
import sys

import pandas as pd

# Ensure project root on sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.loaders import load_flights
from retrieval.retriever import retrieve_flights, Query
from utils.text import parse_time_to_minutes


def diagnose_flights(source: str, destination: str, budget: int, fuzzy: bool) -> None:
    print("=== Flights dataset diagnostics ===")
    df = load_flights()
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    # Basic cleanliness checks
    for col in ["from", "to", "price", "time_taken", "airline", "class"]:
        miss = df[col].isna().sum() if col in df.columns else "(missing col)"
        print(f"NA in {col}: {miss}")

    # Unique city samples
    if "from" in df and "to" in df:
        print("Unique 'from' sample:", sorted(df["from"].dropna().unique())[:10])
        print("Unique 'to' sample:", sorted(df["to"].dropna().unique())[:10])

    # Price sanity
    if "price" in df:
        print("Price min/max:", int(df["price"].min()), int(df["price"].max()))

    # Time parsing sanity (on a small sample)
    if "time_taken" in df:
        sample = df["time_taken"].dropna().head(5).tolist()
        mins = [parse_time_to_minutes(s) for s in sample]
        print("Sample time_taken:", sample)
        print("Parsed minutes:", mins)

    print("\n=== Retrieval run ===")
    q = Query(source=source, destination=destination, budget=budget)
    top = retrieve_flights(q, fuzzy=fuzzy, top_k=5)
    if top.empty:
        print("No results returned by retrieve_flights().")
        # Step-by-step narrowing to see where it drops to zero
        df0 = load_flights().copy()
        df1 = df0[df0["from"].str.contains(source.split()[0], case=False, na=False)] if "from" in df0 else df0
        df2 = df1[df1["to"].str.contains(destination.split()[0], case=False, na=False)] if "to" in df1 else df1
        df3 = df2[df2["price"] <= budget] if "price" in df2 else df2
        print(f"Rows after from filter: {len(df1)}")
        print(f"Rows after to filter:   {len(df2)}")
        print(f"Rows after budget:      {len(df3)}")
        print("Example rows after budget:")
        print(df3.head(10).to_string(index=False))
    else:
        print("Top results (up to 5):")
        print(top.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose flight retrieval pipeline")
    parser.add_argument("--source", default="Hyderabad")
    parser.add_argument("--destination", default="Mumbai")
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--no-fuzzy", action="store_true")
    args = parser.parse_args()

    fuzzy = not args.no_fuzzy
    diagnose_flights(args.source, args.destination, args.budget, fuzzy)


if __name__ == "__main__":
    main()


