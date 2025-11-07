import os
import sys
import warnings
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from functools import lru_cache
from typing import List

import pandas as pd


def _read_csv(path: str, usecols: List[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )


def _to_snake(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@lru_cache(maxsize=1)
def load_bus(path: str = "dataset/cleaned_bus.csv.csv") -> pd.DataFrame:
    df = _read_csv(path)
    df = _to_snake(df)
    # normalize fields
    for c in ("source", "destination", "bus_type"):
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    if "price" in df:
        df["price"] = (
            df["price"].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)").fillna("0").astype(int)
        )
    if "rating" in df:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    return df


@lru_cache(maxsize=1)
def load_flights(path: str = "dataset/flights.csv") -> pd.DataFrame:
    usecols = ["airline", "time_taken", "price", "class", "from", "to", "dep_time"]
    df = _read_csv(path, usecols=None)  # schema issues; read all and then select
    df = _to_snake(df)
    cols = [c for c in usecols if c in df.columns]
    df = df[cols]
    for c in ("from", "to", "airline", "class"):
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    if "price" in df:
        df["price"] = (
            df["price"].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)").fillna("0").astype(int)
        )
    if "time_taken" in df:
        # keep raw; parser in utils
        df["time_taken"] = df["time_taken"].astype(str).str.strip()
    if "dep_time" in df:
        df["dep_time"] = df["dep_time"].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def load_hotels(path: str = "dataset/hotel pricing.csv") -> pd.DataFrame:
    df = _read_csv(path)
    df = _to_snake(df)
    if "city" in df:
        df["city"] = df["city"].astype(str).str.strip()
    # price column may be named differently; harmonize
    price_col = "price_per_night_inr" if "price_per_night_inr" in df.columns else "price_per_night"
    if price_col in df:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0).astype(int)
    if "rating" in df:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    return df


@lru_cache(maxsize=1)
def load_attractions(path: str = "dataset/india_attractions.csv") -> pd.DataFrame:
    df = _read_csv(path)
    df = _to_snake(df)
    if "city" in df:
        df["city"] = df["city"].astype(str).str.strip()
    return df


