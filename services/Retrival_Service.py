import os
import sys
import warnings
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from services.CSV_Service import load_bus, load_flights, load_hotels, load_attractions
from services.Query_Extraction_service import (
    canonicalize_city,
    fuzzy_city_match,
    parse_budget,
    parse_time_to_minutes,
)


@dataclass
class Query:
    source: Optional[str] = None
    destination: Optional[str] = None
    city: Optional[str] = None
    budget: Optional[int] = None


def _apply_city_filters(df: pd.DataFrame, col: str, value: str, fuzzy: bool) -> pd.Series:
    if not fuzzy:
        return df[col].str.casefold() == canonicalize_city(value).casefold()
    return df[col].apply(lambda x: fuzzy_city_match(x, value))


def retrieve_buses(q: Query, fuzzy: bool, top_k: int = 5) -> pd.DataFrame:
    df = load_bus()
    if q.source:
        df = df[_apply_city_filters(df, "source", q.source, fuzzy)]
    if q.destination:
        df = df[_apply_city_filters(df, "destination", q.destination, fuzzy)]
    if q.budget is not None and "price" in df:
        df = df[df["price"] <= int(q.budget)]
    if df.empty:
        return df
    sort_cols = ["price"] + (["rating"] if "rating" in df.columns else [])
    ascending = [True] + ([False] if "rating" in df.columns else [])
    df = df.sort_values(by=sort_cols, ascending=ascending).head(top_k)
    return df


def retrieve_flights(q: Query, fuzzy: bool, top_k: int = 5) -> pd.DataFrame:
    df = load_flights()
    if q.source:
        df = df[_apply_city_filters(df, "from", q.source, fuzzy)]
    if q.destination:
        df = df[_apply_city_filters(df, "to", q.destination, fuzzy)]
    if q.budget is not None and "price" in df:
        df = df[df["price"] <= int(q.budget)]
    if df.empty:
        return df
    if "time_taken" in df:
        df = df.assign(_mins=df["time_taken"].map(parse_time_to_minutes))
    sort_cols = ["price"] + (["_mins"] if "_mins" in df.columns else [])
    ascending = [True] + ([True] if "_mins" in df.columns else [])
    df = df.sort_values(by=sort_cols, ascending=ascending).head(top_k)
    return df.drop(columns=[c for c in ["_mins"] if c in df.columns])


def retrieve_hotels(q: Query, fuzzy: bool, top_k: int = 5) -> Tuple[pd.DataFrame, str]:
    df = load_hotels()
    price_col = "price_per_night_inr" if "price_per_night_inr" in df.columns else "price_per_night"
    if q.city:
        df = df[_apply_city_filters(df, "city", q.city, fuzzy)]
    if q.budget is not None and price_col in df:
        df = df[df[price_col] <= int(q.budget)]
    if df.empty:
        return df, price_col
    sort_cols = [price_col] + (["rating"] if "rating" in df.columns else [])
    ascending = [True] + ([False] if "rating" in df.columns else [])
    df = df.sort_values(by=sort_cols, ascending=ascending).head(top_k)
    return df, price_col


def retrieve_attractions(q: Query, fuzzy: bool, top_k: int = 5) -> pd.DataFrame:
    df = load_attractions()
    if q.city:
        df = df[_apply_city_filters(df, "city", q.city, fuzzy)]
    if df.empty:
        return df
    return df.sample(n=min(top_k, len(df)))


