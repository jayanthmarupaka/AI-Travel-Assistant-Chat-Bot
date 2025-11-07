import os
import sys
import warnings
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from typing import Optional

import pandas as pd

from config import (
    PROMPT_GREETING,
    PROMPT_INTENT,
    PROMPT_BUS,
    PROMPT_FLIGHT,
    PROMPT_HOTEL,
    PROMPT_ATTRACTIONS,
    PROMPT_ITINERARY,
    FALLBACK_BUS,
    FALLBACK_FLIGHT,
    FALLBACK_HOTEL,
    FALLBACK_ATTRACTIONS,
    TOP_K,
    MODEL_NAME,
)

from services.Retrieval_Service import Query, retrieve_buses, retrieve_flights, retrieve_hotels, retrieve_attractions
from services.Gemini_Service import GeminiClient
from services.Query_Extraction_service import (
    extract_bus_params_gemini,
    extract_flight_params_gemini,
    extract_hotel_params_gemini,
    extract_attraction_params_gemini,
    extract_itinerary_params_gemini,
    analyze_sentiment,
    format_currency,
    detect_intent,
)

def classify_intent(user_msg: str, api_key: str, model_name: Optional[str] = None) -> str:
    client = GeminiClient(api_key, model_name or MODEL_NAME)
    label = client.generate(PROMPT_INTENT.format(user_message=user_msg), temperature=0.0, max_output_tokens=100)
    label = (label or "").strip().split()[0].lower()
    if label in {"greeting","bus","flight","hotel","attractions","itinerary","unknown"}:
        return label
    # Fallback to lightweight local classifier if model returns empty/blocked
    return detect_intent(user_msg)


def _rows_to_bulleted_text(df: pd.DataFrame, cols: list[str]) -> str:
    lines: list[str] = []
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            val = row.get(c, "")
            if c == "price" and isinstance(val, (int, float)):
                val = format_currency(int(val))
            parts.append(f"{c}: {val}")
        lines.append(" - " + "; ".join(parts))
    return "\n".join(lines) if lines else ""


def handle_greeting(user_msg: str, api_key: str, model_name: Optional[str] = None) -> str:
    sentiment = analyze_sentiment(user_msg)
    client = GeminiClient(api_key, model_name or MODEL_NAME)
    prompt = PROMPT_GREETING.format(sentiment=sentiment, user_message=user_msg)
    return client.generate(prompt)


def handle_bus_query(user_msg: str, api_key: str, fuzzy: bool, model_name: Optional[str] = None) -> str:
    q = extract_bus_params_gemini(user_msg, api_key, model_name or MODEL_NAME)
    df = retrieve_buses(Query(source=q.source, destination=q.destination, budget=q.budget), fuzzy=fuzzy, top_k=TOP_K)
    if df.empty:
        return FALLBACK_BUS.format(source=q.source or "?", destination=q.destination or "?", budget=q.budget or "?")
    # include departure_time if present from cleaned_bus.csv
    bus_cols = ["source", "destination", "bus_type", "departure_time", "travel_duration", "price", "rating"]
    context_rows = _rows_to_bulleted_text(df, [c for c in bus_cols if c in df.columns])
    client = GeminiClient(api_key, model_name or MODEL_NAME)
    prompt = PROMPT_BUS.format(k=TOP_K, budget=q.budget or "?", context_rows=context_rows, user_question=user_msg)
    return client.generate(prompt)


def handle_flight_query(user_msg: str, api_key: str, fuzzy: bool, model_name: Optional[str] = None) -> str:
    q = extract_flight_params_gemini(user_msg, api_key, model_name or MODEL_NAME)
    df = retrieve_flights(Query(source=q.source, destination=q.destination, budget=q.budget), fuzzy=fuzzy, top_k=TOP_K)
    if df.empty:
        return FALLBACK_FLIGHT.format(source=q.source or "?", destination=q.destination or "?", budget=q.budget or "?")
    # include dep_time if present from flights.csv
    flight_cols = ["from", "to", "airline", "class", "dep_time", "time_taken", "price"]
    context_rows = _rows_to_bulleted_text(df, [c for c in flight_cols if c in df.columns])
    client = GeminiClient(api_key, model_name or MODEL_NAME)
    prompt = PROMPT_FLIGHT.format(k=TOP_K, budget=q.budget or "?", context_rows=context_rows, user_question=user_msg)
    return client.generate(prompt)


def handle_hotel_query(user_msg: str, api_key: str, fuzzy: bool, model_name: Optional[str] = None) -> str:
    q = extract_hotel_params_gemini(user_msg, api_key, model_name or MODEL_NAME)
    df, price_col = retrieve_hotels(Query(city=q.city, budget=q.budget), fuzzy=fuzzy, top_k=TOP_K)
    if df.empty:
        return FALLBACK_HOTEL.format(city=q.city or "?", budget=q.budget or "?")
    cols = ["city", "hotel_name", price_col, "rating"]
    cols = [c for c in cols if c in df.columns]
    # rename price col in display
    disp_df = df.copy()
    if price_col in disp_df.columns:
        disp_df.rename(columns={price_col: "price_per_night"}, inplace=True)
    context_rows = _rows_to_bulleted_text(disp_df, [c for c in ["city", "hotel_name", "price_per_night", "rating"] if c in disp_df.columns])
    client = GeminiClient(api_key, model_name or MODEL_NAME)
    prompt = PROMPT_HOTEL.format(k=TOP_K, budget=q.budget or "?", context_rows=context_rows, user_question=user_msg)
    return client.generate(prompt)


def handle_attractions_query(user_msg: str, api_key: str, fuzzy: bool, model_name: Optional[str] = None) -> str:
    city = extract_attraction_params_gemini(user_msg, api_key, model_name or MODEL_NAME)
    df = retrieve_attractions(Query(city=city), fuzzy=fuzzy, top_k=TOP_K)
    if df.empty:
        return FALLBACK_ATTRACTIONS.format(city=city or "?")
    take_cols = [c for c in ["city", "category", "attraction", "description", "activities", "best_time"] if c in df.columns]
    context_rows = _rows_to_bulleted_text(df, take_cols)
    client = GeminiClient(api_key, model_name or MODEL_NAME)
    prompt = PROMPT_ATTRACTIONS.format(context_rows=context_rows, user_question=user_msg)
    return client.generate(prompt)


def handle_itinerary_query(user_msg: str, api_key: str, fuzzy: bool, model_name: Optional[str] = None) -> str:
    it = extract_itinerary_params_gemini(user_msg, api_key, model_name or MODEL_NAME)
    
    # Calculate sub-budgets from total budget: 40% travel, 40% hotels, 20% activities
    total_budget = it.budget or 50000  # Default to 50000 if not specified
    travel_budget = int(total_budget * 0.4)  # 40% for outbound travel
    hotel_budget = int(total_budget * 0.4)  # 40% for hotels (per night)
    # 20% for activities is implicit (no explicit budget filtering for attractions)
    
    # Outbound travel options (from source to destination)
    bus_df = retrieve_buses(Query(source=it.source, destination=it.destination, budget=travel_budget), fuzzy=fuzzy, top_k=TOP_K)
    flight_df = retrieve_flights(Query(source=it.source, destination=it.destination, budget=travel_budget), fuzzy=fuzzy, top_k=TOP_K)
    
    # Hotels in destination
    hotel_df, price_col = retrieve_hotels(Query(city=it.destination, budget=hotel_budget), fuzzy=fuzzy, top_k=TOP_K)
    
    # Random attractions for each day (one per day)
    pool_df = retrieve_attractions(Query(city=it.destination), fuzzy=fuzzy, top_k=20)  # Get larger pool for randomness
    if not pool_df.empty and len(pool_df) >= it.num_days:
        # Randomly sample num_days attractions
        attr_df = pool_df.sample(n=it.num_days, random_state=None).reset_index(drop=True)
    elif not pool_df.empty:
        attr_df = pool_df.sample(n=len(pool_df), random_state=None).reset_index(drop=True)
    else:
        attr_df = pool_df
    
    # Return journey options (from destination back to source)
    return_bus_df = retrieve_buses(Query(source=it.destination, destination=it.source, budget=travel_budget), fuzzy=fuzzy, top_k=TOP_K)
    return_flight_df = retrieve_flights(Query(source=it.destination, destination=it.source, budget=travel_budget), fuzzy=fuzzy, top_k=TOP_K)

    # Format all data for the prompt
    bus_rows = _rows_to_bulleted_text(bus_df, [c for c in ["source","destination","bus_type","departure_time","travel_duration","price","rating"] if c in bus_df.columns]) if not bus_df.empty else "(no buses found)"
    flight_rows = _rows_to_bulleted_text(flight_df, [c for c in ["from","to","airline","class","dep_time","time_taken","price"] if c in flight_df.columns]) if not flight_df.empty else "(no flights found)"
    
    hotels_disp = hotel_df.copy()
    if not hotels_disp.empty and price_col in hotels_disp.columns:
        hotels_disp.rename(columns={price_col: "price_per_night"}, inplace=True)
    hotel_rows = _rows_to_bulleted_text(hotels_disp, [c for c in ["city", "hotel_name", "price_per_night", "rating"] if c in hotels_disp.columns]) if not hotel_df.empty else "(no hotels found)"
    
    attr_rows = _rows_to_bulleted_text(attr_df, [c for c in ["attraction", "category", "description", "activities"] if c in attr_df.columns]) if not attr_df.empty else "(no attractions found)"
    
    return_bus_rows = _rows_to_bulleted_text(return_bus_df, [c for c in ["source","destination","bus_type","departure_time","travel_duration","price","rating"] if c in return_bus_df.columns]) if not return_bus_df.empty else "(no return buses found)"
    return_flight_rows = _rows_to_bulleted_text(return_flight_df, [c for c in ["from","to","airline","class","dep_time","time_taken","price"] if c in return_flight_df.columns]) if not return_flight_df.empty else "(no return flights found)"

    client = GeminiClient(api_key, model_name or MODEL_NAME)
    prompt = PROMPT_ITINERARY.format(
        num_days=it.num_days,
        destination=it.destination or "?",
        source=it.source or "?",
        budget=total_budget,
        bus_rows=bus_rows,
        flight_rows=flight_rows,
        hotel_rows=hotel_rows,
        attraction_rows=attr_rows,
        return_bus_rows=return_bus_rows,
        return_flight_rows=return_flight_rows,
        user_question=user_msg,
    )
    return client.generate(prompt)


