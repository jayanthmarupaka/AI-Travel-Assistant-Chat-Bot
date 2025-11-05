import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz


def normalize_message(msg: str) -> str:
    return msg.strip()


def format_currency(amount: int) -> str:
    return f"₹{amount:,}".replace(",", ",")


def analyze_sentiment(text: str) -> str:
    text = text.lower()
    positive = ["hi", "hello", "hey", "good", "great", "thanks", "thank you"]
    negative = ["bad", "annoyed", "angry", "worst"]
    if any(w in text for w in negative):
        return "negative"
    if any(w in text for w in positive):
        return "positive"
    return "neutral"


def detect_intent(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(hi|hello|hey|good\s*(morning|evening|night))\b", t):
        return "greeting"
    if "itinerary" in t or "iternary" in t or re.search(r"\bplan\b.*\bday", t):
        return "itinerary"
    if "flight" in t or "airline" in t:
        return "flight"
    if "bus" in t or "sleeper" in t or "coach" in t:
        return "bus"
    if "hotel" in t or "stay" in t or "lodge" in t:
        return "hotel"
    if "place" in t or "visit" in t or "things to do" in t:
        return "attractions"
    return "unknown"


def parse_budget(text: str) -> Optional[int]:
    m = re.search(r"(?:under|upto|up to|budget)\s*₹?\s*([\d,]+)", text, flags=re.I)
    if not m:
        m = re.search(r"₹?\s*([\d,]+)\s*(?:budget|per night|a night)", text, flags=re.I)
    if not m:
        m = re.search(r"₹?\s*([\d,]+)", text)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def parse_time_to_minutes(s: str) -> int:
    h, m = 0, 0
    mh = re.search(r"(\d+)h", s)
    mm = re.search(r"(\d+)m", s)
    if mh:
        h = int(mh.group(1))
    if mm:
        m = int(mm.group(1))
    return h * 60 + m


@dataclass
class RouteQuery:
    source: Optional[str]
    destination: Optional[str]
    budget: Optional[int]


@dataclass
class HotelQuery:
    city: Optional[str]
    budget: Optional[int]


@dataclass
class ItineraryQuery:
    num_days: int
    source: Optional[str]
    destination: Optional[str]
    budget: Optional[int] = None


def extract_bus_params_gemini(user_msg: str, api_key: str, model_name: str = "gemini-2.5-flash") -> RouteQuery:
    """Extract bus query parameters using Gemini."""
    from services.gemini_client import GeminiClient
    from config import PROMPT_EXTRACT_BUS_PARAMS
    
    client = GeminiClient(api_key, model_name)
    prompt = PROMPT_EXTRACT_BUS_PARAMS.format(user_message=user_msg)
    result = client.extract_json(prompt, temperature=0.0)
    
    source = result.get("source", "").strip() if result.get("source") else None
    destination = result.get("destination", "").strip() if result.get("destination") else None
    budget = None
    if result.get("budget") is not None:
        try:
            budget = int(result.get("budget"))
        except (ValueError, TypeError):
            # Fallback to regex parsing
            budget = parse_budget(user_msg)
    else:
        budget = parse_budget(user_msg)
    
    return RouteQuery(source=source, destination=destination, budget=budget)


def extract_flight_params_gemini(user_msg: str, api_key: str, model_name: str = "gemini-2.5-flash") -> RouteQuery:
    """Extract flight query parameters using Gemini."""
    from services.gemini_client import GeminiClient
    from config import PROMPT_EXTRACT_FLIGHT_PARAMS
    
    client = GeminiClient(api_key, model_name)
    prompt = PROMPT_EXTRACT_FLIGHT_PARAMS.format(user_message=user_msg)
    result = client.extract_json(prompt, temperature=0.0)
    
    source = result.get("source", "").strip() if result.get("source") else None
    destination = result.get("destination", "").strip() if result.get("destination") else None
    budget = None
    if result.get("budget") is not None:
        try:
            budget = int(result.get("budget"))
        except (ValueError, TypeError):
            budget = parse_budget(user_msg)
    else:
        budget = parse_budget(user_msg)
    
    return RouteQuery(source=source, destination=destination, budget=budget)


def extract_hotel_params_gemini(user_msg: str, api_key: str, model_name: str = "gemini-2.5-flash") -> HotelQuery:
    """Extract hotel query parameters using Gemini."""
    from services.gemini_client import GeminiClient
    from config import PROMPT_EXTRACT_HOTEL_PARAMS
    
    client = GeminiClient(api_key, model_name)
    prompt = PROMPT_EXTRACT_HOTEL_PARAMS.format(user_message=user_msg)
    result = client.extract_json(prompt, temperature=0.0)
    
    city = result.get("city", "").strip() if result.get("city") else None
    budget = None
    if result.get("budget") is not None:
        try:
            budget = int(result.get("budget"))
        except (ValueError, TypeError):
            budget = parse_budget(user_msg)
    else:
        budget = parse_budget(user_msg)
    
    return HotelQuery(city=city, budget=budget)


def extract_attraction_params_gemini(user_msg: str, api_key: str, model_name: str = "gemini-2.5-flash") -> Optional[str]:
    """Extract city for attractions query using Gemini."""
    from services.gemini_client import GeminiClient
    from config import PROMPT_EXTRACT_ATTRACTION_PARAMS
    
    client = GeminiClient(api_key, model_name)
    prompt = PROMPT_EXTRACT_ATTRACTION_PARAMS.format(user_message=user_msg)
    result = client.extract_json(prompt, temperature=0.0)
    
    city = result.get("city", "").strip() if result.get("city") else None
    if not city:
        # Fallback to regex parsing
        city = extract_city_only(user_msg)
    return city


def extract_itinerary_params_gemini(user_msg: str, api_key: str, model_name: str = "gemini-2.5-flash") -> ItineraryQuery:
    """Extract itinerary query parameters using Gemini."""
    from services.gemini_client import GeminiClient
    from config import PROMPT_EXTRACT_ITINERARY_PARAMS
    
    client = GeminiClient(api_key, model_name)
    prompt = PROMPT_EXTRACT_ITINERARY_PARAMS.format(user_message=user_msg)
    result = client.extract_json(prompt, temperature=0.0)
    
    source = result.get("source", "").strip() if result.get("source") else None
    destination = result.get("destination", "").strip() if result.get("destination") else None
    
    num_days = 3
    if result.get("num_days") is not None:
        try:
            num_days = int(result.get("num_days"))
        except (ValueError, TypeError):
            # Fallback to regex
            m_days = re.search(r"(\d+)\s*-?\s*day", user_msg, flags=re.I)
            if m_days:
                num_days = int(m_days.group(1))
    else:
        m_days = re.search(r"(\d+)\s*-?\s*day", user_msg, flags=re.I)
        if m_days:
            num_days = int(m_days.group(1))
    
    budget = None
    if result.get("budget") is not None:
        try:
            budget = int(result.get("budget"))
        except (ValueError, TypeError):
            budget = parse_budget(user_msg)
    else:
        budget = parse_budget(user_msg)
    
    # Add budget to ItineraryQuery dataclass if needed
    return ItineraryQuery(num_days=num_days, source=source, destination=destination, budget=budget)


def canonicalize_city(city: str) -> str:
    return (city or "").strip().title()


def fuzzy_city_match(a: str, b: str, threshold: int = 85) -> bool:
    if not a or not b:
        return False
    a_norm, b_norm = canonicalize_city(a), canonicalize_city(b)
    score = max(
        fuzz.ratio(a_norm, b_norm),
        fuzz.token_sort_ratio(a_norm, b_norm),
    )
    return score >= threshold


def extract_bus_query(text: str) -> RouteQuery:
    budget = parse_budget(text)
    # Non-greedy capture for cities and stop at common delimiters
    # Examples handled: "from A to B", "from A to B under 5000"
    patterns = [
        r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?=\s*(?:under|within|budget|for|,|\.|$))",
        r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)$",
    ]
    src, dst = None, None
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            src, dst = m.group(1).strip(), m.group(2).strip()
            break
    return RouteQuery(src, dst, budget)


def extract_flight_query(text: str) -> RouteQuery:
    return extract_bus_query(text)


def extract_hotel_query(text: str) -> HotelQuery:
    budget = parse_budget(text)
    # Capture city after "in" but stop at common delimiters like "under/budget/within/for" or punctuation/end
    patterns = [
        r"in\s+([a-zA-Z\s]+?)(?=\s*(?:under|within|budget|for|,|\.|$))",
        r"in\s+([a-zA-Z\s]+)$",
    ]
    city = None
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            city = m.group(1).strip()
            break
    return HotelQuery(city, budget)


def extract_city_only(text: str) -> Optional[str]:
    m = re.search(r"in\s+([a-zA-Z\s]+)", text, flags=re.I)
    if m:
        return m.group(1).strip()
    # fallback: last word heuristic
    tokens = re.findall(r"[a-zA-Z]+", text)
    return tokens[-1].title() if tokens else None


def extract_itinerary_query(text: str) -> ItineraryQuery:
    days = 3
    m_days = re.search(r"(\d+)\s*-?\s*day", text, flags=re.I)
    if m_days:
        days = int(m_days.group(1))
    m = re.search(r"from\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)", text, flags=re.I)
    src, dst = (m.group(1).strip(), m.group(2).strip()) if m else (None, extract_city_only(text))
    return ItineraryQuery(days, src, dst)


