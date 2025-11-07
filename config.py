MODEL_NAME = "gemini-2.5-flash"
TOP_K = 5
FUZZY_THRESHOLD = 85

PROMPT_INTENT = (
    "You are an intent classifier for a travel assistant. Classify the user's message "
    "into exactly one of these labels: greeting, bus, flight, hotel, attractions, itinerary, unknown. "
    "Rules: respond with ONLY the label, lowercase, no punctuation.\n"
    "Message: {user_message}"
)

PROMPT_GREETING = (
    "System: You are a friendly Indian travel assistant. If the user's text is a greeting, "
    "reply concisely with warmth. Adapt tone to sentiment ({sentiment}: positive/neutral/negative). "
    "Offer brief next-step help.\n"
    "User: {user_message}"
)

PROMPT_BUS = (
    "System: Use ONLY the bus options provided. Write a short creative paragraph that reads naturally, "
    "mentioning 3–5 options with bus type, departure_time if available, travel duration, price (₹), rating, and route. "
    "Close with a friendly travel tip.\n"
    "Context (top {k} buses under budget ₹{budget}):\n{context_rows}\n"
    "User: {user_question}"
)

PROMPT_FLIGHT = (
    "Greet the user warmly by mentioning their name and a positive mood\n"
    "System: Use ONLY the flights provided. Write a concise, engaging paragraph that mentions 3–5 options, "
    "including airline, class, dep_time if available, time_taken, price (₹), and route. Keep the order as given.\n"
    "Context (top {k} flights under budget ₹{budget}):\n{context_rows}\n"
    "User: {user_question}"
)

PROMPT_HOTEL = (
    "System: Recommend hotels strictly from the list. Write a short narrative describing 3–5 good fits with "
    "hotel_name, price per night (₹), rating, and city. End with a brief note about dynamic pricing.\n"
    "Context (top {k} hotels under budget ₹{budget}):\n{context_rows}\n"
    "User: {user_question}"
)

PROMPT_ATTRACTIONS = (
    "System: Suggest places to visit in the city using only the items below. Write a lively paragraph highlighting up to 5 spots: "
    "attraction, category, a one-line description, and 1–2 activities. Mention best_time if available.\n"
    "Context:\n{context_rows}\n"
    "User: {user_question}"
)

PROMPT_ITINERARY = (
    "System: Create a practical, inspiring {num_days}-day itinerary for {destination}, starting from {source} with a total budget of ₹{budget}. "
    "Use only the provided travel options, hotels, and attractions. Structure your response as follows:\n\n"
    "1. OUTBOUND TRAVEL: Show both bus and flight options from {source} to {destination} within budget\n"
    "2. HOTELS: Recommend hotels in {destination} within budget\n"
    "3. DAY-WISE ITINERARY: Plan day 1, day 2, day 3 (etc.) with one attraction per day from the provided list\n"
    "4. RETURN JOURNEY: Show bus and flight options from {destination} back to {source} within budget\n\n"
    "Outbound travel options (bus):\n{bus_rows}\n"
    "Outbound travel options (flight):\n{flight_rows}\n"
    "Hotels:\n{hotel_rows}\n"
    "Attractions (select randomly, one per day):\n{attraction_rows}\n"
    "Return journey buses:\n{return_bus_rows}\n"
    "Return journey flights:\n{return_flight_rows}\n"
    "User: {user_question}"
)

PROMPT_EXTRACT_BUS_PARAMS = (
    "Extract parameters from the user's query about bus travel. "
    "Return ONLY a valid JSON object with these exact keys: source, destination, budget. "
    "If a parameter is not mentioned, use null for that key. "
    "For budget, extract the numeric value (remove currency symbols and commas). "
    "Return only the JSON, no additional text or explanation.\n\n"
    "User query: {user_message}\n\n"
    "JSON:"
)

PROMPT_EXTRACT_FLIGHT_PARAMS = (
    "Extract parameters from the user's query about flight travel. "
    "Return ONLY a valid JSON object with these exact keys: source, destination, budget. "
    "If a parameter is not mentioned, use null for that key. "
    "For budget, extract the numeric value (remove currency symbols and commas). "
    "Return only the JSON, no additional text or explanation.\n\n"
    "User query: {user_message}\n\n"
    "JSON:"
)

PROMPT_EXTRACT_HOTEL_PARAMS = (
    "Extract parameters from the user's query about hotels. "
    "Return ONLY a valid JSON object with these exact keys: city, budget. "
    "If a parameter is not mentioned, use null for that key. "
    "For budget, extract the numeric value (remove currency symbols and commas). consider value greater than 1000"
    "If no budget is mentioned, return 2500 for the budget key."
    "Return only the JSON, no additional text or explanation.\n\n"
    "User query: {user_message}\n\n"
    "JSON:"
)

PROMPT_EXTRACT_ATTRACTION_PARAMS = (
    "Extract parameters from the user's query about attractions or places to visit. "
    "Return ONLY a valid JSON object with this exact key: city. "
    "If city is not mentioned, use null. "
    "Return only the JSON, no additional text or explanation.\n\n"
    "User query: {user_message}\n\n"
    "JSON:"
)

PROMPT_EXTRACT_ITINERARY_PARAMS = (
    "Extract parameters from the user's query about planning an itinerary. "
    "Return ONLY a valid JSON object with these exact keys: source, destination, budget, num_days. "
    "If a parameter is not mentioned, use null for that key. "
    "For budget, extract the numeric value (remove currency symbols and commas). "
    "For num_days, extract the number of days (default to 3 if not mentioned). "
    "Return only the JSON, no additional text or explanation.\n\n"
    "User query: {user_message}\n\n"
    "JSON:"
)

FALLBACK_BUS = "Sorry, I couldn’t find buses for {source} → {destination} within ₹{budget}."
FALLBACK_FLIGHT = "Sorry, I couldn’t find flights for {source} → {destination} within ₹{budget}."
FALLBACK_HOTEL = "Sorry, I couldn’t find hotels in {city} within ₹{budget} per night."
FALLBACK_ATTRACTIONS = "Sorry, I couldn’t find attractions in {city}."
