import os
import sys
import warnings
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

from services.Query_Response_Service import (
    handle_greeting,
    handle_bus_query,
    handle_flight_query,
    handle_hotel_query,
    handle_attractions_query,
    handle_itinerary_query,
)
from services.Query_Extraction_service import normalize_message
from services.Query_Response_Service import classify_intent
from config import MODEL_NAME


st.set_page_config(page_title="AI Travel Assistant", page_icon="🧭", layout="wide")
load_dotenv()


def ensure_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("Set GEMINI_API_KEY in your .env file to start.")
    return api_key


with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This AI Travel Assistant helps you plan trips across India using local datasets.

        What it can do:
        - Greet and assist based on your tone
        - Find buses and flights within a budget
        - Suggest hotels under a price cap
        - Recommend attractions in a city
        - Build a day‑wise itinerary

        How to use:
        - Type a question like "flights from Hyderabad to Mumbai under 10000".
        - Or "hotels in Hyderabad under 8000".
        - Or "plan a 3 day itinerary from Mumbai to Hyderabad".

        Configure your `GEMINI_API_KEY` in a `.env` file.
        """
    )
    st.divider()
    st.caption("Datasets are loaded from the local dataset/ folder.")


st.title("🧭 AI Travel Assistant")
st.caption("Powered by Gemini 1.5-flash with local CSV retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)


prompt = st.chat_input("Ask about buses, flights, hotels, attractions, or an itinerary…")
if prompt:
    api_key = ensure_api_key()
    if not api_key:
        st.stop()

    user_msg = normalize_message(prompt)
    # Immediately show the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(("user", prompt))
    intent = classify_intent(user_msg, api_key, MODEL_NAME)

    with st.chat_message("assistant"):
        # Anchor for sidebar jump
        anchor_id = f"resp-{len(st.session_state.messages)}"
        st.markdown(f"<a id='{anchor_id}'></a>", unsafe_allow_html=True)
        # Thinking placeholder (animated)
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("_Thinking…_")

        model_name = MODEL_NAME
        fuzzy = True  # always enabled
        if intent == "greeting":
            response = handle_greeting(user_msg, api_key, model_name)
        elif intent == "bus":
            response = handle_bus_query(user_msg, api_key, fuzzy, model_name)
        elif intent == "flight":
            response = handle_flight_query(user_msg, api_key, fuzzy, model_name)
        elif intent == "hotel":
            response = handle_hotel_query(user_msg, api_key, fuzzy, model_name)
        elif intent == "attractions":
            response = handle_attractions_query(user_msg, api_key, fuzzy, model_name)
        elif intent == "itinerary":
            response = handle_itinerary_query(user_msg, api_key, fuzzy, model_name)
        else:
            response = "I can help with buses, flights, hotels, attractions, or itineraries. Try asking with a city and optional budget."

        thinking_placeholder.markdown(response)
        st.session_state.messages.append(("assistant", response))

# Sidebar history links
with st.sidebar:
    st.subheader("History")
    links = []
    for i, (role, content) in enumerate(st.session_state.messages):
        if role == "user":
            anchor = f"#resp-{i}"
            label = content.strip()
            if len(label) > 60:
                label = label[:57] + "..."
            links.append(f"- [{label}]({anchor})")
    if links:
        st.markdown("\n".join(links))


