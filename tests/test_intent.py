import os
import sys
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from services.Query_Response_Service import classify_intent

def test_classify_intent():

    query ="my friends plan a trip to Delhi they actually live in Agra,they want to see places in agra ,please suggest some good attractions"
    api_key = os.getenv("GEMINI_API_KEY", "")
    response = classify_intent(
        query,
        api_key=api_key
    )
    print("Classified Intent:", response)

if __name__ == "__main__":
    test_classify_intent()