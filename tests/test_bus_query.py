import os
import sys
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.Query_Extraction_service import extract_bus_params_gemini
from services.Query_Response_Service import handle_bus_query
load_dotenv()

def test_handle_bus_query():
     user_query="please give buses under 10000 from Agra to delhi"
     api_key = os.getenv("GEMINI_API_KEY")
     result = extract_bus_params_gemini(
          user_query,
          api_key=api_key,
          model_name="gemini-2.5-flash"
     )
     print(result.destination)
     print(result.source)
     print(result.budget)

def test_handle_query():
     user_query="please give buses under 10000 from Agra to delhi"
     api_key = os.getenv("GEMINI_API_KEY")
     response = handle_bus_query(
          user_query,
          api_key=api_key,
          fuzzy=True,
          model_name="gemini-2.5-flash"
     )
     print("Bus Query Response:\n", response)


if __name__ == "__main__":
     test_handle_query()
     test_handle_bus_query()