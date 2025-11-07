import os
import sys
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.Query_Extraction_service import extract_bus_params_gemini

load_dotenv()

def test_handle_bus_query():
     user_query="my friend and I want to travel from Bangalore to Chennai by bus under 1500"
     api_key = os.getenv("GEMINI_API_KEY")
     result = extract_bus_params_gemini(
          user_query,
          api_key=api_key,
          model_name="gemini-2.5-flash"
     )
     print(result.destination)
     print(result.source)
     print(result.budget)


if __name__ == "__main__":
     test_handle_bus_query()